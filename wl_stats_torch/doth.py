"""
Difference-of-top-hats wavelet transform in PyTorch.

This module provides a GPU-accelerated, differentiable implementation of
multi-scale difference-of-top-hats filtering in Fourier space.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .starlet import fft_convolve2d


class DifferenceOfTopHats2D(nn.Module):
    """
    2D difference-of-top-hats transform using Fourier-space top-hat filters.

    For each detail scale j, the coefficient map is:
        detail_j = smooth(2 * R_j) - smooth(R_j)
    with dyadic radii:
        R_j = base_radius * 2**j

    The final channel is the coarsest smoothed map, so output has `n_scales`
    channels (n_scales - 1 details + 1 coarse), matching the Starlet API shape.
    """

    def __init__(
        self,
        n_scales: int = 5,
        base_radius: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if n_scales < 2:
            raise ValueError(f"n_scales must be at least 2, got {n_scales}")
        if base_radius <= 0:
            raise ValueError(f"base_radius must be > 0, got {base_radius}")

        self.n_scales = n_scales
        self.base_radius = float(base_radius)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.has_coarse = True

        self._window_cache: Dict[Tuple[int, int, str, int, torch.dtype], List[torch.Tensor]] = {}

        self.to(device=self.device, dtype=self.dtype)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move module metadata/cache to a new device and/or dtype."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self._window_cache.clear()
        return super().to(device=device, dtype=dtype)

    def _cache_key(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[int, int, str, int, torch.dtype]:
        device_index = -1 if device.index is None else device.index
        return (height, width, device.type, device_index, dtype)

    def _top_hat_window(self, k_radius: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        kr = k_radius * radius
        window = torch.ones_like(kr)

        nonzero = kr != 0
        if nonzero.any():
            window[nonzero] = 2.0 * torch.special.bessel_j1(kr[nonzero]) / kr[nonzero]

        return window

    def _get_windows(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        key = self._cache_key(height, width, device, dtype)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        k_radius = 2.0 * torch.pi * torch.sqrt(kx**2 + ky**2)

        radii = self.base_radius * (
            2.0 ** torch.arange(self.n_scales, device=device, dtype=dtype)
        )
        windows = [self._top_hat_window(k_radius, radius) for radius in radii]

        self._window_cache[key] = windows
        return windows

    def forward(
        self, x: torch.Tensor, return_coarse: bool = True, return_dict: bool = False
    ) -> torch.Tensor:
        """
        Apply the forward DoTH transform.

        Args:
            x: Input tensor of shape (B, C, H, W) or (H, W) or (C, H, W).
               For single-channel input, C should be 1.
            return_coarse: If True, include final coarse channel.
            return_dict: If True, return a dictionary with additional tensors.

        Returns:
            If return_dict is False:
                Tensor of shape (B, n_scales, H, W) if return_coarse=True,
                or (B, n_scales-1, H, W) if return_coarse=False.
            If return_dict is True:
                Dictionary with keys:
                    'coeffs', 'detail_scales', 'coarse_scale'
        """
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Input must be 2D, 3D, or 4D, got shape {original_shape}")

        if x.shape[1] != 1:
            raise ValueError(
                f"DifferenceOfTopHats2D requires single-channel input, got {x.shape[1]} channels."
            )

        x = x.to(self.device, dtype=self.dtype)
        _, _, height, width = x.shape

        windows = self._get_windows(height, width, x.device, x.dtype)
        x_fft = torch.fft.fft2(x, dim=(-2, -1))

        smoothed_maps = []
        for window in windows:
            smoothed_fft = x_fft * window.unsqueeze(0).unsqueeze(0)
            smoothed_map = torch.fft.ifft2(smoothed_fft, dim=(-2, -1)).real
            smoothed_maps.append(smoothed_map)

        detail_scales = [
            smoothed_maps[scale_idx + 1] - smoothed_maps[scale_idx]
            for scale_idx in range(self.n_scales - 1)
        ]

        if return_coarse:
            all_scales = detail_scales + [smoothed_maps[-1]]
        else:
            all_scales = detail_scales

        coeffs = torch.cat(all_scales, dim=1)

        if return_dict:
            return {"coeffs": coeffs, "detail_scales": detail_scales, "coarse_scale": smoothed_maps[-1]}

        return coeffs

    def get_scale_resolution(self, pixel_size_arcmin: float = 1.0) -> List[float]:
        """
        Get approximate effective resolution (arcmin) for each output scale.
        """
        return [
            self.base_radius * (2 ** (scale_idx + 1)) * pixel_size_arcmin
            for scale_idx in range(self.n_scales)
        ]

    def get_noise_levels(
        self, noise_sigma: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute noise levels for each DoTH coefficient via impulse-response propagation.

        Args:
            noise_sigma: Noise std map, shape (H, W), (B, H, W), or (B, 1, H, W).
            mask: Optional mask where 1 = observed, 0 = unobserved.

        Returns:
            Noise levels of shape (B, n_scales, H, W).
        """
        if noise_sigma.ndim == 2:
            noise_sigma = noise_sigma.unsqueeze(0).unsqueeze(0)
        elif noise_sigma.ndim == 3:
            noise_sigma = noise_sigma.unsqueeze(1)
        elif noise_sigma.ndim != 4:
            raise ValueError(
                f"noise_sigma must be 2D, 3D, or 4D, got shape {noise_sigma.shape}"
            )

        if noise_sigma.shape[1] != 1:
            raise ValueError(
                f"noise_sigma must have one channel, got shape {noise_sigma.shape}"
            )

        noise_sigma = noise_sigma.to(self.device, dtype=self.dtype)
        batch_size, _, height, width = noise_sigma.shape

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            elif mask.ndim == 3:
                if mask.shape[0] != batch_size:
                    raise ValueError("Batch size mismatch between noise_sigma and mask")
                mask = mask.unsqueeze(1)
            elif mask.ndim == 4:
                if mask.shape[0] != batch_size:
                    raise ValueError("Batch size mismatch between noise_sigma and mask")
            else:
                raise ValueError(f"mask must be 2D, 3D, or 4D, got shape {mask.shape}")

            mask = mask.to(self.device) != 0
            noise_sigma = noise_sigma.clone()
            for batch_idx in range(batch_size):
                valid = mask[batch_idx]
                max_noise = (
                    noise_sigma[batch_idx][valid].max()
                    if valid.any()
                    else noise_sigma[batch_idx].max()
                )
                noise_sigma[batch_idx][~valid] = max_noise

        variance_map = noise_sigma**2

        impulse = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        impulse[0, 0, height // 2, width // 2] = 1.0

        impulse_coeffs = self.forward(impulse, return_coarse=True)
        impulse_coeffs_squared = impulse_coeffs[0] ** 2

        variance_all = torch.zeros(
            batch_size, self.n_scales, height, width, device=self.device, dtype=self.dtype
        )

        for batch_idx in range(batch_size):
            variance_map_2d = variance_map[batch_idx, 0]
            for scale_idx in range(self.n_scales):
                kernel = impulse_coeffs_squared[scale_idx]
                var_scale = fft_convolve2d(variance_map_2d, kernel)
                variance_all[batch_idx, scale_idx] = torch.clamp(var_scale, min=0.0)

        return torch.sqrt(variance_all)

    def get_snr(
        self,
        image: torch.Tensor,
        noise_sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        keep_sign: bool = False,
    ) -> torch.Tensor:
        """
        Compute SNR maps for DoTH coefficients.
        """
        wavelet_coeffs = self.forward(image, return_coarse=True)
        noise_levels = self.get_noise_levels(noise_sigma, mask=mask)

        snr = torch.zeros_like(wavelet_coeffs)
        valid_mask = noise_levels != 0
        snr[valid_mask] = wavelet_coeffs[valid_mask] / noise_levels[valid_mask]

        if not keep_sign:
            snr = torch.abs(snr)

        return snr

    def extra_repr(self) -> str:
        return (
            f"n_scales={self.n_scales}, base_radius={self.base_radius}, device={self.device}"
        )
