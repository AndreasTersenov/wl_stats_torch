"""
2D Starlet (à trous wavelet) Transform in PyTorch

This module implements the 2D starlet transform using PyTorch for GPU acceleration.
The starlet transform is an isotropic undecimated wavelet transform using a B3-spline
scaling function.

References:
    Starck, J.-L., Fadili, J., & Murtagh, F. (2007).
    "The Undecimated Wavelet Decomposition and its Reconstruction"
    IEEE Transactions on Image Processing, 16(2), 297-309.
"""

from typing import Optional

import torch
import torch.nn as nn


def fft_convolve2d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform 2D convolution using FFT (equivalent to scipy.signal.fftconvolve with mode='same').

    This function stays entirely in PyTorch, avoiding CPU transfers and numpy conversions.
    It matches scipy's behavior for 'same' mode convolution.

    Args:
        signal: Input signal, shape (H, W)
        kernel: Convolution kernel, shape (H, W)

    Returns:
        Convolved result, shape (H, W), same size as input signal
    """
    # Get shapes
    s_h, s_w = signal.shape
    k_h, k_w = kernel.shape

    # For 'same' mode, we need to pad to avoid circular convolution artifacts
    # and then extract the center region
    pad_h = k_h - 1
    pad_w = k_w - 1

    # FFT size should be at least signal_size + kernel_size - 1 for linear convolution
    fft_h = s_h + pad_h
    fft_w = s_w + pad_w

    # Compute FFTs using rfft2 (optimized for real inputs)
    signal_fft = torch.fft.rfft2(signal, s=(fft_h, fft_w))
    kernel_fft = torch.fft.rfft2(kernel, s=(fft_h, fft_w))

    # Multiply in frequency domain
    result_fft = signal_fft * kernel_fft

    # Inverse FFT
    result = torch.fft.irfft2(result_fft, s=(fft_h, fft_w))

    # Extract 'same' mode result (center region matching signal size)
    # This matches scipy's centering behavior
    start_h = pad_h // 2
    start_w = pad_w // 2
    result_same = result[start_h : start_h + s_h, start_w : start_w + s_w]

    return result_same


class Starlet2D(nn.Module):
    """
    2D Starlet Transform (à trous algorithm) with B3-spline kernel.

    This implements an undecimated wavelet transform where each scale is computed
    by taking the difference between successive smoothed versions of the image.
    The smoothing uses a B3-spline kernel with increasing dilation (holes).

    The transform decomposes an image into multiple detail (wavelet) scales and
    a final coarse (approximation) scale.

    Attributes:
        n_scales (int): Total number of scales (including coarse scale)
        kernel_2d (torch.Tensor): The 2D B3-spline convolution kernel
        device (torch.device): Device for computation (cpu or cuda)

    Example:
        >>> starlet = Starlet2D(n_scales=5)
        >>> image = torch.randn(1, 1, 256, 256)  # (batch, channel, H, W)
        >>> coeffs = starlet(image)  # (batch, n_scales, H, W)
    """

    def __init__(
        self,
        n_scales: int = 5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the Starlet2D transform.

        Args:
            n_scales: Total number of scales (detail scales + 1 coarse scale).
                     For example, n_scales=5 gives 4 detail scales and 1 coarse scale.
            device: torch device for computation. If None, uses CPU.
            dtype: Data type for computations. Default: torch.float32 (PyTorch standard).
        """
        super(Starlet2D, self).__init__()

        if n_scales < 2:
            raise ValueError(f"n_scales must be at least 2, got {n_scales}")

        self.n_scales = n_scales
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        # Define the 1D B3-spline kernel coefficients
        # These are the coefficients for the B3-spline: [1/16, 1/4, 3/8, 1/4, 1/16]
        kernel_1d = torch.tensor(
            [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0],
            dtype=self.dtype,
            device=self.device,
        )

        # Create 2D kernel via outer product (separable filter)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)

        # Store as (out_channels, in_channels, H, W) for Conv2d
        # We use (1, 1, 5, 5) since we process single channel at a time
        self.register_buffer("kernel_2d", kernel_2d.unsqueeze(0).unsqueeze(0))

        # Pre-compute dilated kernels for each scale (optional optimization)
        self._setup_dilated_convolutions()

    def _setup_dilated_convolutions(self):
        """
        Pre-setup dilated convolution layers for each scale.
        This avoids dynamic kernel creation during forward pass.
        """
        self.convs = nn.ModuleList()

        for scale_idx in range(self.n_scales - 1):
            dilation = 2**scale_idx
            # Padding to maintain same size: (kernel_size - 1) * dilation / 2
            padding = 2 * dilation  # (5 - 1) / 2 * dilation = 2 * dilation

            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
                bias=False,
                padding_mode="reflect",
                dtype=self.dtype,  # Use the specified dtype
            )

            # Set the kernel weights and freeze them (non-trainable)
            with torch.no_grad():
                conv.weight.copy_(self.kernel_2d)
            conv.weight.requires_grad = False

            self.convs.append(conv)

        # Move all convolutions to the correct device
        self.to(self.device)

    def forward(
        self, x: torch.Tensor, return_coarse: bool = True, return_dict: bool = False
    ) -> torch.Tensor:
        """
        Apply the forward Starlet transform.

        Args:
            x: Input tensor of shape (B, C, H, W) or (H, W) or (C, H, W).
               For single-channel input, C should be 1.
            return_coarse: If True, include the final coarse scale in output.
            return_dict: If True, return a dictionary with additional info.

        Returns:
            If return_dict is False:
                Wavelet coefficients of shape (B, n_scales, H, W) if return_coarse=True,
                or (B, n_scales-1, H, W) if return_coarse=False.
            If return_dict is True:
                Dictionary with keys:
                    'coeffs': wavelet coefficients
                    'detail_scales': list of detail scale tensors
                    'coarse_scale': final coarse scale tensor

        Raises:
            ValueError: If input has more than 1 channel (excluding batch dimension)
        """
        # Handle different input shapes
        original_shape = x.shape
        if x.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            # Already (B, C, H, W)
            pass
        else:
            raise ValueError(f"Input must be 2D, 3D, or 4D, got shape {original_shape}")

        # Check channel dimension
        if x.shape[1] != 1:
            raise ValueError(
                f"Starlet2D requires single-channel input, got {x.shape[1]} channels. "
                "Process each channel separately if needed."
            )

        detail_scales = []
        coarse = x

        # Compute wavelet scales using à trous algorithm
        for scale_idx in range(self.n_scales - 1):
            # Smooth the current coarse approximation
            next_coarse = self.convs[scale_idx](coarse)

            # Detail scale = difference between successive approximations
            detail = coarse - next_coarse
            detail_scales.append(detail)

            # Update coarse for next iteration
            coarse = next_coarse

        # Combine all scales
        if return_coarse:
            all_scales = detail_scales + [coarse]
        else:
            all_scales = detail_scales

        # Stack along channel dimension: (B, n_scales, H, W)
        coeffs = torch.cat(all_scales, dim=1)

        if return_dict:
            return {"coeffs": coeffs, "detail_scales": detail_scales, "coarse_scale": coarse}

        return coeffs

    def reconstruct(self, wavelet_coeffs: torch.Tensor, gen2: bool = True) -> torch.Tensor:
        """
        Reconstruct image from wavelet coefficients.

        Args:
            wavelet_coeffs: Wavelet coefficients of shape (B, n_scales, H, W)
            gen2: If True, use second generation reconstruction (default).
                  If False, use first generation (simple sum).

        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        if wavelet_coeffs.shape[1] != self.n_scales:
            raise ValueError(f"Expected {self.n_scales} scales, got {wavelet_coeffs.shape[1]}")

        if not gen2:
            # First generation: simple sum of all scales
            return wavelet_coeffs.sum(dim=1, keepdim=True)

        # Second generation reconstruction: h' = h, g' = Dirac
        # Reconstruct from coarse to fine

        # Start with the coarse scale
        reconstructed = wavelet_coeffs[:, -1:, :, :]  # (B, 1, H, W)

        # Add detail scales from coarse to fine
        for scale_idx in range(self.n_scales - 2, -1, -1):
            # Smooth the current reconstruction
            smoothed = self.convs[scale_idx](reconstructed)

            # Add the detail scale
            detail = wavelet_coeffs[:, scale_idx : scale_idx + 1, :, :]
            reconstructed = smoothed + detail

        return reconstructed

    def get_scale_resolution(self, pixel_size_arcmin: float = 1.0) -> list:
        """
        Get the effective resolution (FWHM) of each wavelet scale in arcminutes.

        Args:
            pixel_size_arcmin: Size of one pixel in arcminutes

        Returns:
            List of resolutions for each scale (including coarse)
        """
        # Each scale j has effective resolution ~ 2^(j+1) pixels
        resolutions = [
            2 ** (scale_idx + 1) * pixel_size_arcmin for scale_idx in range(self.n_scales)
        ]
        return resolutions

    def get_noise_levels(
        self, noise_sigma: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute noise standard deviation for each wavelet coefficient.

        This uses the same approach as CosmoStat: compute the wavelet impulse response
        (by transforming a delta function), square it, and convolve with the variance map.

        Args:
            noise_sigma: Noise std deviation map, shape (H, W) or (B, 1, H, W)
            mask: Optional mask where 1 = observed, 0 = not observed

        Returns:
            Noise levels for each coefficient, shape (B, n_scales, H, W)
        """
        # Handle input shapes
        if noise_sigma.ndim == 2:
            noise_sigma = noise_sigma.unsqueeze(0).unsqueeze(0)
        elif noise_sigma.ndim == 3:
            noise_sigma = noise_sigma.unsqueeze(0)

        batch_size, _, height, width = noise_sigma.shape

        # If mask provided, set unobserved regions to maximum noise
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)

            max_noise = noise_sigma[mask == 1].max()
            noise_sigma = noise_sigma.clone()
            noise_sigma[mask == 0] = max_noise

        # Compute variance map
        variance_map = noise_sigma**2  # (B, 1, H, W)

        # Create impulse (delta function) at the center
        impulse = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        impulse[0, 0, height // 2, width // 2] = 1.0

        # Get the wavelet impulse response by transforming the delta function
        impulse_coeffs = self.forward(impulse, return_coarse=True)  # (1, n_scales, H, W)

        # Square the impulse response coefficients
        impulse_coeffs_squared = impulse_coeffs**2  # (1, n_scales, H, W)

        # For each scale, convolve the variance map with the squared impulse response
        # Use PyTorch FFT convolution to stay on GPU and avoid numpy conversion
        variance_coeffs = []

        # Extract the variance map for convolution (no need to convert to numpy)
        variance_map_2d = variance_map[0, 0, :, :]  # (H, W)

        for scale_idx in range(self.n_scales):
            # Get the squared impulse response for this scale
            kernel = impulse_coeffs_squared[0, scale_idx, :, :]  # (H, W)

            # Convolve using PyTorch FFT (equivalent to scipy's mode='same')
            var_scale = fft_convolve2d(variance_map_2d, kernel)

            # Clamp to avoid negative values due to numerical errors
            var_scale = torch.clamp(var_scale, min=0.0)

            # Replicate for batch dimension
            var_scale_batch = var_scale.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            variance_coeffs.append(var_scale_batch)  # (B, 1, H, W)

        # Stack all scales: (B, n_scales, H, W)
        variance_all = torch.cat(variance_coeffs, dim=1)

        # Take square root to get standard deviations
        noise_levels = torch.sqrt(variance_all)

        return noise_levels

    def get_snr(
        self,
        image: torch.Tensor,
        noise_sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        keep_sign: bool = False,
    ) -> torch.Tensor:
        """
        Compute Signal-to-Noise Ratio (SNR) for wavelet coefficients.

        Args:
            image: Input image, shape (H, W) or (B, 1, H, W)
            noise_sigma: Noise std deviation map
            mask: Optional observation mask
            keep_sign: If True, preserve sign of coefficients in SNR

        Returns:
            SNR map for each scale, shape (B, n_scales, H, W)
        """
        # Get wavelet coefficients
        wavelet_coeffs = self.forward(image, return_coarse=True)

        # Get noise levels for each coefficient
        noise_levels = self.get_noise_levels(noise_sigma, mask=mask)

        # Compute SNR
        snr = torch.zeros_like(wavelet_coeffs)
        valid_mask = noise_levels != 0
        snr[valid_mask] = wavelet_coeffs[valid_mask] / noise_levels[valid_mask]

        if not keep_sign:
            snr = torch.abs(snr)

        return snr

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"n_scales={self.n_scales}, device={self.device}"


def test_starlet():
    """Quick test of Starlet2D implementation."""
    print("Testing Starlet2D...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test image
    img_size = 128
    n_scales = 5
    image = torch.randn(1, 1, img_size, img_size, device=device)

    # Initialize and transform
    starlet = Starlet2D(n_scales=n_scales, device=device)
    coeffs = starlet(image)

    print(f"Input shape: {image.shape}")
    print(f"Output shape: {coeffs.shape}")
    assert coeffs.shape == (1, n_scales, img_size, img_size)

    # Test reconstruction
    reconstructed = starlet.reconstruct(coeffs, gen2=True)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Check reconstruction error
    error = torch.abs(image - reconstructed).max().item()
    print(f"Max reconstruction error: {error:.2e}")
    assert error < 1e-5, f"Reconstruction error too large: {error}"

    # Test SNR computation
    noise_sigma = torch.ones(img_size, img_size, device=device) * 0.1
    snr = starlet.get_snr(image, noise_sigma)
    print(f"SNR shape: {snr.shape}")

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_starlet()
