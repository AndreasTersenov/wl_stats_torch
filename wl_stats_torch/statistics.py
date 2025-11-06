"""
Weak Lensing Statistics Module

Main class for computing weak lensing summary statistics including:
- Wavelet peak counts at multiple scales
- Wavelet L1-norm statistics
- Mono-scale peak counts with Gaussian smoothing
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .peaks import find_peaks_2d, find_peaks_batch, mono_scale_peaks_smoothed, peaks_to_histogram
from .starlet import Starlet2D


class WLStatistics:
    """
    Complete weak lensing statistics calculator.

    This class provides all the functionality from the original CosmoStat
    HOS_starlet_l1norm_peaks class, but implemented in PyTorch for GPU acceleration.

    Attributes:
        n_scales (int): Number of wavelet scales
        device (torch.device): Computation device (cpu or cuda)
        starlet (Starlet2D): Starlet transform instance
        pixel_arcmin (float): Pixel resolution in arcminutes

    Example:
        >>> stats = WLStatistics(n_scales=5, device='cuda')
        >>> results = stats.compute_all_statistics(kappa_map, sigma_map)
        >>> peak_counts = results['wavelet_peak_counts']
        >>> l1_norms = results['wavelet_l1_norms']
    """

    def __init__(
        self,
        n_scales: int = 5,
        device: Optional[torch.device] = None,
        pixel_arcmin: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initialize weak lensing statistics calculator.

        Args:
            n_scales: Number of wavelet scales (including coarse)
            device: torch device for computation. If None, auto-detects.
            pixel_arcmin: Pixel resolution in arcminutes
            dtype: Data type for computations. Default: torch.float64 to match NumPy.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_scales = n_scales
        self.device = device
        self.pixel_arcmin = pixel_arcmin
        self.dtype = dtype

        # Initialize starlet transform
        self.starlet = Starlet2D(n_scales=n_scales, device=device, dtype=dtype)

        # Storage for computed results
        self.wavelet_coeffs = None
        self.noise_levels = None
        self.snr_coeffs = None

        # Computed statistics
        self.wavelet_peak_counts = None
        self.wavelet_peak_positions = None
        self.wavelet_peak_heights = None
        self.l1_norms = None
        self.l1_bins = None
        self.mono_peak_counts = None

    def get_scale_resolutions(self) -> List[float]:
        """Get effective resolution of each scale in arcminutes."""
        return self.starlet.get_scale_resolution(self.pixel_arcmin)

    def _is_batched(self, image: torch.Tensor) -> bool:
        """Check if input is batched (B, H, W) vs single (H, W)."""
        return image.ndim == 3

    def _broadcast_noise_sigma(
        self, noise_sigma: Union[float, torch.Tensor], image: torch.Tensor, is_batched: bool
    ) -> torch.Tensor:
        """
        Broadcast noise_sigma to match image shape.
        
        Args:
            noise_sigma: Scalar, (H, W), or (B, H, W)
            image: Input image (H, W) or (B, H, W)
            is_batched: Whether image is batched
            
        Returns:
            Broadcasted noise_sigma matching image shape
        """
        if isinstance(noise_sigma, (int, float)):
            # Scalar: broadcast to full image shape
            return torch.full_like(image, noise_sigma, dtype=self.dtype, device=self.device)
        
        noise_sigma = noise_sigma.to(self.device, dtype=self.dtype)
        
        if is_batched:
            # Image is (B, H, W)
            if noise_sigma.ndim == 2:
                # noise_sigma is (H, W): broadcast to all batch items
                return noise_sigma.unsqueeze(0).expand(image.shape[0], -1, -1)
            elif noise_sigma.ndim == 3:
                # noise_sigma is (B, H, W): already correct shape
                if noise_sigma.shape[0] != image.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: image has {image.shape[0]} samples, "
                        f"noise_sigma has {noise_sigma.shape[0]}"
                    )
                return noise_sigma
            else:
                raise ValueError(f"Invalid noise_sigma shape: {noise_sigma.shape}")
        else:
            # Image is (H, W)
            if noise_sigma.ndim == 2:
                # noise_sigma is (H, W): already correct shape
                return noise_sigma
            elif noise_sigma.ndim == 3:
                raise ValueError(
                    "noise_sigma has batch dimension (B, H, W) but image is single (H, W)"
                )
            else:
                raise ValueError(f"Invalid noise_sigma shape: {noise_sigma.shape}")

    def _broadcast_mask(
        self, mask: Optional[torch.Tensor], image: torch.Tensor, is_batched: bool
    ) -> Optional[torch.Tensor]:
        """
        Broadcast mask to match image shape.
        
        Args:
            mask: None, (H, W), or (B, H, W)
            image: Input image (H, W) or (B, H, W)
            is_batched: Whether image is batched
            
        Returns:
            Broadcasted mask matching image shape, or None
        """
        if mask is None:
            return None
        
        mask = mask.to(self.device)
        
        if is_batched:
            # Image is (B, H, W)
            if mask.ndim == 2:
                # mask is (H, W): broadcast to all batch items
                return mask.unsqueeze(0).expand(image.shape[0], -1, -1)
            elif mask.ndim == 3:
                # mask is (B, H, W): already correct shape
                if mask.shape[0] != image.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: image has {image.shape[0]} samples, "
                        f"mask has {mask.shape[0]}"
                    )
                return mask
            else:
                raise ValueError(f"Invalid mask shape: {mask.shape}")
        else:
            # Image is (H, W)
            if mask.ndim == 2:
                # mask is (H, W): already correct shape
                return mask
            elif mask.ndim == 3:
                raise ValueError("mask has batch dimension (B, H, W) but image is single (H, W)")
            else:
                raise ValueError(f"Invalid mask shape: {mask.shape}")

    def compute_wavelet_transform(
        self, image: torch.Tensor, noise_sigma: Union[float, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute wavelet transform and SNR for an image or batch of images.

        Args:
            image: Input convergence map(s), shape (H, W) or (B, H, W)
            noise_sigma: Noise standard deviation, scalar, (H, W), or (B, H, W)
            mask: Optional observation mask, None, (H, W), or (B, H, W)

        Returns:
            Dictionary with keys:
                - 'wavelet_coeffs': Wavelet coefficients (n_scales, H, W) or (B, n_scales, H, W)
                - 'noise_levels': Noise std for each coefficient (n_scales, H, W) or (B, n_scales, H, W)
                - 'snr': Signal-to-noise ratio (n_scales, H, W) or (B, n_scales, H, W)
        """
        # Ensure image is on correct device and dtype
        image = image.to(self.device, dtype=self.dtype)
        
        # Check if input is batched
        is_batched = self._is_batched(image)
        
        # Broadcast noise_sigma and mask to match image shape
        noise_sigma = self._broadcast_noise_sigma(noise_sigma, image, is_batched)
        mask = self._broadcast_mask(mask, image, is_batched)

        # Prepare for Starlet transform (needs (B, 1, H, W))
        if is_batched:
            # (B, H, W) -> (B, 1, H, W)
            image_4d = image.unsqueeze(1)
            noise_sigma_4d = noise_sigma.unsqueeze(1)
            mask_4d = mask.unsqueeze(1) if mask is not None else None
        else:
            # (H, W) -> (1, 1, H, W)
            image_4d = image.unsqueeze(0).unsqueeze(0)
            noise_sigma_4d = noise_sigma.unsqueeze(0).unsqueeze(0)
            mask_4d = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None

        # Compute wavelet transform
        wavelet_coeffs = self.starlet(image_4d, return_coarse=True)  # (B, n_scales, H, W)

        # Compute noise levels
        noise_levels = self.starlet.get_noise_levels(noise_sigma_4d, mask=mask_4d)  # (B, n_scales, H, W)

        # Compute SNR
        snr = torch.zeros_like(wavelet_coeffs)
        valid_mask = noise_levels != 0
        snr[valid_mask] = wavelet_coeffs[valid_mask] / noise_levels[valid_mask]

        # Remove batch dimension for single image case
        if not is_batched:
            wavelet_coeffs = wavelet_coeffs.squeeze(0)  # (n_scales, H, W)
            noise_levels = noise_levels.squeeze(0)  # (n_scales, H, W)
            snr = snr.squeeze(0)  # (n_scales, H, W)

        # Store results
        self.wavelet_coeffs = wavelet_coeffs
        self.noise_levels = noise_levels
        self.snr_coeffs = snr

        return {"wavelet_coeffs": wavelet_coeffs, "noise_levels": noise_levels, "snr": snr}

    def compute_wavelet_peak_counts(
        self,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        mask: Optional[torch.Tensor] = None,
        verbose: bool = False,
        clamp_overflow: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute histogram of peak counts at all wavelet scales.

        Args:
            min_snr: Minimum SNR for histogram bins
            max_snr: Maximum SNR for histogram bins
            n_bins: Number of histogram bins
            mask: Optional mask to restrict peak detection (H, W) or (B, H, W)
            verbose: Print min/max SNR at each scale
            clamp_overflow: If True, peaks outside SNR range are included in edge bins.
                          If False (default), peaks outside range are excluded.
                          False matches cosmostat/pycs behavior.

        Returns:
            Tuple of (bin_centers, peak_counts_list) where:
                - bin_centers: Tensor of shape (n_bins,)
                - peak_counts_list: List of tensors per scale, shape (n_bins,) or (B, n_bins)
        """
        if self.snr_coeffs is None:
            raise RuntimeError("Must call compute_wavelet_transform first")

        # Check if we're processing a batch
        is_batched = self.snr_coeffs.ndim == 4

        # Create bins
        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=self.device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        peak_counts_list = []
        peak_positions_list = []
        peak_heights_list = []

        for scale_idx in range(self.n_scales):
            if is_batched:
                # Batched processing: snr_scale is (B, H, W)
                snr_scale = self.snr_coeffs[:, scale_idx, :, :]
                batch_size = snr_scale.shape[0]

                if verbose:
                    print(
                        f"Scale {scale_idx + 1}: "
                        f"Min SNR = {snr_scale.min():.4f}, "
                        f"Max SNR = {snr_scale.max():.4f}"
                    )

                # Find peaks for all images in batch
                # snr_scale needs to be (B, 1, H, W) for find_peaks_batch
                snr_batch_4d = snr_scale.unsqueeze(1)
                
                # Prepare mask for find_peaks_batch
                if mask is not None:
                    if mask.ndim == 2:
                        # Shared mask (H, W): broadcast to batch and add channel dim
                        mask_batch_4d = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
                    else:
                        # Per-sample mask (B, H, W): add channel dim
                        mask_batch_4d = mask.unsqueeze(1)
                else:
                    mask_batch_4d = None

                batch_results = find_peaks_batch(
                    snr_batch_4d,
                    threshold=None,
                    masks=mask_batch_4d,
                    include_border=False,
                    ordered=True,
                )

                # Compute histograms for each image in batch
                batch_counts = torch.zeros(batch_size, n_bins, device=self.device)
                batch_positions = []
                batch_heights = []

                for i, (positions, heights) in enumerate(batch_results):
                    counts = peaks_to_histogram(heights, bins, clamp_overflow=clamp_overflow)
                    batch_counts[i] = counts
                    batch_positions.append(positions)
                    batch_heights.append(heights)

                peak_counts_list.append(batch_counts)
                peak_positions_list.append(batch_positions)
                peak_heights_list.append(batch_heights)

            else:
                # Single image processing: snr_scale is (H, W)
                snr_scale = self.snr_coeffs[scale_idx]

                if verbose:
                    print(
                        f"Scale {scale_idx + 1}: "
                        f"Min SNR = {snr_scale.min():.4f}, "
                        f"Max SNR = {snr_scale.max():.4f}"
                    )

                # Find peaks
                positions, heights = find_peaks_2d(
                    snr_scale, threshold=None, mask=mask, include_border=False, ordered=True
                )

                # Compute histogram
                counts = peaks_to_histogram(heights, bins, clamp_overflow=clamp_overflow)

                peak_counts_list.append(counts)
                peak_positions_list.append(positions)
                peak_heights_list.append(heights)

        # Store results
        self.wavelet_peak_counts = peak_counts_list
        self.wavelet_peak_positions = peak_positions_list
        self.wavelet_peak_heights = peak_heights_list

        return bin_centers, peak_counts_list

    def compute_wavelet_l1_norms(
        self,
        n_bins: int = 40,
        mask: Optional[torch.Tensor] = None,
        min_snr: Optional[float] = None,
        max_snr: Optional[float] = None,
        clamp_overflow: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute L1-norm as a function of SNR threshold for each scale.

        For each scale, we bin the SNR values and compute the sum of
        absolute wavelet coefficients in each bin.

        Args:
            n_bins: Number of bins for SNR binning
            mask: Optional mask to restrict calculation (H, W) or (B, H, W)
            min_snr: Minimum SNR (if None, uses data minimum)
            max_snr: Maximum SNR (if None, uses data maximum)
            clamp_overflow: If True, values outside SNR range are included in edge bins.
                          If False (default), values outside range are excluded.
                          False matches cosmostat/pycs behavior.

        Returns:
            Tuple of (bins_list, l1_norms_list) where:
                - bins_list: List of bin center tensors per scale
                - l1_norms_list: List of L1-norm tensors per scale, shape (n_bins,) or (B, n_bins)
        """
        if self.snr_coeffs is None:
            raise RuntimeError("Must call compute_wavelet_transform first")

        # Check if we're processing a batch
        is_batched = self.snr_coeffs.ndim == 4

        bins_list = []
        l1_norms_list = []

        for scale_idx in range(self.n_scales):
            if is_batched:
                # Batched processing: snr_scale is (B, H, W)
                snr_scale = self.snr_coeffs[:, scale_idx, :, :]
                batch_size = snr_scale.shape[0]

                # Determine SNR range across entire batch
                if mask is not None:
                    # Apply mask to get valid values
                    if mask.ndim == 2:
                        # Shared mask: broadcast to batch
                        mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        mask_expanded = mask
                    
                    # Collect all valid SNR values across batch
                    snr_all_valid = []
                    for b in range(batch_size):
                        snr_all_valid.append(snr_scale[b][mask_expanded[b] != 0])
                    snr_all_valid = torch.cat(snr_all_valid)
                else:
                    snr_all_valid = snr_scale.flatten()

                current_min = min_snr if min_snr is not None else snr_all_valid.min().item()
                current_max = max_snr if max_snr is not None else snr_all_valid.max().item()

                # Create bins (shared across batch)
                thresholds = torch.linspace(current_min, current_max, n_bins + 1, device=self.device)
                bin_centers = 0.5 * (thresholds[:-1] + thresholds[1:])

                # Process each image in batch
                l1_batch = torch.zeros(batch_size, n_bins, device=self.device)

                for b in range(batch_size):
                    snr_img = snr_scale[b]

                    # Apply mask if provided
                    if mask is not None:
                        if mask.ndim == 2:
                            mask_img = mask
                        else:
                            mask_img = mask[b]
                        snr_masked = snr_img[mask_img != 0]
                    else:
                        snr_masked = snr_img.flatten()

                    # Digitize SNR values
                    bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)

                    if clamp_overflow:
                        bin_indices = torch.clamp(bin_indices, 1, n_bins)
                        for bin_idx in range(1, n_bins + 1):
                            mask_bin = bin_indices == bin_idx
                            if mask_bin.any():
                                l1_batch[b, bin_idx - 1] = torch.abs(snr_masked[mask_bin]).sum()
                    else:
                        valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)
                        if valid_mask.any():
                            bin_indices_valid = bin_indices[valid_mask]
                            snr_valid = snr_masked[valid_mask]
                            for bin_idx in range(1, n_bins + 1):
                                mask_bin = bin_indices_valid == bin_idx
                                if mask_bin.any():
                                    l1_batch[b, bin_idx - 1] = torch.abs(snr_valid[mask_bin]).sum()

                bins_list.append(bin_centers)
                l1_norms_list.append(l1_batch)

            else:
                # Single image processing: snr_scale is (H, W)
                snr_scale = self.snr_coeffs[scale_idx]

                # Apply mask if provided
                if mask is not None:
                    mask_2d = mask.to(self.device)
                    snr_masked = snr_scale[mask_2d != 0]
                else:
                    snr_masked = snr_scale.flatten()

                # Determine SNR range
                current_min = min_snr if min_snr is not None else snr_masked.min().item()
                current_max = max_snr if max_snr is not None else snr_masked.max().item()

                # Create bins
                thresholds = torch.linspace(current_min, current_max, n_bins + 1, device=self.device)
                bin_centers = 0.5 * (thresholds[:-1] + thresholds[1:])

                # Digitize SNR values
                bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)

                if clamp_overflow:
                    bin_indices = torch.clamp(bin_indices, 1, n_bins)
                    l1_per_bin = torch.zeros(n_bins, device=self.device)
                    for bin_idx in range(1, n_bins + 1):
                        mask_bin = bin_indices == bin_idx
                        if mask_bin.any():
                            l1_per_bin[bin_idx - 1] = torch.abs(snr_masked[mask_bin]).sum()
                else:
                    valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)
                    l1_per_bin = torch.zeros(n_bins, device=self.device)
                    if valid_mask.any():
                        bin_indices_valid = bin_indices[valid_mask]
                        snr_valid = snr_masked[valid_mask]
                        for bin_idx in range(1, n_bins + 1):
                            mask_bin = bin_indices_valid == bin_idx
                            if mask_bin.any():
                                l1_per_bin[bin_idx - 1] = torch.abs(snr_valid[mask_bin]).sum()

                bins_list.append(bin_centers)
                l1_norms_list.append(l1_per_bin)

        # Store results
        self.l1_bins = bins_list
        self.l1_norms = l1_norms_list

        return bins_list, l1_norms_list

    def compute_mono_scale_peaks(
        self,
        image: torch.Tensor,
        noise_sigma: Union[float, torch.Tensor],
        smoothing_sigma: float = 2.0,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        mask: Optional[torch.Tensor] = None,
        clamp_overflow: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mono-scale peak counts with Gaussian smoothing.

        Args:
            image: Input convergence map (H, W) or (B, H, W)
            noise_sigma: Standard deviation of noise, scalar or tensor
            smoothing_sigma: Gaussian smoothing scale in pixels
            min_snr: Minimum SNR for histogram
            max_snr: Maximum SNR for histogram
            n_bins: Number of histogram bins
            mask: Optional observation mask (H, W) or (B, H, W)
            clamp_overflow: If True, peaks outside SNR range are included in edge bins.
                          If False (default), peaks outside range are excluded.
                          False matches cosmostat/pycs behavior.

        Returns:
            Tuple of (bin_centers, peak_counts)
                - peak_counts shape: (n_bins,) or (B, n_bins)
        """
        image = image.to(self.device, dtype=self.dtype)
        is_batched = self._is_batched(image)

        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=self.device)

        if is_batched:
            # Process batch of images
            batch_size = image.shape[0]
            batch_counts = torch.zeros(batch_size, n_bins, device=self.device)

            # Determine mean noise if noise_sigma is a map
            if isinstance(noise_sigma, torch.Tensor) and noise_sigma.numel() > 1:
                noise_sigma = noise_sigma.to(self.device, dtype=self.dtype)
                if noise_sigma.ndim == 2:
                    # Shared noise map: use mean
                    if mask is not None and mask.ndim == 2:
                        mean_noise = noise_sigma[mask != 0].mean().item()
                    else:
                        mean_noise = noise_sigma.mean().item()
                    noise_sigma_scalar = mean_noise
                elif noise_sigma.ndim == 3:
                    # Per-sample noise: will handle per-sample below
                    noise_sigma_scalar = None
                else:
                    noise_sigma_scalar = noise_sigma.item()
            else:
                noise_sigma_scalar = float(noise_sigma) if isinstance(noise_sigma, (int, float)) else noise_sigma.item()

            for b in range(batch_size):
                img_b = image[b]
                mask_b = mask[b] if (mask is not None and mask.ndim == 3) else mask

                # Get noise for this sample
                if noise_sigma_scalar is None:
                    # Per-sample noise map
                    noise_b = noise_sigma[b]
                    if mask_b is not None:
                        noise_scalar_b = noise_b[mask_b != 0].mean().item()
                    else:
                        noise_scalar_b = noise_b.mean().item()
                else:
                    noise_scalar_b = noise_sigma_scalar

                bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
                    img_b,
                    sigma_noise=noise_scalar_b,
                    smoothing_sigma=smoothing_sigma,
                    mask=mask_b,
                    bins=bins,
                    clamp_overflow=clamp_overflow,
                )
                batch_counts[b] = counts

            self.mono_peak_counts = batch_counts
            return bin_centers, batch_counts

        else:
            # Single image processing
            if mask is not None:
                mask = mask.to(self.device)

            # Convert noise_sigma to scalar if it's a map
            if isinstance(noise_sigma, torch.Tensor) and noise_sigma.numel() > 1:
                noise_sigma = noise_sigma.to(self.device, dtype=self.dtype)
                if mask is not None:
                    mean_noise = noise_sigma[mask != 0].mean().item()
                else:
                    mean_noise = noise_sigma.mean().item()
                noise_sigma_scalar = mean_noise
            elif isinstance(noise_sigma, torch.Tensor):
                noise_sigma_scalar = noise_sigma.item()
            else:
                noise_sigma_scalar = float(noise_sigma)

            bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
                image,
                sigma_noise=noise_sigma_scalar,
                smoothing_sigma=smoothing_sigma,
                mask=mask,
                bins=bins,
                clamp_overflow=clamp_overflow,
            )

            self.mono_peak_counts = counts
            return bin_centers, counts

    def compute_all_statistics(
        self,
        image: torch.Tensor,
        noise_sigma: Union[float, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        l1_nbins: int = 40,
        l1_min_snr: Optional[float] = None,
        l1_max_snr: Optional[float] = None,
        compute_mono: bool = True,
        mono_smoothing_sigma: float = 2.0,
        verbose: bool = False,
        clamp_overflow: bool = False,
    ) -> Dict[str, any]:
        """
        Compute all weak lensing statistics in one call.

        Supports both single images (H, W) and batches (B, H, W).

        Args:
            image: Convergence map (H, W) or batch (B, H, W)
            noise_sigma: Noise std, scalar, (H, W), or (B, H, W)
            mask: Optional observation mask, None, (H, W), or (B, H, W)
            min_snr: Minimum SNR for peak count histograms
            max_snr: Maximum SNR for peak count histograms
            n_bins: Number of bins for peak histograms
            l1_nbins: Number of bins for L1-norm
            l1_min_snr: Minimum SNR for L1-norm (if None, uses min_snr)
            l1_max_snr: Maximum SNR for L1-norm (if None, uses max_snr)
            compute_mono: Whether to compute mono-scale peaks
            mono_smoothing_sigma: Smoothing scale for mono-scale peaks
            verbose: Print progress information
            clamp_overflow: If True, values outside SNR range are included in edge bins.
                          If False (default), values outside range are excluded.
                          False matches cosmostat/pycs behavior.

        Returns:
            Dictionary with all computed statistics.
            For batched input (B, H, W):
                - 'wavelet_coeffs': (B, n_scales, H, W)
                - 'wavelet_peak_counts': list of (B, n_bins) per scale
                - 'wavelet_l1_norms': list of (B, l1_nbins) per scale
                - 'mono_peak_counts': (B, n_bins) if compute_mono=True
            For single input (H, W):
                - 'wavelet_coeffs': (n_scales, H, W)
                - 'wavelet_peak_counts': list of (n_bins,) per scale
                - 'wavelet_l1_norms': list of (l1_nbins,) per scale
                - 'mono_peak_counts': (n_bins,) if compute_mono=True
        """
        results = {}

        # Use peak SNR ranges for L1-norm if not separately specified
        l1_min_snr_use = l1_min_snr if l1_min_snr is not None else min_snr
        l1_max_snr_use = l1_max_snr if l1_max_snr is not None else max_snr

        # 1. Wavelet transform and SNR
        if verbose:
            is_batched = self._is_batched(image)
            batch_info = f" (batch size {image.shape[0]})" if is_batched else ""
            print(f"Computing wavelet transform{batch_info}...")
        wt_results = self.compute_wavelet_transform(image, noise_sigma, mask)
        results.update(wt_results)

        # 2. Wavelet peak counts
        if verbose:
            print("Computing wavelet peak counts...")
        bin_centers, peak_counts = self.compute_wavelet_peak_counts(
            min_snr=min_snr,
            max_snr=max_snr,
            n_bins=n_bins,
            mask=mask,
            verbose=verbose,
            clamp_overflow=clamp_overflow,
        )
        results["peak_bins"] = bin_centers
        results["wavelet_peak_counts"] = peak_counts
        results["wavelet_peak_positions"] = self.wavelet_peak_positions
        results["wavelet_peak_heights"] = self.wavelet_peak_heights

        # 3. Wavelet L1-norms
        if verbose:
            print("Computing wavelet L1-norms...")
        l1_bins, l1_norms = self.compute_wavelet_l1_norms(
            n_bins=l1_nbins,
            mask=mask,
            min_snr=l1_min_snr_use,
            max_snr=l1_max_snr_use,
            clamp_overflow=clamp_overflow,
        )
        results["l1_bins"] = l1_bins
        results["wavelet_l1_norms"] = l1_norms

        # 4. Mono-scale peaks (optional)
        if compute_mono:
            if verbose:
                print("Computing mono-scale peaks...")

            mono_bins, mono_counts = self.compute_mono_scale_peaks(
                image,
                noise_sigma=noise_sigma,
                smoothing_sigma=mono_smoothing_sigma,
                min_snr=min_snr,
                max_snr=max_snr,
                n_bins=n_bins,
                mask=mask,
                clamp_overflow=clamp_overflow,
            )
            results["mono_peak_bins"] = mono_bins
            results["mono_peak_counts"] = mono_counts

        if verbose:
            print("✓ All statistics computed!")

        return results

    def to(self, device: torch.device):
        """Move all components to specified device."""
        self.device = device
        self.starlet.to(device)
        return self


def test_statistics():
    """Test the complete statistics pipeline."""
    print("Testing WLStatistics...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic data
    img_size = 128
    kappa = torch.randn(img_size, img_size, device=device)
    sigma = torch.ones(img_size, img_size, device=device) * 0.1

    # Add a mask
    mask = torch.ones(img_size, img_size, device=device)
    mask[:20, :] = 0  # Mask out top region

    # Initialize statistics calculator
    stats = WLStatistics(n_scales=5, device=device, pixel_arcmin=0.4)

    # Compute all statistics
    results = stats.compute_all_statistics(kappa, sigma, mask=mask, verbose=True)

    # Check results
    print("\nResults summary:")
    print(f"  Wavelet scales: {len(results['wavelet_peak_counts'])}")
    print(f"  Peak bins: {len(results['peak_bins'])}")
    print(f"  L1 bins (scale 0): {len(results['l1_bins'][0])}")

    for i, counts in enumerate(results["wavelet_peak_counts"]):
        n_peaks = counts.sum().item()
        print(f"  Scale {i+1}: {int(n_peaks)} peaks")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_statistics()
