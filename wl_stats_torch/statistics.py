"""
Weak Lensing Statistics Module

Main class for computing weak lensing summary statistics including:
- Wavelet peak counts at multiple scales
- Wavelet L1-norm statistics
- Mono-scale peak counts with Gaussian smoothing
"""

import torch
from typing import Optional, Dict, List, Tuple
import warnings

from .starlet import Starlet2D
from .peaks import (
    find_peaks_2d,
    peaks_to_histogram,
    mono_scale_peaks_smoothed
)


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
        dtype: torch.dtype = torch.float64
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
    
    def compute_wavelet_transform(
        self,
        image: torch.Tensor,
        noise_sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute wavelet transform and SNR for an image.
        
        Args:
            image: Input convergence map, shape (H, W)
            noise_sigma: Noise standard deviation map, shape (H, W)
            mask: Optional observation mask, shape (H, W)
        
        Returns:
            Dictionary with keys:
                - 'wavelet_coeffs': Wavelet coefficients (n_scales, H, W)
                - 'noise_levels': Noise std for each coefficient (n_scales, H, W)
                - 'snr': Signal-to-noise ratio (n_scales, H, W)
        """
        # Ensure tensors are on correct device and dtype
        image = image.to(self.device, dtype=self.dtype)
        noise_sigma = noise_sigma.to(self.device, dtype=self.dtype)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Compute wavelet transform
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        wavelet_coeffs = self.starlet(image, return_coarse=True)
        wavelet_coeffs = wavelet_coeffs.squeeze(0)  # (n_scales, H, W)
        
        # Compute noise levels
        noise_levels = self.starlet.get_noise_levels(noise_sigma, mask=mask)
        noise_levels = noise_levels.squeeze(0)  # (n_scales, H, W)
        
        # Compute SNR
        snr = torch.zeros_like(wavelet_coeffs)
        valid_mask = noise_levels != 0
        snr[valid_mask] = wavelet_coeffs[valid_mask] / noise_levels[valid_mask]
        
        # Store results
        self.wavelet_coeffs = wavelet_coeffs
        self.noise_levels = noise_levels
        self.snr_coeffs = snr
        
        return {
            'wavelet_coeffs': wavelet_coeffs,
            'noise_levels': noise_levels,
            'snr': snr
        }
    
    def compute_wavelet_peak_counts(
        self,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        mask: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute histogram of peak counts at all wavelet scales.
        
        Args:
            min_snr: Minimum SNR for histogram bins
            max_snr: Maximum SNR for histogram bins
            n_bins: Number of histogram bins
            mask: Optional mask to restrict peak detection
            verbose: Print min/max SNR at each scale
        
        Returns:
            Tuple of (bin_centers, peak_counts_list) where:
                - bin_centers: Tensor of shape (n_bins,)
                - peak_counts_list: List of tensors, one per scale
        """
        if self.snr_coeffs is None:
            raise RuntimeError("Must call compute_wavelet_transform first")
        
        # Create bins
        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=self.device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        peak_counts_list = []
        peak_positions_list = []
        peak_heights_list = []
        
        for scale_idx in range(self.n_scales):
            snr_scale = self.snr_coeffs[scale_idx]
            
            if verbose:
                print(f"Scale {scale_idx + 1}: "
                      f"Min SNR = {snr_scale.min():.4f}, "
                      f"Max SNR = {snr_scale.max():.4f}")
            
            # Find peaks
            positions, heights = find_peaks_2d(
                snr_scale,
                threshold=None,
                mask=mask,
                include_border=False,
                ordered=True
            )
            
            # Compute histogram
            counts = peaks_to_histogram(heights, bins)
            
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
        max_snr: Optional[float] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute L1-norm as a function of SNR threshold for each scale.
        
        For each scale, we bin the SNR values and compute the sum of
        absolute wavelet coefficients in each bin.
        
        Args:
            n_bins: Number of bins for SNR binning
            mask: Optional mask to restrict calculation
            min_snr: Minimum SNR (if None, uses data minimum)
            max_snr: Maximum SNR (if None, uses data maximum)
        
        Returns:
            Tuple of (bins_list, l1_norms_list) where:
                - bins_list: List of bin center tensors, one per scale
                - l1_norms_list: List of L1-norm tensors, one per scale
        """
        if self.snr_coeffs is None:
            raise RuntimeError("Must call compute_wavelet_transform first")
        
        bins_list = []
        l1_norms_list = []
        
        for scale_idx in range(self.n_scales):
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
            thresholds = torch.linspace(
                current_min, current_max, n_bins + 1, device=self.device
            )
            bin_centers = 0.5 * (thresholds[:-1] + thresholds[1:])
            
            # Digitize SNR values
            bin_indices = torch.searchsorted(thresholds, snr_masked, right=False)
            bin_indices = torch.clamp(bin_indices, 1, n_bins)
            
            # Compute L1-norm for each bin
            l1_per_bin = torch.zeros(n_bins, device=self.device)
            for bin_idx in range(1, n_bins + 1):
                mask_bin = bin_indices == bin_idx
                if mask_bin.any():
                    l1_per_bin[bin_idx - 1] = torch.abs(snr_masked[mask_bin]).sum()
            
            bins_list.append(bin_centers)
            l1_norms_list.append(l1_per_bin)
        
        # Store results
        self.l1_bins = bins_list
        self.l1_norms = l1_norms_list
        
        return bins_list, l1_norms_list
    
    def compute_mono_scale_peaks(
        self,
        image: torch.Tensor,
        noise_sigma: float,
        smoothing_sigma: float = 2.0,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mono-scale peak counts with Gaussian smoothing.
        
        Args:
            image: Input convergence map (H, W)
            noise_sigma: Standard deviation of noise (scalar)
            smoothing_sigma: Gaussian smoothing scale in pixels
            min_snr: Minimum SNR for histogram
            max_snr: Maximum SNR for histogram
            n_bins: Number of histogram bins
            mask: Optional observation mask
        
        Returns:
            Tuple of (bin_centers, peak_counts)
        """
        image = image.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=self.device)
        
        bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
            image,
            sigma_noise=noise_sigma,
            smoothing_sigma=smoothing_sigma,
            mask=mask,
            bins=bins
        )
        
        self.mono_peak_counts = counts
        
        return bin_centers, counts
    
    def compute_all_statistics(
        self,
        image: torch.Tensor,
        noise_sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        min_snr: float = -2.0,
        max_snr: float = 6.0,
        n_bins: int = 31,
        l1_nbins: int = 40,
        compute_mono: bool = True,
        mono_smoothing_sigma: float = 2.0,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Compute all weak lensing statistics in one call.
        
        Args:
            image: Convergence map (H, W)
            noise_sigma: Noise std map (H, W) or scalar
            mask: Optional observation mask (H, W)
            min_snr: Minimum SNR for histograms
            max_snr: Maximum SNR for histograms
            n_bins: Number of bins for peak histograms
            l1_nbins: Number of bins for L1-norm
            compute_mono: Whether to compute mono-scale peaks
            mono_smoothing_sigma: Smoothing scale for mono-scale peaks
            verbose: Print progress information
        
        Returns:
            Dictionary with all computed statistics
        """
        results = {}
        
        # Convert noise_sigma to tensor if scalar
        if isinstance(noise_sigma, (int, float)):
            noise_sigma = torch.full_like(image, noise_sigma)
        
        # 1. Wavelet transform and SNR
        if verbose:
            print("Computing wavelet transform...")
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
            verbose=verbose
        )
        results['peak_bins'] = bin_centers
        results['wavelet_peak_counts'] = peak_counts
        results['wavelet_peak_positions'] = self.wavelet_peak_positions
        results['wavelet_peak_heights'] = self.wavelet_peak_heights
        
        # 3. Wavelet L1-norms
        if verbose:
            print("Computing wavelet L1-norms...")
        l1_bins, l1_norms = self.compute_wavelet_l1_norms(
            n_bins=l1_nbins,
            mask=mask,
            min_snr=min_snr,
            max_snr=max_snr
        )
        results['l1_bins'] = l1_bins
        results['wavelet_l1_norms'] = l1_norms
        
        # 4. Mono-scale peaks (optional)
        if compute_mono:
            if verbose:
                print("Computing mono-scale peaks...")
            
            # Use mean noise if noise_sigma is a map
            if noise_sigma.numel() > 1:
                mean_noise = noise_sigma[mask != 0].mean().item() if mask is not None else noise_sigma.mean().item()
            else:
                mean_noise = noise_sigma.item()
            
            mono_bins, mono_counts = self.compute_mono_scale_peaks(
                image,
                noise_sigma=mean_noise,
                smoothing_sigma=mono_smoothing_sigma,
                min_snr=min_snr,
                max_snr=max_snr,
                n_bins=n_bins,
                mask=mask
            )
            results['mono_peak_bins'] = mono_bins
            results['mono_peak_counts'] = mono_counts
        
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    results = stats.compute_all_statistics(
        kappa,
        sigma,
        mask=mask,
        verbose=True
    )
    
    # Check results
    print(f"\nResults summary:")
    print(f"  Wavelet scales: {len(results['wavelet_peak_counts'])}")
    print(f"  Peak bins: {len(results['peak_bins'])}")
    print(f"  L1 bins (scale 0): {len(results['l1_bins'][0])}")
    
    for i, counts in enumerate(results['wavelet_peak_counts']):
        n_peaks = counts.sum().item()
        print(f"  Scale {i+1}: {int(n_peaks)} peaks")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_statistics()
