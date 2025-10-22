"""
Peak Detection for 2D Images in PyTorch

This module provides fast, GPU-accelerated peak detection for 2D images.
A peak is defined as a local maximum - a pixel with a value greater than
all of its 8 neighbors.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


def find_peaks_2d(
    image: torch.Tensor,
    threshold: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    include_border: bool = False,
    ordered: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find local maxima (peaks) in a 2D image.
    
    A peak is defined as a pixel with value strictly greater than all 8 neighbors.
    
    Args:
        image: 2D tensor (H, W) or (1, 1, H, W)
        threshold: Minimum value to consider as peak. If None, uses image minimum.
        mask: Optional binary mask (H, W) where 1 = consider, 0 = ignore
        include_border: If True, include peaks on the image border
        ordered: If True, return peaks sorted by height (descending)
    
    Returns:
        Tuple of (positions, heights):
            - positions: Tensor of shape (N, 2) with (row, col) coordinates
            - heights: Tensor of shape (N,) with peak values
    
    Example:
        >>> image = torch.randn(128, 128)
        >>> positions, heights = find_peaks_2d(image, threshold=2.0)
        >>> print(f"Found {len(positions)} peaks above threshold")
    """
    # Handle input shapes
    if image.ndim == 4:
        if image.shape[0] != 1 or image.shape[1] != 1:
            raise ValueError("For 4D input, batch and channel dimensions must be 1")
        image = image.squeeze(0).squeeze(0)
    elif image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError("For 3D input, first dimension must be 1")
        image = image.squeeze(0)
    elif image.ndim != 2:
        raise ValueError(f"Image must be 2D, 3D, or 4D, got shape {image.shape}")
    
    H, W = image.shape
    device = image.device
    
    # Handle threshold
    if threshold is None:
        threshold = image.min().item()
    
    # Handle mask
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.squeeze(0).squeeze(0)
        elif mask.ndim == 3:
            mask = mask.squeeze(0)
        
        if mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {image.shape}")
        
        mask = mask.bool()
    else:
        mask = torch.ones(H, W, dtype=torch.bool, device=device)
    
    # If not including borders, zero out the border
    if not include_border:
        border_mask = torch.ones(H, W, dtype=torch.bool, device=device)
        border_mask[0, :] = False
        border_mask[-1, :] = False
        border_mask[:, 0] = False
        border_mask[:, -1] = False
        mask = mask & border_mask
    
    # Prepare image for comparison (add batch and channel dims)
    img_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Get all 8 neighbors using shifting
    # We'll use padding to handle boundaries
    pad = 1
    img_padded = F.pad(img_4d, (pad, pad, pad, pad), mode='constant', value=float('-inf'))
    
    # Extract shifted versions (the 8 neighbors)
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip center pixel
            # Crop the padded image to get shifted version
            shifted = img_padded[:, :, 1+di:1+di+H, 1+dj:1+dj+W]
            neighbors.append(shifted.squeeze(0).squeeze(0))
    
    # Stack all neighbors: (8, H, W)
    neighbors_tensor = torch.stack(neighbors, dim=0)
    
    # Find maximum neighbor value at each position
    max_neighbor, _ = neighbors_tensor.max(dim=0)
    
    # A pixel is a peak if it's strictly greater than all neighbors
    # AND it's above threshold AND it's in the mask
    is_peak = (image > max_neighbor) & (image >= threshold) & mask
    
    # Get peak positions
    peak_indices = torch.nonzero(is_peak, as_tuple=False)  # (N, 2) with (row, col)
    
    if peak_indices.numel() == 0:
        # No peaks found
        return torch.empty((0, 2), device=device), torch.empty(0, device=device)
    
    # Extract peak heights
    peak_heights = image[is_peak]
    
    # Sort by height if requested
    if ordered:
        sorted_indices = torch.argsort(peak_heights, descending=True)
        peak_indices = peak_indices[sorted_indices]
        peak_heights = peak_heights[sorted_indices]
    
    return peak_indices, peak_heights


def find_peaks_batch(
    images: torch.Tensor,
    threshold: Optional[float] = None,
    masks: Optional[torch.Tensor] = None,
    include_border: bool = False,
    ordered: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Find peaks in a batch of images.
    
    Args:
        images: Tensor of shape (B, 1, H, W) or (B, H, W)
        threshold: Minimum value to consider as peak
        masks: Optional masks of shape (B, 1, H, W) or (B, H, W)
        include_border: If True, include peaks on borders
        ordered: If True, sort peaks by height
    
    Returns:
        List of (positions, heights) tuples, one per image in batch
    """
    if images.ndim == 3:
        images = images.unsqueeze(1)  # Add channel dimension
    
    batch_size = images.shape[0]
    
    if masks is not None:
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between images and masks")
    
    results = []
    for i in range(batch_size):
        image = images[i, 0]  # (H, W)
        mask = masks[i, 0] if masks is not None else None
        
        positions, heights = find_peaks_2d(
            image,
            threshold=threshold,
            mask=mask,
            include_border=include_border,
            ordered=ordered
        )
        results.append((positions, heights))
    
    return results


def peaks_to_histogram(
    peak_heights: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """
    Compute histogram of peak heights.
    
    Args:
        peak_heights: Tensor of peak values, shape (N,)
        bins: Bin edges, shape (n_bins+1,)
    
    Returns:
        Histogram counts, shape (n_bins,)
    """
    if peak_heights.numel() == 0:
        return torch.zeros(len(bins) - 1, device=bins.device)
    
    # Use torch.histc or manual binning
    # histc doesn't work well with custom bins, so we'll use searchsorted
    bins = bins.to(peak_heights.device)
    
    # Find which bin each peak belongs to
    bin_indices = torch.searchsorted(bins, peak_heights, right=False)
    
    # Clip to valid range [1, n_bins]
    bin_indices = torch.clamp(bin_indices, 1, len(bins) - 1)
    
    # Count peaks in each bin (shift by -1 since searchsorted returns 1-indexed)
    counts = torch.bincount(
        bin_indices - 1,
        minlength=len(bins) - 1
    )
    
    return counts[:len(bins)-1].float()


def mono_scale_peaks_smoothed(
    image: torch.Tensor,
    sigma_noise: float,
    smoothing_sigma: float = 2.0,
    mask: Optional[torch.Tensor] = None,
    bins: Optional[torch.Tensor] = None,
    min_snr: float = -2.0,
    max_snr: float = 6.0,
    n_bins: int = 31
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute mono-scale peak counts with Gaussian smoothing.
    
    This applies Gaussian smoothing to the image, computes SNR, finds peaks,
    and returns histogram of peak counts.
    
    Args:
        image: Input image (H, W)
        sigma_noise: Standard deviation of noise
        smoothing_sigma: Std dev for Gaussian smoothing (in pixels)
        mask: Optional observation mask
        bins: Optional custom bin edges for histogram
        min_snr: Minimum SNR for histogram (if bins not provided)
        max_snr: Maximum SNR for histogram (if bins not provided)
        n_bins: Number of bins for histogram (if bins not provided)
    
    Returns:
        Tuple of (bin_centers, counts, (peak_positions, peak_heights))
    """
    device = image.device
    
    # Create Gaussian kernel for smoothing
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Create Gaussian kernel
    kernel_size = int(6 * smoothing_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian
    x = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x = x - kernel_size // 2
    gaussian_1d = torch.exp(-x**2 / (2 * smoothing_sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # Create 2D Gaussian via outer product
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    
    # Smooth the image
    padding = kernel_size // 2
    if padding > 0:
        image_padded = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    else:
        image_padded = image

    image_smoothed = F.conv2d(
        image_padded,
        gaussian_2d,
        bias=None,
        stride=1,
        padding=0,
    )
    
    # Propagate noise through smoothing
    # Variance after convolution with normalized kernel G:
    # var(G * X) = sigma^2 * sum(G^2)
    gaussian_squared = gaussian_2d ** 2
    noise_factor_squared = gaussian_squared.sum()
    smoothed_noise_sigma = sigma_noise * torch.sqrt(noise_factor_squared)
    
    # Compute SNR
    snr_image = image_smoothed / smoothed_noise_sigma.item()
    snr_image = snr_image.squeeze(0).squeeze(0)  # Back to (H, W)
    
    # Find peaks
    peak_positions, peak_heights = find_peaks_2d(
        snr_image,
        threshold=None,
        mask=mask,
        include_border=False,
        ordered=True
    )
    
    # Create histogram bins if not provided
    if bins is None:
        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=device)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Compute histogram
    counts = peaks_to_histogram(peak_heights, bins)
    
    return bin_centers, counts, (peak_positions, peak_heights)


def test_peaks():
    """Test peak detection functions."""
    print("Testing peak detection...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test image with known peaks
    img_size = 64
    image = torch.zeros(img_size, img_size, device=device)
    
    # Add some peaks
    peak_locs = [(10, 10), (30, 30), (50, 50)]
    peak_vals = [5.0, 3.0, 4.0]
    
    for (i, j), val in zip(peak_locs, peak_vals):
        image[i, j] = val
    
    # Add some noise
    image += torch.randn_like(image) * 0.1
    
    # Find peaks
    positions, heights = find_peaks_2d(image, threshold=2.0, ordered=True)
    
    print(f"Found {len(positions)} peaks")
    print(f"Peak positions: {positions[:5]}")
    print(f"Peak heights: {heights[:5]}")
    
    # Test histogram
    bins = torch.linspace(0, 6, 31, device=device)
    counts = peaks_to_histogram(heights, bins)
    print(f"Histogram shape: {counts.shape}")
    
    # Test mono-scale peaks
    bin_centers, counts, (pos, hts) = mono_scale_peaks_smoothed(
        image,
        sigma_noise=0.1,
        smoothing_sigma=2.0
    )
    print(f"Mono-scale: found {len(pos)} peaks")
    print(f"Histogram bins: {len(bin_centers)}")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_peaks()
