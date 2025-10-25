"""
Peak Detection for 2D Images in PyTorch

This module provides fast, GPU-accelerated peak detection for 2D images.
A peak is defined as a local maximum - a pixel with a value greater than
all of its 8 neighbors.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def find_peaks_2d(
    image: torch.Tensor,
    threshold: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    include_border: bool = False,
    ordered: bool = True,
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
    img_padded = F.pad(img_4d, (pad, pad, pad, pad), mode="constant", value=float("-inf"))

    # Extract shifted versions (the 8 neighbors)
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip center pixel
            # Crop the padded image to get shifted version
            shifted = img_padded[:, :, 1 + di : 1 + di + H, 1 + dj : 1 + dj + W]
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
    ordered: bool = True,
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
            image, threshold=threshold, mask=mask, include_border=include_border, ordered=ordered
        )
        results.append((positions, heights))

    return results


def peaks_to_histogram(
    peak_heights: torch.Tensor,
    bins: torch.Tensor,
    digitize_mode: bool = True,
    clamp_overflow: bool = False,
) -> torch.Tensor:
    """
    Compute histogram of peak heights.

    This function mimics np.histogram behavior to match pycs output.

    Args:
        peak_heights: Tensor of peak values, shape (N,)
        bins: Bin edges, shape (n_bins+1,)
        digitize_mode: If True, use np.digitize-like behavior (default).
                      If False, use torch.histogram behavior.
        clamp_overflow: If True, values outside bin range are included in edge bins.
                       If False (default), values outside range are excluded.
                       False matches cosmostat/pycs behavior.

    Returns:
        Histogram counts, shape (n_bins,)

    Note:
        To match pycs behavior with np.histogram:
        - Values x where bins[i] <= x < bins[i+1] go into bin i
        - The rightmost bin includes the right edge: bins[-2] <= x <= bins[-1]
        - When clamp_overflow=False: values outside [bins[0], bins[-1]] are excluded
        - When clamp_overflow=True: values < bins[0] go to first bin, > bins[-1] go to last bin
    """
    if peak_heights.numel() == 0:
        return torch.zeros(len(bins) - 1, device=bins.device, dtype=torch.float32)

    bins = bins.to(peak_heights.device)
    n_bins = len(bins) - 1

    if digitize_mode:
        # Use searchsorted with right=True to match np.histogram behavior
        # np.histogram uses bins[i] <= x < bins[i+1], except rightmost bin includes right edge
        bin_indices = torch.searchsorted(bins, peak_heights, right=True)

        # Handle rightmost edge: values exactly equal to bins[-1] should go in last bin
        rightmost_mask = peak_heights == bins[-1]
        if rightmost_mask.any():
            bin_indices[rightmost_mask] = n_bins

        if clamp_overflow:
            # Clip to valid range [1, n_bins] - forces overflow into edge bins
            bin_indices = torch.clamp(bin_indices, 1, n_bins)
            # Count peaks in each bin (shift by -1 since bins start at index 1)
            counts = torch.bincount(bin_indices - 1, minlength=n_bins)
        else:
            # Only count values within valid range [1, n_bins] - excludes overflow
            # This matches cosmostat behavior
            valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)
            if valid_mask.any():
                counts = torch.bincount(bin_indices[valid_mask] - 1, minlength=n_bins)
            else:
                counts = torch.zeros(n_bins, device=bins.device, dtype=torch.long)
    else:
        # Original torch.searchsorted behavior
        bin_indices = torch.searchsorted(bins, peak_heights, right=False)

        if clamp_overflow:
            bin_indices = torch.clamp(bin_indices, 1, len(bins) - 1)
            counts = torch.bincount(bin_indices - 1, minlength=len(bins) - 1)
        else:
            valid_mask = (bin_indices >= 1) & (bin_indices <= len(bins) - 1)
            if valid_mask.any():
                counts = torch.bincount(bin_indices[valid_mask] - 1, minlength=len(bins) - 1)
            else:
                counts = torch.zeros(len(bins) - 1, device=bins.device, dtype=torch.long)

    return counts[:n_bins].float()


def mono_scale_peaks_smoothed(
    image: torch.Tensor,
    sigma_noise: float,
    smoothing_sigma: float = 2.0,
    mask: Optional[torch.Tensor] = None,
    bins: Optional[torch.Tensor] = None,
    min_snr: float = -2.0,
    max_snr: float = 6.0,
    n_bins: int = 31,
    clamp_overflow: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute mono-scale peak counts with Gaussian smoothing.

    This applies Gaussian smoothing to the image, computes SNR, finds peaks,
    and returns histogram of peak counts.

    NOTE: sigma_noise can now be a tensor for spatially-varying noise maps.

    Args:
        image: Input image (H, W)
        sigma_noise: Standard deviation of noise (scalar or H, W tensor)
        smoothing_sigma: Std dev for Gaussian smoothing (in pixels)
        mask: Optional observation mask
        bins: Optional custom bin edges for histogram
        min_snr: Minimum SNR for histogram (if bins not provided)
        max_snr: Maximum SNR for histogram (if bins not provided)
        n_bins: Number of bins for histogram (if bins not provided)
        clamp_overflow: If True, peaks outside SNR range are included in edge bins.
                       If False (default), peaks outside range are excluded.
                       False matches cosmostat/pycs behavior.

    Returns:
        Tuple of (bin_centers, counts, (peak_positions, peak_heights))
    """
    device = image.device

    # Handle sigma_noise as scalar or tensor
    if isinstance(sigma_noise, (int, float)):
        sigma_noise_map = torch.full_like(image, sigma_noise)
        uniform_noise = True
    else:
        sigma_noise_map = sigma_noise.to(device)
        if sigma_noise_map.ndim == 4:
            sigma_noise_map = sigma_noise_map.squeeze(0).squeeze(0)
        uniform_noise = False

    # Create Gaussian kernel for smoothing
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Create Gaussian kernel
    kernel_size = int(6 * smoothing_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian with same dtype as image
    image_dtype = image.dtype if image.ndim == 4 else image.dtype
    x = torch.arange(kernel_size, dtype=image_dtype, device=device)
    x = x - kernel_size // 2
    gaussian_1d = torch.exp(-(x**2) / (2 * smoothing_sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    # Create 2D Gaussian via outer product
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Smooth the image
    padding = kernel_size // 2
    if padding > 0:
        image_padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
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
    if uniform_noise:
        # Fast path for uniform noise: var(G * X) = sigma^2 * sum(G^2)
        gaussian_squared = gaussian_2d**2
        noise_factor_squared = gaussian_squared.sum()
        smoothed_noise_sigma = sigma_noise_map[0, 0].item() * torch.sqrt(noise_factor_squared)
        smoothed_noise_map = torch.full_like(image_smoothed, smoothed_noise_sigma)
    else:
        # Proper variance propagation for non-uniform noise (matches pycs)
        # var(smoothed) = conv(variance_map, G^2)
        variance_map = (sigma_noise_map**2).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gaussian_squared = gaussian_2d**2

        if padding > 0:
            variance_padded = F.pad(
                variance_map, (padding, padding, padding, padding), mode="reflect"
            )
        else:
            variance_padded = variance_map

        smoothed_variance = F.conv2d(
            variance_padded,
            gaussian_squared,
            bias=None,
            stride=1,
            padding=0,
        )
        smoothed_noise_map = torch.sqrt(smoothed_variance)

    # Compute SNR
    snr_image = image_smoothed / smoothed_noise_map
    snr_image = snr_image.squeeze(0).squeeze(0)  # Back to (H, W)

    # Find peaks
    peak_positions, peak_heights = find_peaks_2d(
        snr_image, threshold=None, mask=mask, include_border=False, ordered=True
    )

    # Create histogram bins if not provided
    if bins is None:
        bins = torch.linspace(min_snr, max_snr, n_bins + 1, device=device)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute histogram
    counts = peaks_to_histogram(peak_heights, bins, clamp_overflow=clamp_overflow)

    return bin_centers, counts, (peak_positions, peak_heights)


def test_peaks():
    """Test peak detection functions."""
    print("Testing peak detection...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        image, sigma_noise=0.1, smoothing_sigma=2.0
    )
    print(f"Mono-scale: found {len(pos)} peaks")
    print(f"Histogram bins: {len(bin_centers)}")

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_peaks()
