"""
Unit tests for peak detection.
"""

import pytest
import torch

from wl_stats_torch.peaks import (
    find_peaks_2d,
    find_peaks_batch,
    mono_scale_peaks_smoothed,
    peaks_to_histogram,
)


class TestPeakDetection:
    """Test suite for peak detection functions."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_find_peaks_simple(self, device):
        """Test peak detection on simple image with known peaks."""
        # Create image with 3 clear peaks
        image = torch.zeros(50, 50, device=device)
        image[10, 10] = 5.0
        image[25, 25] = 3.0
        image[40, 40] = 4.0

        positions, heights = find_peaks_2d(image, threshold=2.0, ordered=True)

        # Should find all 3 peaks
        assert len(positions) == 3
        assert len(heights) == 3

        # Check ordering (should be descending)
        assert heights[0] >= heights[1] >= heights[2]
        assert torch.allclose(heights[0], torch.tensor(5.0, device=device))

    def test_find_peaks_threshold(self, device):
        """Test threshold filtering."""
        image = torch.zeros(50, 50, device=device)
        image[10, 10] = 5.0
        image[25, 25] = 2.0  # Below threshold
        image[40, 40] = 4.0

        positions, heights = find_peaks_2d(image, threshold=3.0)

        # Should only find 2 peaks above threshold
        assert len(positions) == 2
        assert torch.all(heights >= 3.0)

    def test_find_peaks_with_mask(self, device):
        """Test peak detection with mask."""
        image = torch.zeros(50, 50, device=device)
        image[10, 10] = 5.0
        image[25, 25] = 3.0
        image[40, 40] = 4.0

        # Mask out middle peak
        mask = torch.ones(50, 50, device=device)
        mask[20:30, 20:30] = 0

        positions, heights = find_peaks_2d(image, mask=mask)

        # Should only find 2 peaks (middle one masked)
        assert len(positions) == 2

    def test_find_peaks_border(self, device):
        """Test border handling."""
        image = torch.zeros(50, 50, device=device)
        image[0, 0] = 5.0  # Corner
        image[25, 25] = 3.0  # Center

        # Without border
        positions, heights = find_peaks_2d(image, include_border=False)
        assert len(positions) == 1  # Only center peak

        # With border
        positions, heights = find_peaks_2d(image, include_border=True)
        assert len(positions) == 2  # Both peaks

    def test_find_peaks_no_peaks(self, device):
        """Test behavior when no peaks found."""
        image = torch.randn(50, 50, device=device) * 0.1  # Noise only

        positions, heights = find_peaks_2d(image, threshold=10.0)  # Very high threshold

        assert len(positions) == 0
        assert len(heights) == 0
        assert positions.shape == (0, 2)

    def test_find_peaks_batch(self, device):
        """Test batch peak detection."""
        batch_size = 3
        images = torch.zeros(batch_size, 1, 50, 50, device=device)

        # Add different numbers of peaks to each image
        images[0, 0, 10, 10] = 5.0
        images[1, 0, 15, 15] = 4.0
        images[1, 0, 30, 30] = 3.0
        images[2, 0, 20, 20] = 6.0
        images[2, 0, 35, 35] = 5.0
        images[2, 0, 40, 40] = 4.0

        results = find_peaks_batch(images)

        assert len(results) == batch_size
        assert len(results[0][0]) == 1  # First image: 1 peak
        assert len(results[1][0]) == 2  # Second image: 2 peaks
        assert len(results[2][0]) == 3  # Third image: 3 peaks

    def test_peaks_to_histogram(self, device):
        """Test histogram creation from peaks."""
        heights = torch.tensor([1.5, 2.5, 3.5, 4.5], device=device)
        bins = torch.tensor([0, 2, 4, 6], device=device)

        counts = peaks_to_histogram(heights, bins)

        assert len(counts) == 2  # n_bins - 1
        assert counts[0] == 1  # One peak in [0, 2)
        assert counts[1] == 3  # Three peaks in [2, 4)

    def test_peaks_to_histogram_empty(self, device):
        """Test histogram with no peaks."""
        heights = torch.tensor([], device=device)
        bins = torch.linspace(0, 10, 11, device=device)

        counts = peaks_to_histogram(heights, bins)

        assert len(counts) == 10
        assert torch.all(counts == 0)

    def test_mono_scale_peaks(self, device):
        """Test mono-scale peak detection with smoothing."""
        image = torch.randn(128, 128, device=device)

        # Add some strong peaks
        image[50, 50] += 2.0
        image[80, 80] += 1.5

        bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
            image, sigma_noise=0.1, smoothing_sigma=2.0, n_bins=31
        )

        assert len(bin_centers) == 31
        assert len(counts) == 31
        assert len(positions) > 0
        assert len(heights) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
