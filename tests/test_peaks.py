"""
Unit tests for peak detection.
"""

import pytest
import torch

from wl_stats_torch.peaks import (
    MonoScalePeakResult,
    PeakResult,
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

        result = find_peaks_2d(image, threshold=2.0, ordered=True)

        # Should find all 3 peaks
        assert len(result.positions) == 3
        assert len(result.heights) == 3

        # Check ordering (should be descending)
        assert result.heights[0] >= result.heights[1] >= result.heights[2]
        assert torch.allclose(result.heights[0], torch.tensor(5.0, device=device))

    def test_find_peaks_returns_namedtuple(self, device):
        """Test that find_peaks_2d returns a PeakResult NamedTuple."""
        image = torch.zeros(50, 50, device=device)
        image[25, 25] = 5.0

        result = find_peaks_2d(image)
        assert isinstance(result, PeakResult)

        # Should also work with tuple destructuring
        positions, heights = find_peaks_2d(image)
        assert positions.shape[1] == 2

    def test_find_peaks_threshold(self, device):
        """Test threshold filtering."""
        image = torch.zeros(50, 50, device=device)
        image[10, 10] = 5.0
        image[25, 25] = 2.0  # Below threshold
        image[40, 40] = 4.0

        result = find_peaks_2d(image, threshold=3.0)

        # Should only find 2 peaks above threshold
        assert len(result.positions) == 2
        assert torch.all(result.heights >= 3.0)

    def test_find_peaks_with_mask(self, device):
        """Test peak detection with mask."""
        image = torch.zeros(50, 50, device=device)
        image[10, 10] = 5.0
        image[25, 25] = 3.0
        image[40, 40] = 4.0

        # Mask out middle peak
        mask = torch.ones(50, 50, device=device)
        mask[20:30, 20:30] = 0

        result = find_peaks_2d(image, mask=mask)

        # Should only find 2 peaks (middle one masked)
        assert len(result.positions) == 2

    def test_find_peaks_border(self, device):
        """Test border handling."""
        image = torch.zeros(50, 50, device=device)
        image[0, 0] = 5.0  # Corner
        image[25, 25] = 3.0  # Center

        # Without border
        result = find_peaks_2d(image, include_border=False)
        assert len(result.positions) == 1  # Only center peak

        # With border
        result = find_peaks_2d(image, include_border=True)
        assert len(result.positions) == 2  # Both peaks

    def test_find_peaks_no_peaks(self, device):
        """Test behavior when no peaks found."""
        image = torch.randn(50, 50, device=device) * 0.1  # Noise only

        result = find_peaks_2d(image, threshold=10.0)  # Very high threshold

        assert len(result.positions) == 0
        assert len(result.heights) == 0
        assert result.positions.shape == (0, 2)

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
        assert isinstance(results[0], PeakResult)
        assert len(results[0].positions) == 1  # First image: 1 peak
        assert len(results[1].positions) == 2  # Second image: 2 peaks
        assert len(results[2].positions) == 3  # Third image: 3 peaks

    def test_find_peaks_batch_3d_input(self, device):
        """Test batch peak detection with (B, H, W) input."""
        images = torch.zeros(2, 50, 50, device=device)
        images[0, 25, 25] = 5.0
        images[1, 30, 30] = 4.0

        results = find_peaks_batch(images)
        assert len(results) == 2
        assert isinstance(results[0], PeakResult)

    def test_peaks_to_histogram(self, device):
        """Test histogram creation from peaks."""
        heights = torch.tensor([1.5, 2.5, 3.5, 4.5], device=device)
        bins = torch.tensor([0, 2, 4, 6], device=device)

        counts = peaks_to_histogram(heights, bins)

        assert len(counts) == 3  # n_bins = len(bins) - 1
        assert counts[0] == 1  # One peak in [0, 2): 1.5
        assert counts[1] == 2  # Two peaks in [2, 4): 2.5, 3.5
        assert counts[2] == 1  # One peak in [4, 6]: 4.5

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

        result = mono_scale_peaks_smoothed(image, sigma_noise=0.1, smoothing_sigma=2.0, n_bins=31)

        assert isinstance(result, MonoScalePeakResult)
        assert len(result.bin_centers) == 31
        assert len(result.counts) == 31
        assert len(result.peaks.positions) > 0
        assert len(result.peaks.heights) > 0

    def test_mono_scale_peaks_destructuring(self, device):
        """Test that mono_scale_peaks_smoothed result can be destructured."""
        image = torch.randn(64, 64, device=device)

        # Old-style destructuring should still work
        bin_centers, counts, (positions, heights) = mono_scale_peaks_smoothed(
            image, sigma_noise=0.1, smoothing_sigma=2.0, n_bins=31
        )

        assert len(bin_centers) == 31
        assert len(counts) == 31


class TestDtypePreservation:
    """Test that peak detection preserves dtypes."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_float32_peaks(self, device):
        """Peak heights should preserve float32 dtype."""
        image = torch.zeros(50, 50, device=device, dtype=torch.float32)
        image[25, 25] = 5.0

        result = find_peaks_2d(image)
        assert result.heights.dtype == torch.float32

    def test_float64_peaks(self, device):
        """Peak heights should preserve float64 dtype."""
        image = torch.zeros(50, 50, device=device, dtype=torch.float64)
        image[25, 25] = 5.0

        result = find_peaks_2d(image)
        assert result.heights.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
