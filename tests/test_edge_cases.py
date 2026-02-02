"""
Edge case tests for wl_stats_torch.
"""

import pytest
import torch

from wl_stats_torch.peaks import find_peaks_2d, find_peaks_batch
from wl_stats_torch.starlet import Starlet2D
from wl_stats_torch.statistics import WLStatistics


class TestNaNInfPropagation:
    """Test behavior with NaN/Inf inputs."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_nan_image_peaks(self, device):
        """NaN pixels should not be detected as peaks."""
        image = torch.randn(50, 50, device=device)
        image[25, 25] = float("nan")

        positions, heights = find_peaks_2d(image, threshold=0.0)
        # NaN should not appear in heights
        assert not torch.any(torch.isnan(heights))

    def test_inf_image_peaks(self, device):
        """Inf pixel should be the highest peak."""
        image = torch.randn(50, 50, device=device)
        image[25, 25] = float("inf")

        positions, heights = find_peaks_2d(image, ordered=True)
        assert heights[0] == float("inf")
        assert positions[0, 0] == 25
        assert positions[0, 1] == 25


class TestAllZeroImages:
    """Test behavior with all-zero images."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_zero_image_no_peaks(self, device):
        """All-zero image should have no peaks (no pixel > all neighbors)."""
        image = torch.zeros(50, 50, device=device)
        positions, heights = find_peaks_2d(image)
        assert len(positions) == 0

    def test_zero_image_statistics(self, device):
        """Statistics pipeline should handle all-zero images."""
        stats = WLStatistics(n_scales=3, device=device)
        image = torch.zeros(64, 64, device=device, dtype=torch.float64)
        sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.1

        results = stats.compute_all_statistics(image, sigma, compute_mono=True, verbose=False)
        assert "wavelet_peak_counts" in results

    def test_zero_image_starlet(self, device):
        """Starlet of zero image should produce zero coefficients."""
        starlet = Starlet2D(n_scales=3, device=device)
        image = torch.zeros(1, 1, 64, 64, device=device)
        coeffs = starlet(image)
        assert torch.allclose(coeffs, torch.zeros_like(coeffs))


class TestNonSquareImages:
    """Test behavior with non-square images."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_peaks_nonsquare(self, device):
        """Peak detection should work with non-square images."""
        image = torch.randn(128, 64, device=device)
        image[50, 30] = 10.0

        positions, heights = find_peaks_2d(image, threshold=5.0)
        assert len(positions) >= 1
        assert any((positions[:, 0] == 50) & (positions[:, 1] == 30))

    def test_starlet_nonsquare(self, device):
        """Starlet transform should work with non-square images."""
        starlet = Starlet2D(n_scales=3, device=device)
        image = torch.randn(1, 1, 128, 64, device=device)
        coeffs = starlet(image)
        assert coeffs.shape == (1, 3, 128, 64)

    def test_statistics_nonsquare(self, device):
        """Statistics pipeline should work with non-square images."""
        stats = WLStatistics(n_scales=3, device=device)
        image = torch.randn(128, 64, device=device, dtype=torch.float64)
        sigma = torch.ones(128, 64, device=device, dtype=torch.float64) * 0.1

        results = stats.compute_all_statistics(image, sigma, compute_mono=True, verbose=False)
        assert results["wavelet_coeffs"].shape == (3, 128, 64)


class TestVerySmallImages:
    """Test behavior with very small images."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_4x4_image_starlet(self, device):
        """Starlet should work with 4x4 images."""
        starlet = Starlet2D(n_scales=2, device=device)
        image = torch.randn(1, 1, 4, 4, device=device)
        coeffs = starlet(image)
        assert coeffs.shape == (1, 2, 4, 4)

    def test_4x4_image_peaks(self, device):
        """Peak detection should work with 4x4 images (only interior pixels)."""
        image = torch.zeros(4, 4, device=device)
        image[2, 2] = 5.0

        positions, heights = find_peaks_2d(image, include_border=False)
        assert len(positions) == 1

    def test_2x2_image_peaks_no_border(self, device):
        """2x2 image with no border should find no peaks."""
        image = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)

        positions, heights = find_peaks_2d(image, include_border=False)
        assert len(positions) == 0


class TestSNRValidation:
    """Test SNR range validation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def stats(self, device):
        stats = WLStatistics(n_scales=3, device=device)
        image = torch.randn(64, 64, device=device, dtype=torch.float64)
        sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.1
        stats.compute_wavelet_transform(image, sigma)
        return stats

    def test_min_snr_equals_max_snr_peaks(self, stats):
        """min_snr == max_snr should raise ValueError."""
        with pytest.raises(ValueError, match="min_snr"):
            stats.compute_wavelet_peak_counts(min_snr=3.0, max_snr=3.0)

    def test_min_snr_greater_than_max_snr_peaks(self, stats):
        """min_snr > max_snr should raise ValueError."""
        with pytest.raises(ValueError, match="min_snr"):
            stats.compute_wavelet_peak_counts(min_snr=5.0, max_snr=2.0)

    def test_min_snr_equals_max_snr_l1(self, stats):
        """min_snr == max_snr should raise ValueError for L1 norms."""
        with pytest.raises(ValueError, match="min_snr"):
            stats.compute_wavelet_l1_norms(min_snr=3.0, max_snr=3.0)

    def test_min_snr_greater_than_max_snr_mono(self, device):
        """min_snr > max_snr should raise ValueError for mono peaks."""
        stats = WLStatistics(n_scales=3, device=device)
        image = torch.randn(64, 64, device=device, dtype=torch.float64)

        with pytest.raises(ValueError, match="min_snr"):
            stats.compute_mono_scale_peaks(image, noise_sigma=0.1, min_snr=5.0, max_snr=2.0)


class TestBatchPeaks3DInput:
    """Test that find_peaks_batch accepts (B, H, W) input."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_3d_input(self, device):
        """find_peaks_batch should accept (B, H, W) and auto-add channel dim."""
        images = torch.zeros(2, 50, 50, device=device)
        images[0, 25, 25] = 5.0
        images[1, 30, 30] = 4.0

        results = find_peaks_batch(images)
        assert len(results) == 2
        assert len(results[0].positions) >= 1
        assert len(results[1].positions) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
