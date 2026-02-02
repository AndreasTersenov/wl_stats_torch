"""
Smoke tests for visualization functions.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, must be set before importing pyplot

import pytest  # noqa: E402
import torch  # noqa: E402

from wl_stats_torch.statistics import WLStatistics  # noqa: E402
from wl_stats_torch.visualization import (  # noqa: E402
    plot_comparison,
    plot_l1_norms,
    plot_peak_histograms,
    plot_snr_map,
    plot_wavelet_scales,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def single_results(device):
    """Run statistics pipeline to get single-image results for plotting."""
    stats = WLStatistics(n_scales=3, device=device)
    image = torch.randn(64, 64, device=device, dtype=torch.float64)
    sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.1
    return stats.compute_all_statistics(image, sigma, compute_mono=True, verbose=False)


@pytest.fixture
def batch_results(device):
    """Run statistics pipeline to get batch results for plotting."""
    stats = WLStatistics(n_scales=3, device=device)
    images = torch.randn(2, 64, 64, device=device, dtype=torch.float64)
    sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.1
    return stats.compute_all_statistics(images, sigma, compute_mono=True, verbose=False)


class TestPlotPeakHistograms:
    """Smoke tests for plot_peak_histograms."""

    def test_single_image(self, single_results):
        """Should not raise with valid single-image results."""
        plot_peak_histograms(
            single_results["peak_bins"],
            single_results["wavelet_peak_counts"],
        )

    def test_with_labels(self, single_results):
        """Should not raise with custom labels."""
        plot_peak_histograms(
            single_results["peak_bins"],
            single_results["wavelet_peak_counts"],
            scale_labels=["S1", "S2", "S3"],
        )


class TestPlotL1Norms:
    """Smoke tests for plot_l1_norms."""

    def test_single_image(self, single_results):
        """Should not raise with valid single-image results."""
        plot_l1_norms(
            single_results["l1_bins"],
            single_results["wavelet_l1_norms"],
        )

    def test_with_xlim(self, single_results):
        """Should not raise with xlim parameter."""
        plot_l1_norms(
            single_results["l1_bins"],
            single_results["wavelet_l1_norms"],
            xlim=(-2.0, 2.0),
        )


class TestPlotWaveletScales:
    """Smoke tests for plot_wavelet_scales."""

    def test_single_image(self, single_results):
        """Should not raise with valid wavelet coefficients."""
        plot_wavelet_scales(single_results["wavelet_coeffs"])

    def test_with_peaks(self, single_results):
        """Should not raise when marking peaks."""
        plot_wavelet_scales(
            single_results["wavelet_coeffs"],
            peak_positions=single_results["wavelet_peak_positions"],
            mark_peaks=True,
        )


class TestPlotSNRMap:
    """Smoke tests for plot_snr_map."""

    def test_default(self, single_results):
        """Should not raise with valid SNR coefficients."""
        plot_snr_map(single_results["snr"], scale_idx=0)

    def test_with_peaks(self, single_results):
        """Should not raise when marking peaks."""
        plot_snr_map(
            single_results["snr"],
            scale_idx=0,
            peak_positions=single_results["wavelet_peak_positions"][0],
        )


class TestPlotComparison:
    """Smoke tests for plot_comparison."""

    def test_compare_two_results(self, device):
        """Should not raise when comparing two result sets."""
        stats = WLStatistics(n_scales=3, device=device)

        image1 = torch.randn(64, 64, device=device, dtype=torch.float64)
        image2 = torch.randn(64, 64, device=device, dtype=torch.float64)
        sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.1

        results1 = stats.compute_all_statistics(image1, sigma, compute_mono=True, verbose=False)
        results2 = stats.compute_all_statistics(image2, sigma, compute_mono=True, verbose=False)

        plot_comparison(
            [results1, results2],
            labels=["Run 1", "Run 2"],
            statistic="wavelet_peak_counts",
            scale_idx=0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
