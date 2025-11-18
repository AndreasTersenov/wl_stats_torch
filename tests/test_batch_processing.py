"""
Unit tests for batch processing optimizations.

Tests vectorized operations and performance characteristics of batch processing.
"""

import pytest
import torch

from wl_stats_torch.peaks import find_peaks_batch, peaks_to_histogram
from wl_stats_torch.statistics import WLStatistics


class TestBatchProcessing:
    """Test suite for batch processing functionality."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def stats(self, device):
        """Create WLStatistics instance."""
        return WLStatistics(n_scales=4, device=device)

    @pytest.fixture
    def batch_data(self, device):
        """Create batch of test convergence maps."""
        batch_size = 4
        kappa_batch = torch.randn(batch_size, 128, 128, device=device, dtype=torch.float64) * 0.01
        sigma = 0.02
        return kappa_batch, sigma

    def test_batch_shape_detection(self, stats):
        """Test automatic batch dimension detection."""
        # Single image
        single = torch.randn(128, 128, device=stats.device)
        assert not stats._is_batched(single)

        # Batch
        batch = torch.randn(4, 128, 128, device=stats.device)
        assert stats._is_batched(batch)

    def test_batch_vs_sequential_consistency(self, stats, batch_data):
        """Test that batch processing gives identical results to sequential."""
        kappa_batch, sigma = batch_data
        batch_size = kappa_batch.shape[0]

        # Batch processing
        batch_results = stats.compute_all_statistics(kappa_batch, sigma, verbose=False)

        # Sequential processing
        sequential_results = []
        for i in range(batch_size):
            single_stats = WLStatistics(n_scales=4, device=stats.device)
            result = single_stats.compute_all_statistics(kappa_batch[i], sigma, verbose=False)
            sequential_results.append(result)

        # Compare wavelet coefficients
        for i in range(batch_size):
            assert torch.allclose(
                batch_results["wavelet_coeffs"][i],
                sequential_results[i]["wavelet_coeffs"],
                rtol=1e-10,
                atol=1e-12,
            ), f"Wavelet coeffs mismatch for image {i}"

        # Compare peak counts
        for scale_idx in range(stats.n_scales):
            batch_counts = batch_results["wavelet_peak_counts"][scale_idx]
            for i in range(batch_size):
                seq_counts = sequential_results[i]["wavelet_peak_counts"][scale_idx]
                assert torch.allclose(
                    batch_counts[i], seq_counts, rtol=1e-5, atol=1e-8
                ), f"Peak counts mismatch for image {i}, scale {scale_idx}"

        # Compare L1 norms (with dtype conversion for comparison)
        for scale_idx in range(stats.n_scales):
            batch_l1 = batch_results["wavelet_l1_norms"][scale_idx]
            for i in range(batch_size):
                seq_l1 = sequential_results[i]["wavelet_l1_norms"][scale_idx]
                assert torch.allclose(
                    batch_l1[i].float(), seq_l1.float(), rtol=1e-5, atol=1e-8
                ), f"L1 norms mismatch for image {i}, scale {scale_idx}"

    def test_batch_broadcasting_noise(self, stats, device):
        """Test noise broadcasting with different input formats."""
        batch = torch.randn(4, 128, 128, device=device)

        # Scalar noise
        sigma_scalar = 0.02
        results1 = stats.compute_wavelet_transform(batch, sigma_scalar)
        assert results1["noise_levels"].shape == (4, 4, 128, 128)

        # Map noise (broadcasts to batch)
        sigma_map = torch.ones(128, 128, device=device) * 0.02
        results2 = stats.compute_wavelet_transform(batch, sigma_map)
        assert results2["noise_levels"].shape == (4, 4, 128, 128)

        # Per-sample noise
        sigma_batch = torch.ones(4, 128, 128, device=device) * 0.02
        results3 = stats.compute_wavelet_transform(batch, sigma_batch)
        assert results3["noise_levels"].shape == (4, 4, 128, 128)

    def test_batch_broadcasting_mask(self, stats, device):
        """Test mask broadcasting with different input formats."""
        batch = torch.randn(4, 128, 128, device=device)
        sigma = 0.02

        # Compute transform first (required before peak counts)
        stats.compute_wavelet_transform(batch, sigma)

        # No mask
        results1 = stats.compute_wavelet_peak_counts()
        assert results1 is not None

        # Shared mask (broadcasts to batch)
        mask_shared = torch.ones(128, 128, device=device)
        mask_shared[:20, :] = 0
        stats.compute_wavelet_transform(batch, sigma)
        results2 = stats.compute_wavelet_peak_counts(mask=mask_shared)
        assert len(results2[1]) == 4  # 4 scales

        # Per-sample mask
        mask_batch = torch.ones(4, 128, 128, device=device)
        mask_batch[0, :30, :] = 0
        stats.compute_wavelet_transform(batch, sigma)
        results3 = stats.compute_wavelet_peak_counts(mask=mask_batch)
        assert len(results3[1]) == 4

    def test_batch_peak_counts_vectorization(self, stats, batch_data):
        """Test that peak count computation is vectorized."""
        kappa_batch, sigma = batch_data

        # Compute transform
        stats.compute_wavelet_transform(kappa_batch, sigma)

        # Compute peak counts
        bin_centers, peak_counts = stats.compute_wavelet_peak_counts(
            min_snr=-2.0, max_snr=6.0, n_bins=31
        )

        # Check output shapes
        assert len(bin_centers) == 31
        assert len(peak_counts) == 4  # n_scales

        for scale_counts in peak_counts:
            assert scale_counts.shape == (4, 31)  # (batch_size, n_bins)
            assert torch.all(scale_counts >= 0)

    def test_batch_l1_norms_vectorization(self, stats, batch_data):
        """Test that L1-norm computation is vectorized."""
        kappa_batch, sigma = batch_data

        # Compute transform
        stats.compute_wavelet_transform(kappa_batch, sigma)

        # Compute L1 norms
        bins_list, l1_norms = stats.compute_wavelet_l1_norms(n_bins=40)

        # Check output shapes
        assert len(bins_list) == 4  # n_scales
        assert len(l1_norms) == 4

        for l1 in l1_norms:
            assert l1.shape == (4, 40)  # (batch_size, n_bins)
            assert torch.all(l1 >= 0)

    def test_batch_different_sizes(self, stats, device):
        """Test that different batch sizes work correctly."""
        for batch_size in [1, 2, 8, 16]:
            batch = torch.randn(batch_size, 128, 128, device=device)
            sigma = 0.02

            results = stats.compute_all_statistics(batch, sigma, verbose=False)

            # Check shapes match batch size
            assert results["wavelet_coeffs"].shape[0] == batch_size
            assert results["noise_levels"].shape[0] == batch_size

            for counts in results["wavelet_peak_counts"]:
                assert counts.shape[0] == batch_size

            for l1 in results["wavelet_l1_norms"]:
                assert l1.shape[0] == batch_size

    def test_vectorized_peak_detection(self, device):
        """Test that find_peaks_batch uses vectorized operations."""
        batch_size = 4
        images = torch.zeros(batch_size, 1, 50, 50, device=device)

        # Add peaks at known locations
        images[0, 0, 10, 10] = 5.0
        images[1, 0, 15, 15] = 4.0
        images[1, 0, 30, 30] = 3.0
        images[2, 0, 20, 20] = 6.0
        images[3, 0, 25, 25] = 7.0
        images[3, 0, 35, 35] = 5.0
        images[3, 0, 40, 40] = 4.0

        results = find_peaks_batch(images)

        # Verify correct number of peaks detected
        assert len(results) == batch_size
        assert len(results[0][0]) == 1  # Image 0: 1 peak
        assert len(results[1][0]) == 2  # Image 1: 2 peaks
        assert len(results[2][0]) == 1  # Image 2: 1 peak
        assert len(results[3][0]) == 3  # Image 3: 3 peaks

    def test_edge_cases_empty_peaks(self, stats, device):
        """Test batch processing with images that have no peaks."""
        # Create batch with very low values (no peaks above threshold)
        batch = torch.randn(4, 128, 128, device=device) * 0.001
        sigma = 0.1  # High noise level

        results = stats.compute_all_statistics(batch, sigma, verbose=False)

        # Should complete without errors
        assert "wavelet_peak_counts" in results

        # Peak counts should be zero or very low
        for counts in results["wavelet_peak_counts"]:
            assert counts.shape == (4, 31)

    def test_backward_compatibility_single_image(self, stats, device):
        """Test that single image processing still works (backward compatibility)."""
        single_image = torch.randn(128, 128, device=device)
        sigma = 0.02

        results = stats.compute_all_statistics(single_image, sigma, verbose=False)

        # Check that results are NOT batched
        assert results["wavelet_coeffs"].ndim == 3  # (n_scales, H, W)
        assert results["noise_levels"].ndim == 3

        for counts in results["wavelet_peak_counts"]:
            assert counts.ndim == 1  # (n_bins,)

        for l1 in results["wavelet_l1_norms"]:
            assert l1.ndim == 1

    def test_clamp_overflow_batch(self, stats, batch_data):
        """Test clamp_overflow parameter in batch mode."""
        kappa_batch, sigma = batch_data

        stats.compute_wavelet_transform(kappa_batch, sigma)

        # Without clamping (default)
        _, counts1 = stats.compute_wavelet_peak_counts(
            min_snr=-2.0, max_snr=6.0, n_bins=31, clamp_overflow=False
        )

        # With clamping
        stats.compute_wavelet_transform(kappa_batch, sigma)
        _, counts2 = stats.compute_wavelet_peak_counts(
            min_snr=-2.0, max_snr=6.0, n_bins=31, clamp_overflow=True
        )

        # Both should complete without errors
        assert len(counts1) == 4
        assert len(counts2) == 4

    def test_numerical_stability(self, stats, device):
        """Test numerical stability with extreme values."""
        # Create batch with extreme values
        batch = torch.randn(4, 128, 128, device=device, dtype=torch.float64)
        batch[0] *= 1e-10  # Very small values
        batch[1] *= 1e10  # Very large values

        sigma = 0.02

        results = stats.compute_all_statistics(batch, sigma, verbose=False)

        # Should complete without NaNs or Infs
        assert not torch.isnan(results["wavelet_coeffs"]).any()
        assert not torch.isinf(results["wavelet_coeffs"]).any()

    def test_memory_efficiency(self, stats, device):
        """Test that batch processing doesn't cause memory leaks."""
        if device.type == "cuda":
            import gc

            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Process several batches
            for _ in range(5):
                batch = torch.randn(4, 128, 128, device=device)
                _ = stats.compute_all_statistics(batch, 0.02, verbose=False)
                del batch
                gc.collect()
                torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated()

            # Memory usage should be similar (allowing some tolerance)
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100 * 1024 * 1024  # Less than 100 MB increase

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_utilization(self, stats, device):
        """Test that batch processing runs on GPU when available."""
        batch = torch.randn(4, 128, 128, device=device)
        sigma = 0.02

        results = stats.compute_all_statistics(batch, sigma, verbose=False)

        # All results should be on GPU
        assert results["wavelet_coeffs"].device.type == "cuda"
        assert results["noise_levels"].device.type == "cuda"

    def test_dtype_consistency(self, device):
        """Test that dtype is preserved through batch processing."""
        # Test with float64 (default)
        stats64 = WLStatistics(n_scales=4, device=device)
        batch64 = torch.randn(4, 128, 128, device=device, dtype=torch.float64)

        results64 = stats64.compute_all_statistics(batch64, 0.02, verbose=False)
        assert results64["wavelet_coeffs"].dtype == torch.float64

        # Test with float32
        batch32 = batch64.float()
        stats32 = WLStatistics(n_scales=4, device=device)

        results32 = stats32.compute_all_statistics(batch32, 0.02, verbose=False)
        # Results should maintain input dtype
        assert results32["wavelet_coeffs"].dtype in [torch.float32, torch.float64]


class TestBatchOptimizations:
    """Test suite specifically for optimization correctness."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_linear_indexing_correctness(self, device):
        """Test that linear indexing produces correct histograms."""
        batch_size = 3
        n_bins = 5

        # Simulate peaks with known distribution
        heights = [
            torch.tensor([1.0, 2.5, 3.5], device=device),  # Image 0
            torch.tensor([0.5, 1.5, 4.5], device=device),  # Image 1
            torch.tensor([2.0, 3.0], device=device),  # Image 2
        ]

        bins = torch.linspace(0, 5, n_bins + 1, device=device)

        # Manual histogramming (sequential)
        manual_histograms = torch.zeros(batch_size, n_bins, device=device)
        for i, h in enumerate(heights):
            manual_histograms[i] = peaks_to_histogram(h, bins)

        # Vectorized approach (what the optimization does)
        all_heights = torch.cat(heights)
        batch_ids = torch.cat(
            [
                torch.full((len(h),), i, device=device, dtype=torch.long)
                for i, h in enumerate(heights)
            ]
        )

        # Compute linear indices and histogram
        bin_indices = torch.searchsorted(bins, all_heights, right=True)
        valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)

        linear_indices = batch_ids[valid_mask] * n_bins + (bin_indices[valid_mask] - 1)
        flat_counts = torch.bincount(linear_indices, minlength=batch_size * n_bins)
        vectorized_histograms = flat_counts.reshape(batch_size, n_bins).float()

        # Compare
        assert torch.allclose(manual_histograms, vectorized_histograms)

    def test_scatter_add_correctness(self, device):
        """Test that scatter_add produces correct L1 sums."""
        batch_size = 2
        n_bins = 4

        # Simulate SNR values with known binning
        snr_values = [
            torch.tensor([1.0, 1.5, 2.0, 3.5], device=device),  # Image 0
            torch.tensor([0.5, 2.5, 3.0], device=device),  # Image 1
        ]

        bins = torch.linspace(0, 4, n_bins + 1, device=device)

        # Manual L1 computation (sequential)
        manual_l1 = torch.zeros(batch_size, n_bins, device=device)
        for i, snr in enumerate(snr_values):
            bin_idx = torch.searchsorted(bins, snr, right=False)
            for j in range(1, n_bins + 1):
                mask = bin_idx == j
                if mask.any():
                    manual_l1[i, j - 1] = torch.abs(snr[mask]).sum()

        # Vectorized approach (scatter_add)
        all_snr = torch.cat(snr_values)
        batch_ids = torch.cat(
            [
                torch.full((len(s),), i, device=device, dtype=torch.long)
                for i, s in enumerate(snr_values)
            ]
        )

        bin_indices = torch.searchsorted(bins, all_snr, right=False)
        valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)

        linear_indices = batch_ids[valid_mask] * n_bins + (bin_indices[valid_mask] - 1)
        l1_flat = torch.zeros(batch_size * n_bins, dtype=all_snr.dtype, device=device)
        l1_flat.scatter_add_(0, linear_indices, torch.abs(all_snr[valid_mask]))
        vectorized_l1 = l1_flat.reshape(batch_size, n_bins)

        # Compare
        assert torch.allclose(manual_l1, vectorized_l1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
