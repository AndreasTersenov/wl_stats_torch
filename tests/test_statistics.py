"""
Unit tests for statistics module.
"""

import torch
import pytest
from wl_stats_torch.statistics import WLStatistics


class TestWLStatistics:
    """Test suite for WLStatistics class."""
    
    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def stats(self, device):
        """Create WLStatistics instance."""
        return WLStatistics(n_scales=5, device=device, pixel_arcmin=0.5)
    
    @pytest.fixture
    def test_data(self, device):
        """Create test convergence and noise maps."""
        kappa = torch.randn(128, 128, device=device) * 0.01
        sigma = torch.ones(128, 128, device=device) * 0.02
        return kappa, sigma
    
    def test_initialization(self, device):
        """Test WLStatistics initialization."""
        stats = WLStatistics(n_scales=5, device=device, pixel_arcmin=0.5)
        
        assert stats.n_scales == 5
        assert stats.device == device
        assert stats.pixel_arcmin == 0.5
        assert stats.starlet is not None
    
    def test_get_scale_resolutions(self, stats):
        """Test scale resolution calculation."""
        resolutions = stats.get_scale_resolutions()
        
        assert len(resolutions) == 5
        # Each scale should double
        assert resolutions[0] == 1.0  # 2^1 * 0.5
        assert resolutions[1] == 2.0  # 2^2 * 0.5
        assert resolutions[2] == 4.0  # 2^3 * 0.5
    
    def test_compute_wavelet_transform(self, stats, test_data):
        """Test wavelet transform computation."""
        kappa, sigma = test_data
        
        results = stats.compute_wavelet_transform(kappa, sigma)
        
        assert 'wavelet_coeffs' in results
        assert 'noise_levels' in results
        assert 'snr' in results
        
        assert results['wavelet_coeffs'].shape == (5, 128, 128)
        assert results['noise_levels'].shape == (5, 128, 128)
        assert results['snr'].shape == (5, 128, 128)
    
    def test_compute_wavelet_peak_counts(self, stats, test_data):
        """Test wavelet peak count computation."""
        kappa, sigma = test_data
        
        # First compute transform
        stats.compute_wavelet_transform(kappa, sigma)
        
        # Then compute peaks
        bin_centers, peak_counts = stats.compute_wavelet_peak_counts(
            min_snr=-2, max_snr=6, n_bins=31
        )
        
        assert len(bin_centers) == 31
        assert len(peak_counts) == 5  # One per scale
        
        for counts in peak_counts:
            assert len(counts) == 31
            assert torch.all(counts >= 0)
    
    def test_compute_wavelet_l1_norms(self, stats, test_data):
        """Test L1-norm computation."""
        kappa, sigma = test_data
        
        # First compute transform
        stats.compute_wavelet_transform(kappa, sigma)
        
        # Then compute L1-norms
        bins_list, l1_norms_list = stats.compute_wavelet_l1_norms(n_bins=40)
        
        assert len(bins_list) == 5
        assert len(l1_norms_list) == 5
        
        for bins, norms in zip(bins_list, l1_norms_list):
            assert len(bins) == 40
            assert len(norms) == 40
            assert torch.all(norms >= 0)
    
    def test_compute_mono_scale_peaks(self, stats, test_data):
        """Test mono-scale peak computation."""
        kappa, sigma = test_data
        
        # Use scalar noise
        noise_sigma = sigma[0, 0].item()
        
        bin_centers, counts = stats.compute_mono_scale_peaks(
            kappa,
            noise_sigma=noise_sigma,
            smoothing_sigma=2.0,
            n_bins=31
        )
        
        assert len(bin_centers) == 31
        assert len(counts) == 31
        assert torch.all(counts >= 0)
    
    def test_compute_all_statistics(self, stats, test_data):
        """Test full statistics pipeline."""
        kappa, sigma = test_data
        
        results = stats.compute_all_statistics(
            kappa,
            sigma,
            min_snr=-2,
            max_snr=6,
            n_bins=31,
            l1_nbins=40,
            compute_mono=True,
            verbose=False
        )
        
        # Check all expected keys
        expected_keys = [
            'wavelet_coeffs',
            'noise_levels',
            'snr',
            'peak_bins',
            'wavelet_peak_counts',
            'wavelet_peak_positions',
            'wavelet_peak_heights',
            'l1_bins',
            'wavelet_l1_norms',
            'mono_peak_bins',
            'mono_peak_counts'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check shapes
        assert len(results['wavelet_peak_counts']) == 5
        assert len(results['wavelet_l1_norms']) == 5
        assert len(results['mono_peak_counts']) == 31
    
    def test_with_mask(self, stats, test_data):
        """Test statistics with mask."""
        kappa, sigma = test_data
        
        # Create mask
        mask = torch.ones(128, 128, device=stats.device)
        mask[:30, :] = 0  # Mask out top region
        
        results = stats.compute_all_statistics(
            kappa,
            sigma,
            mask=mask,
            compute_mono=True,
            verbose=False
        )
        
        # Should complete without errors
        assert 'wavelet_peak_counts' in results
    
    def test_scalar_noise(self, stats, test_data):
        """Test with scalar noise value instead of map."""
        kappa, _ = test_data
        
        results = stats.compute_all_statistics(
            kappa,
            noise_sigma=0.02,  # Scalar
            compute_mono=True,
            verbose=False
        )
        
        assert 'wavelet_peak_counts' in results
    
    def test_device_transfer(self):
        """Test moving statistics to different device."""
        stats_cpu = WLStatistics(n_scales=5, device=torch.device('cpu'))
        
        assert stats_cpu.device.type == 'cpu'
        
        if torch.cuda.is_available():
            stats_cpu.to(torch.device('cuda'))
            assert stats_cpu.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
