"""
Unit tests for starlet transform.
"""

import torch
import pytest
from wl_stats_torch.starlet import Starlet2D


class TestStarlet2D:
    """Test suite for Starlet2D class."""
    
    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def image(self, device):
        """Create a test image."""
        return torch.randn(128, 128, device=device)
    
    def test_initialization(self, device):
        """Test Starlet2D initialization."""
        starlet = Starlet2D(n_scales=5, device=device)
        assert starlet.n_scales == 5
        assert starlet.device == device
        assert len(starlet.convs) == 4  # n_scales - 1
    
    def test_invalid_n_scales(self, device):
        """Test that invalid n_scales raises error."""
        with pytest.raises(ValueError):
            Starlet2D(n_scales=1, device=device)
    
    def test_forward_shape(self, device, image):
        """Test forward pass output shape."""
        n_scales = 5
        starlet = Starlet2D(n_scales=n_scales, device=device)
        
        # Test 2D input
        coeffs = starlet(image)
        assert coeffs.shape == (1, n_scales, 128, 128)
        
        # Test 3D input
        image_3d = image.unsqueeze(0)  # (1, H, W)
        coeffs = starlet(image_3d)
        assert coeffs.shape == (1, n_scales, 128, 128)
        
        # Test 4D input
        image_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        coeffs = starlet(image_4d)
        assert coeffs.shape == (1, n_scales, 128, 128)
    
    def test_reconstruction(self, device, image):
        """Test that reconstruction recovers original image."""
        starlet = Starlet2D(n_scales=5, device=device)
        
        # Transform
        coeffs = starlet(image)
        
        # Reconstruct
        reconstructed = starlet.reconstruct(coeffs, gen2=True)
        reconstructed = reconstructed.squeeze()
        
        # Check error
        max_error = torch.abs(image - reconstructed).max().item()
        assert max_error < 1e-5, f"Reconstruction error too large: {max_error}"
    
    def test_no_coarse_scale(self, device, image):
        """Test forward without coarse scale."""
        n_scales = 5
        starlet = Starlet2D(n_scales=n_scales, device=device)
        
        coeffs = starlet(image, return_coarse=False)
        assert coeffs.shape == (1, n_scales - 1, 128, 128)
    
    def test_noise_levels(self, device):
        """Test noise level calculation."""
        starlet = Starlet2D(n_scales=5, device=device)
        
        # Create constant noise map
        noise_sigma = torch.ones(128, 128, device=device) * 0.1
        
        noise_levels = starlet.get_noise_levels(noise_sigma)
        
        assert noise_levels.shape == (1, 5, 128, 128)
        assert torch.all(noise_levels >= 0)
    
    def test_snr_computation(self, device, image):
        """Test SNR computation."""
        starlet = Starlet2D(n_scales=5, device=device)
        
        noise_sigma = torch.ones(128, 128, device=device) * 0.1
        
        snr = starlet.get_snr(image, noise_sigma, keep_sign=False)
        
        assert snr.shape == (1, 5, 128, 128)
        assert torch.all(snr >= 0)  # Absolute SNR
        
        # Test with sign
        snr_signed = starlet.get_snr(image, noise_sigma, keep_sign=True)
        assert not torch.all(snr_signed >= 0)  # Some negative values
    
    def test_scale_resolution(self, device):
        """Test scale resolution calculation."""
        pixel_arcmin = 0.5
        starlet = Starlet2D(n_scales=5, device=device)
        
        resolutions = starlet.get_scale_resolution(pixel_arcmin)
        
        assert len(resolutions) == 5
        # Each scale should double the resolution
        for i in range(len(resolutions) - 1):
            assert resolutions[i+1] == 2 * resolutions[i]
    
    def test_multichannel_error(self, device):
        """Test that multi-channel input raises error."""
        starlet = Starlet2D(n_scales=5, device=device)
        
        # Create 3-channel image
        image_rgb = torch.randn(1, 3, 128, 128, device=device)
        
        with pytest.raises(ValueError):
            starlet(image_rgb)
    
    def test_device_transfer(self):
        """Test moving between devices."""
        # Start on CPU
        starlet = Starlet2D(n_scales=5, device=torch.device('cpu'))
        image_cpu = torch.randn(64, 64)
        
        coeffs_cpu = starlet(image_cpu)
        assert coeffs_cpu.device.type == 'cpu'
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            starlet.to(torch.device('cuda'))
            image_gpu = image_cpu.cuda()
            
            coeffs_gpu = starlet(image_gpu)
            assert coeffs_gpu.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
