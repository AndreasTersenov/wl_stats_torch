"""
Unit tests for difference-of-top-hats transform.
"""

import numpy as np
import pytest
import torch
from scipy import special as sp

from wl_stats_torch.doth import DifferenceOfTopHats2D


def wale_like_doth_numpy(image: np.ndarray, n_scales: int, base_radius: float = 1.0) -> np.ndarray:
    """Reference WALE-style DoTH: smooth(2R)-smooth(R) with dyadic scales."""
    height, width = image.shape
    fy = np.fft.fftfreq(height, d=1.0)
    fx = np.fft.fftfreq(width, d=1.0)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    k = 2.0 * np.pi * np.sqrt(kx**2 + ky**2)

    x_fft = np.fft.fft2(image)
    smoothed = []
    for scale_idx in range(n_scales):
        radius = base_radius * (2**scale_idx)
        kr = k * radius
        window = np.ones_like(kr)
        nz = kr != 0
        window[nz] = 2.0 * sp.j1(kr[nz]) / kr[nz]
        smoothed.append(np.fft.ifft2(x_fft * window).real)

    details = [smoothed[j + 1] - smoothed[j] for j in range(n_scales - 1)]
    return np.stack(details + [smoothed[-1]], axis=0)


class TestDifferenceOfTopHats2D:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def image(self, device):
        return torch.randn(64, 64, device=device, dtype=torch.float64)

    def test_initialization(self, device):
        doth = DifferenceOfTopHats2D(n_scales=5, base_radius=1.0, device=device, dtype=torch.float64)
        assert doth.n_scales == 5
        assert doth.base_radius == 1.0
        assert doth.device == device

    def test_invalid_params(self, device):
        with pytest.raises(ValueError):
            DifferenceOfTopHats2D(n_scales=1, device=device)
        with pytest.raises(ValueError):
            DifferenceOfTopHats2D(n_scales=5, base_radius=0.0, device=device)

    def test_forward_shape(self, device, image):
        n_scales = 5
        doth = DifferenceOfTopHats2D(n_scales=n_scales, device=device, dtype=torch.float64)

        coeffs_2d = doth(image)
        assert coeffs_2d.shape == (1, n_scales, 64, 64)

        coeffs_3d = doth(image.unsqueeze(0))
        assert coeffs_3d.shape == (1, n_scales, 64, 64)

        coeffs_4d = doth(image.unsqueeze(0).unsqueeze(0))
        assert coeffs_4d.shape == (1, n_scales, 64, 64)

        details = doth(image, return_coarse=False)
        assert details.shape == (1, n_scales - 1, 64, 64)

    def test_noise_levels_shape_and_positive(self, device):
        doth = DifferenceOfTopHats2D(n_scales=4, base_radius=1.0, device=device, dtype=torch.float64)
        sigma = torch.ones(64, 64, device=device, dtype=torch.float64) * 0.02

        noise_levels = doth.get_noise_levels(sigma)
        assert noise_levels.shape == (1, 4, 64, 64)
        assert torch.all(noise_levels >= 0)

    def test_differentiable(self, device):
        doth = DifferenceOfTopHats2D(n_scales=4, base_radius=1.0, device=device, dtype=torch.float64)
        x = torch.randn(1, 1, 64, 64, device=device, dtype=torch.float64, requires_grad=True)
        coeffs = doth(x)
        loss = coeffs.square().mean()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_reference_consistency_against_wale_style_numpy(self):
        torch.manual_seed(0)
        image_t = torch.randn(32, 32, dtype=torch.float64)
        image_np = image_t.numpy()

        n_scales = 4
        base_radius = 1.0

        ref = wale_like_doth_numpy(image_np, n_scales=n_scales, base_radius=base_radius)
        doth = DifferenceOfTopHats2D(
            n_scales=n_scales, base_radius=base_radius, device=torch.device("cpu"), dtype=torch.float64
        )
        out = doth(image_t).squeeze(0).detach().numpy()

        assert np.allclose(out, ref, rtol=1e-6, atol=1e-6)

    def test_multichannel_error(self, device):
        doth = DifferenceOfTopHats2D(n_scales=4, device=device)
        bad = torch.randn(1, 3, 64, 64, device=device)
        with pytest.raises(ValueError):
            doth(bad)
