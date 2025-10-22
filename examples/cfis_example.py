"""
CFIS-like simulation example.

This example demonstrates usage with realistic CFIS survey parameters.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from wl_stats_torch import WLStatistics
from wl_stats_torch.visualization import (
    plot_peak_histograms,
    plot_l1_norms,
    plot_comparison
)


def create_power_law_field(size, power_index=-3.0, seed=None):
    """
    Create a field with power-law power spectrum.
    
    Args:
        size: Image size (assumes square)
        power_index: Power spectrum index (typically -3 for CDM)
        seed: Random seed for reproducibility
    
    Returns:
        2D field with specified power spectrum
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create frequency grid
    kx = torch.fft.fftfreq(size) * size
    ky = torch.fft.fftfreq(size) * size
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k = torch.sqrt(kx**2 + ky**2)
    k[0, 0] = 1  # Avoid division by zero
    
    # Power spectrum P(k) ∝ k^power_index
    power_spectrum = k ** power_index
    power_spectrum[0, 0] = 0  # Zero mean
    
    # Generate random Fourier coefficients
    amplitude = torch.sqrt(power_spectrum)
    phase = torch.rand(size, size) * 2 * np.pi
    fourier_field = amplitude * torch.exp(1j * phase)
    
    # Inverse FFT to get real space field
    field = torch.fft.ifft2(fourier_field).real
    
    # Normalize
    field = field / field.std()
    
    return field


def main():
    print("=" * 70)
    print("CFIS-like Weak Lensing Simulation")
    print("=" * 70)
    
    # CFIS-like parameters
    SHAPE_NOISE = 0.44
    PIX_ARCMIN = 0.4
    N_GAL = 7  # galaxies per square arcmin
    IMG_SIZE = 512
    N_SCALES = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print(f"\nCFIS Survey Parameters:")
    print(f"  Shape noise: {SHAPE_NOISE}")
    print(f"  Pixel resolution: {PIX_ARCMIN} arcmin/pixel")
    print(f"  Galaxy density: {N_GAL} gal/arcmin²")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE} pixels")
    print(f"  Field of view: {IMG_SIZE * PIX_ARCMIN:.1f} x {IMG_SIZE * PIX_ARCMIN:.1f} arcmin²")
    
    # Calculate noise level
    sigma_noise = SHAPE_NOISE / np.sqrt(2 * N_GAL * PIX_ARCMIN**2)
    print(f"  Effective noise per pixel: {sigma_noise:.4f}")
    
    # Generate realistic convergence field
    print("\nGenerating convergence field with power-law spectrum...")
    kappa_true = create_power_law_field(IMG_SIZE, power_index=-2.5, seed=42)
    kappa_true = kappa_true.to(device) * 0.03  # Scale to realistic amplitude
    
    # Add noise
    print("Adding shape noise...")
    noise = torch.randn(IMG_SIZE, IMG_SIZE, device=device) * sigma_noise
    kappa_noisy = kappa_true + noise
    
    # Create noise map
    sigma_map = torch.ones(IMG_SIZE, IMG_SIZE, device=device) * sigma_noise
    
    # Create realistic mask (survey footprint)
    print("Creating survey mask...")
    mask = torch.ones(IMG_SIZE, IMG_SIZE, device=device)
    
    # Remove corners (typical survey geometry)
    corner_size = IMG_SIZE // 4
    mask[:corner_size, :corner_size] = 0
    mask[:corner_size, -corner_size:] = 0
    mask[-corner_size:, :corner_size] = 0
    mask[-corner_size:, -corner_size:] = 0
    
    # Add some random holes (bad pixels, bright stars, etc.)
    n_holes = 20
    for _ in range(n_holes):
        x = torch.randint(50, IMG_SIZE-50, (1,)).item()
        y = torch.randint(50, IMG_SIZE-50, (1,)).item()
        hole_size = torch.randint(5, 15, (1,)).item()
        mask[x-hole_size:x+hole_size, y-hole_size:y+hole_size] = 0
    
    observed_fraction = mask.sum() / mask.numel()
    print(f"  Observed fraction: {observed_fraction:.1%}")
    
    # Visualize input
    print("\nVisualizing input...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    im0 = axes[0, 0].imshow(kappa_true.cpu().numpy(), cmap='RdBu_r', 
                             origin='lower', vmin=-0.05, vmax=0.05)
    axes[0, 0].set_title('True Convergence Field')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(kappa_noisy.cpu().numpy(), cmap='RdBu_r',
                             origin='lower', vmin=-0.05, vmax=0.05)
    axes[0, 1].set_title('Noisy Convergence Field')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow((kappa_noisy * mask).cpu().numpy(), cmap='RdBu_r',
                             origin='lower', vmin=-0.05, vmax=0.05)
    axes[1, 0].set_title('Masked Noisy Convergence')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(mask.cpu().numpy(), cmap='gray', origin='lower')
    axes[1, 1].set_title('Survey Mask')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('cfis_input_data.png', dpi=150)
    print("Saved to 'cfis_input_data.png'")
    plt.close()
    
    # Initialize statistics
    print("\nInitializing WLStatistics...")
    stats = WLStatistics(
        n_scales=N_SCALES,
        device=device,
        pixel_arcmin=PIX_ARCMIN
    )
    
    print("\nScale resolutions:")
    for i, res in enumerate(stats.get_scale_resolutions()):
        print(f"  Scale {i+1}: {res:.2f} arcmin")
    
    # Compute statistics
    print("\nComputing statistics...")
    results = stats.compute_all_statistics(
        kappa_noisy,
        sigma_map,
        mask=mask,
        min_snr=-2,
        max_snr=6,
        n_bins=31,
        l1_nbins=40,
        compute_mono=True,
        mono_smoothing_sigma=2.0,
        verbose=True
    )
    
    # Results summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    print("\nWavelet Peak Counts:")
    for i, counts in enumerate(results['wavelet_peak_counts']):
        n_peaks = int(counts.sum().item())
        if len(results['wavelet_peak_heights'][i]) > 0:
            max_snr = results['wavelet_peak_heights'][i].max().item()
            mean_snr = results['wavelet_peak_heights'][i].mean().item()
            print(f"  Scale {i+1}: {n_peaks:5d} peaks | "
                  f"max SNR: {max_snr:6.2f} | mean SNR: {mean_snr:6.2f}")
        else:
            print(f"  Scale {i+1}: {n_peaks:5d} peaks")
    
    print(f"\nMono-scale peaks: {int(results['mono_peak_counts'].sum().item())} peaks")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Peak histograms
    plot_peak_histograms(
        results['peak_bins'],
        results['wavelet_peak_counts'],
        title="CFIS-like: Wavelet Peak Counts",
        log_scale=True,
        save_path='cfis_peak_histograms.png'
    )
    
    # L1-norms
    plot_l1_norms(
        results['l1_bins'],
        results['wavelet_l1_norms'],
        title="CFIS-like: Wavelet L1-Norms",
        save_path='cfis_l1_norms.png'
    )
    
    # Compare with no-mask case
    print("\nComputing statistics without mask for comparison...")
    results_nomask = stats.compute_all_statistics(
        kappa_noisy,
        sigma_map,
        mask=None,
        min_snr=-2,
        max_snr=6,
        n_bins=31,
        l1_nbins=40,
        compute_mono=False,
        verbose=False
    )
    
    # Comparison plot
    plot_comparison(
        [results, results_nomask],
        ['With Mask', 'Without Mask'],
        statistic='wavelet_peak_counts',
        scale_idx=1,
        title='CFIS-like: Effect of Masking (Scale 2)',
        save_path='cfis_mask_comparison.png'
    )
    
    print("\n" + "=" * 70)
    print("✓ CFIS example completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - cfis_input_data.png")
    print("  - cfis_peak_histograms.png")
    print("  - cfis_l1_norms.png")
    print("  - cfis_mask_comparison.png")


if __name__ == "__main__":
    main()
