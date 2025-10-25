"""
Basic usage example for wl_stats_torch package.

This demonstrates the core functionality with synthetic data.
"""

import matplotlib.pyplot as plt
import torch

from wl_stats_torch import WLStatistics
from wl_stats_torch.visualization import (
    plot_l1_norms,
    plot_peak_histograms,
    plot_snr_map,
    plot_wavelet_scales,
)


def main():
    print("=" * 60)
    print("WL Statistics Torch - Basic Usage Example")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Parameters
    img_size = 256
    n_scales = 5
    pixel_arcmin = 0.5
    noise_level = 0.02

    print(f"\nParameters:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of scales: {n_scales}")
    print(f"  Pixel resolution: {pixel_arcmin} arcmin/pixel")
    print(f"  Noise level: {noise_level}")

    # Create synthetic convergence map
    print("\nGenerating synthetic data...")
    torch.manual_seed(42)

    # Create a convergence map with some structure
    kappa = torch.randn(img_size, img_size, device=device) * 0.01

    # Add some peaks
    n_peaks = 10
    for _ in range(n_peaks):
        x, y = torch.randint(50, img_size - 50, (2,))
        amplitude = torch.randn(1).item() * 0.05
        sigma = 10

        xx, yy = torch.meshgrid(
            torch.arange(img_size, device=device),
            torch.arange(img_size, device=device),
            indexing="ij",
        )
        gaussian = amplitude * torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        kappa += gaussian

    # Add noise
    noise = torch.randn(img_size, img_size, device=device) * noise_level
    kappa_noisy = kappa + noise

    # Constant noise map
    sigma_map = torch.ones(img_size, img_size, device=device) * noise_level

    # Optional: Create a mask (e.g., circular footprint)
    center = img_size // 2
    radius = img_size // 2 - 10
    xx, yy = torch.meshgrid(
        torch.arange(img_size, device=device) - center,
        torch.arange(img_size, device=device) - center,
        indexing="ij",
    )
    mask = ((xx**2 + yy**2) < radius**2).float()

    # Visualize input
    print("\nVisualizing input data...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(kappa.cpu().numpy(), cmap="viridis", origin="lower")
    axes[0].set_title("True Convergence")
    axes[0].colorbar()

    axes[1].imshow(kappa_noisy.cpu().numpy(), cmap="viridis", origin="lower")
    axes[1].set_title("Noisy Convergence")
    axes[1].colorbar()

    axes[2].imshow(mask.cpu().numpy(), cmap="gray", origin="lower")
    axes[2].set_title("Observation Mask")

    plt.tight_layout()
    plt.savefig("example_input_data.png", dpi=150)
    print("Saved input visualization to 'example_input_data.png'")
    plt.close()

    # Initialize statistics calculator
    print("\nInitializing WLStatistics...")
    stats = WLStatistics(n_scales=n_scales, device=device, pixel_arcmin=pixel_arcmin)

    # Print scale resolutions
    resolutions = stats.get_scale_resolutions()
    print("\nScale resolutions (FWHM):")
    for i, res in enumerate(resolutions):
        print(f"  Scale {i+1}: {res:.2f} arcmin")

    # Compute all statistics
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
        verbose=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\nWavelet Peak Counts:")
    for i, counts in enumerate(results["wavelet_peak_counts"]):
        n_peaks = int(counts.sum().item())
        max_snr = (
            results["wavelet_peak_heights"][i].max().item()
            if len(results["wavelet_peak_heights"][i]) > 0
            else 0
        )
        print(f"  Scale {i+1}: {n_peaks:4d} peaks (max SNR: {max_snr:6.2f})")

    print(f"\nMono-scale peaks: {int(results['mono_peak_counts'].sum().item())} peaks")

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Wavelet peak histograms
    plot_peak_histograms(
        results["peak_bins"],
        results["wavelet_peak_counts"],
        title="Wavelet Peak Counts (All Scales)",
        log_scale=True,
        save_path="example_peak_histograms.png",
    )

    # 2. L1-norms
    plot_l1_norms(
        results["l1_bins"],
        results["wavelet_l1_norms"],
        title="Wavelet L1-Norms (All Scales)",
        save_path="example_l1_norms.png",
    )

    # 3. Wavelet scales with peaks
    plot_wavelet_scales(
        results["snr"],
        peak_positions=results["wavelet_peak_positions"],
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
        mark_peaks=True,
        save_path="example_wavelet_scales.png",
    )

    # 4. SNR map for scale 1
    plot_snr_map(
        results["snr"],
        scale_idx=0,
        peak_positions=results["wavelet_peak_positions"][0],
        title="SNR Map - Scale 1",
        save_path="example_snr_scale1.png",
    )

    # 5. Mono-scale peaks
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["mono_peak_bins"].cpu().numpy(),
        results["mono_peak_counts"].cpu().numpy(),
        linewidth=2,
    )
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Peak Counts", fontsize=12)
    plt.title("Mono-Scale Peak Counts", fontsize=14)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("example_mono_peaks.png", dpi=150)
    print("Saved to 'example_mono_peaks.png'")
    plt.close()

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - example_input_data.png")
    print("  - example_peak_histograms.png")
    print("  - example_l1_norms.png")
    print("  - example_wavelet_scales.png")
    print("  - example_snr_scale1.png")
    print("  - example_mono_peaks.png")


if __name__ == "__main__":
    main()
