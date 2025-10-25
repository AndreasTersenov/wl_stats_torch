"""
Batch processing example - efficiently process multiple convergence maps.

This demonstrates how to process multiple maps efficiently using GPU acceleration.
"""

import time

import matplotlib.pyplot as plt
import torch

from wl_stats_torch import WLStatistics
from wl_stats_torch.visualization import plot_comparison


def create_synthetic_maps(n_maps, img_size, device):
    """Create batch of synthetic convergence maps."""
    maps = []
    for i in range(n_maps):
        # Create map with varying structure
        kappa = torch.randn(img_size, img_size, device=device) * 0.01

        # Add some Gaussian peaks
        n_peaks = torch.randint(5, 15, (1,)).item()
        for _ in range(n_peaks):
            x = torch.randint(20, img_size - 20, (1,)).item()
            y = torch.randint(20, img_size - 20, (1,)).item()
            amplitude = torch.randn(1).item() * 0.03
            sigma = torch.randint(5, 15, (1,)).item()

            xx, yy = torch.meshgrid(
                torch.arange(img_size, device=device),
                torch.arange(img_size, device=device),
                indexing="ij",
            )
            gaussian = amplitude * torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
            kappa += gaussian

        maps.append(kappa)

    return torch.stack(maps)


def main():
    print("=" * 70)
    print("Batch Processing Example")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Parameters
    n_maps = 10
    img_size = 256
    n_scales = 5
    noise_level = 0.02

    print(f"\nParameters:")
    print(f"  Number of maps: {n_maps}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of scales: {n_scales}")
    print(f"  Noise level: {noise_level}")

    # Generate synthetic maps
    print(f"\nGenerating {n_maps} synthetic convergence maps...")
    start_time = time.time()
    kappa_maps = create_synthetic_maps(n_maps, img_size, device)
    gen_time = time.time() - start_time
    print(f"Generation time: {gen_time:.2f} seconds")

    # Add noise
    noise_maps = torch.randn_like(kappa_maps) * noise_level
    kappa_noisy = kappa_maps + noise_maps

    # Initialize statistics calculator
    stats = WLStatistics(n_scales=n_scales, device=device, pixel_arcmin=0.5)

    # Process all maps
    print(f"\nProcessing {n_maps} maps...")
    start_time = time.time()

    all_results = []
    for i in range(n_maps):
        if i % 5 == 0:
            print(f"  Processing map {i+1}/{n_maps}...")

        results = stats.compute_all_statistics(
            kappa_noisy[i], noise_level, compute_mono=False, verbose=False  # Skip for speed
        )
        all_results.append(results)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per map: {total_time/n_maps:.2f} seconds")

    # Benchmark comparison: CPU vs GPU
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("CPU vs GPU Benchmark")
        print("=" * 70)

        # Test single map on GPU
        test_map = kappa_noisy[0]
        stats_gpu = WLStatistics(n_scales=n_scales, device=torch.device("cuda"))

        torch.cuda.synchronize()
        start = time.time()
        _ = stats_gpu.compute_all_statistics(test_map, noise_level, verbose=False)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        # Test same map on CPU
        stats_cpu = WLStatistics(n_scales=n_scales, device=torch.device("cpu"))
        test_map_cpu = test_map.cpu()

        start = time.time()
        _ = stats_cpu.compute_all_statistics(test_map_cpu, noise_level, verbose=False)
        cpu_time = time.time() - start

        print(f"\nSingle map processing time:")
        print(f"  GPU: {gpu_time:.3f} seconds")
        print(f"  CPU: {cpu_time:.3f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

    # Analyze results across maps
    print("\n" + "=" * 70)
    print("Analysis Across Maps")
    print("=" * 70)

    # Compute mean and std of peak counts across maps
    all_peak_counts = []
    for scale_idx in range(n_scales):
        scale_counts = []
        for results in all_results:
            counts = results["wavelet_peak_counts"][scale_idx]
            scale_counts.append(counts)

        # Stack and compute statistics
        scale_counts_tensor = torch.stack(scale_counts)
        mean_counts = scale_counts_tensor.mean(dim=0)
        std_counts = scale_counts_tensor.std(dim=0)

        all_peak_counts.append((mean_counts, std_counts))

    # Plot mean peak counts with error bars
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    bin_centers = all_results[0]["peak_bins"].cpu().numpy()

    for scale_idx in range(n_scales):
        ax = axes[scale_idx]
        mean_counts, std_counts = all_peak_counts[scale_idx]

        mean_np = mean_counts.cpu().numpy()
        std_np = std_counts.cpu().numpy()

        ax.plot(bin_centers, mean_np, "b-", linewidth=2, label="Mean")
        ax.fill_between(bin_centers, mean_np - std_np, mean_np + std_np, alpha=0.3, label="±1 std")

        ax.set_xlabel("SNR", fontsize=10)
        ax.set_ylabel("Peak Counts", fontsize=10)
        ax.set_title(f"Scale {scale_idx + 1}", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide last subplot
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig("batch_mean_peak_counts.png", dpi=150)
    print("Saved to 'batch_mean_peak_counts.png'")
    plt.close()

    # Compare first 3 maps directly
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for map_idx in range(min(3, n_maps)):
        ax = axes[map_idx]

        for scale_idx in range(n_scales):
            counts = all_results[map_idx]["wavelet_peak_counts"][scale_idx].cpu().numpy()
            ax.plot(bin_centers, counts, label=f"Scale {scale_idx+1}", alpha=0.7)

        ax.set_xlabel("SNR", fontsize=10)
        ax.set_ylabel("Peak Counts", fontsize=10)
        ax.set_title(f"Map {map_idx + 1}", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if map_idx == 2:
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig("batch_individual_maps.png", dpi=150)
    print("Saved to 'batch_individual_maps.png'")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print("\nTotal peaks detected per scale (summed across all maps):")
    for scale_idx in range(n_scales):
        total_peaks = sum(
            results["wavelet_peak_counts"][scale_idx].sum().item() for results in all_results
        )
        avg_peaks = total_peaks / n_maps
        print(
            f"  Scale {scale_idx+1}: {int(total_peaks):6d} total, "
            f"{avg_peaks:6.1f} average per map"
        )

    print("\n" + "=" * 70)
    print("✓ Batch processing example completed!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - batch_mean_peak_counts.png")
    print("  - batch_individual_maps.png")

    print("\nTips for large-scale processing:")
    print("  1. Process maps in batches to manage memory")
    print("  2. Use GPU for significant speedup")
    print("  3. Set compute_mono=False to save time if not needed")
    print("  4. Consider using torch.cuda.empty_cache() between batches")


if __name__ == "__main__":
    main()
