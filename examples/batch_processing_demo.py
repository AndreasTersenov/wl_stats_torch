"""
Batch Processing Demo for wl_stats_torch

This example demonstrates the new batch processing capabilities of WLStatistics,
showing how to efficiently process multiple convergence maps simultaneously for
GPU-accelerated training workflows.
"""

import time
import torch
from wl_stats_torch import WLStatistics


def demo_basic_batch_usage():
    """Basic example: process a batch of convergence maps."""
    print("="*80)
    print("DEMO 1: Basic Batch Processing")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create batch of convergence maps
    batch_size = 32
    height, width = 256, 128
    
    print(f"Creating batch of {batch_size} convergence maps ({height}×{width})...")
    images = torch.randn(batch_size, height, width, device=device)
    noise_sigma = 0.001
    
    # Initialize statistics calculator
    stats = WLStatistics(n_scales=6, device=device)
    
    # Compute statistics for entire batch at once
    print("Computing statistics for batch...")
    results = stats.compute_all_statistics(
        images,
        noise_sigma,
        min_snr=-4.0,
        max_snr=15.0,
        n_bins=51,
        l1_nbins=100,
        compute_mono=False,
        verbose=True
    )
    
    # Inspect results
    print("\nResults:")
    print(f"  Wavelet coefficients: {results['wavelet_coeffs'].shape}")
    print(f"  Peak counts per scale: {[pc.shape for pc in results['wavelet_peak_counts']]}")
    print(f"  L1-norms per scale: {[l1.shape for l1 in results['wavelet_l1_norms']]}")
    
    print("\n✓ Basic batch processing complete!\n")


def demo_feature_extraction():
    """Example: Extract feature vectors for ML training."""
    print("="*80)
    print("DEMO 2: Feature Extraction for ML Training")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training batch
    batch_size = 128
    height, width = 256, 128
    n_scales = 6
    
    print(f"\nProcessing training batch: {batch_size} samples")
    images = torch.randn(batch_size, height, width, device=device)
    noise_sigma = 0.001
    mask = torch.ones(height, width, device=device)
    
    # Compute statistics
    stats = WLStatistics(n_scales=n_scales, device=device)
    results = stats.compute_all_statistics(
        images,
        noise_sigma,
        mask=mask,
        min_snr=-4.0,
        max_snr=15.0,
        n_bins=51,
        l1_nbins=100,
        compute_mono=False,
        verbose=False
    )
    
    # Extract feature vectors
    print("Extracting features...")
    wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (n_scales, B, n_bins)
    wavelet_l1 = torch.stack(results['wavelet_l1_norms'])  # (n_scales, B, l1_nbins)
    
    # Reshape to (B, features)
    features = torch.cat([
        wavelet_peaks.permute(1, 0, 2).flatten(1),  # (B, n_scales*n_bins)
        wavelet_l1.permute(1, 0, 2).flatten(1)      # (B, n_scales*l1_nbins)
    ], dim=1)
    
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dimensions: {features.shape[1]} per sample")
    print(f"    - Peak counts: {n_scales} scales × 51 bins = {n_scales * 51}")
    print(f"    - L1-norms: {n_scales} scales × 100 bins = {n_scales * 100}")
    
    # These features can now be used directly for training
    print("\n✓ Features ready for ML training!\n")


def demo_performance_comparison():
    """Compare batch vs sequential processing performance."""
    print("="*80)
    print("DEMO 3: Performance Comparison (Batch vs Sequential)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("\nRunning on CPU - speedup will be limited")
    else:
        print(f"\nRunning on GPU: {torch.cuda.get_device_name(device)}")
    
    # Test parameters
    batch_size = 64
    height, width = 256, 128
    
    print(f"\nTest: {batch_size} convergence maps ({height}×{width})")
    
    # Create data
    images = torch.randn(batch_size, height, width, device=device)
    noise_sigma = 0.001
    
    stats = WLStatistics(n_scales=6, device=device)
    
    # Method 1: Batch processing
    print("\n1. Batch processing...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    results_batch = stats.compute_all_statistics(
        images,
        noise_sigma,
        compute_mono=False,
        verbose=False
    )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_batch = time.time() - start
    print(f"   Time: {time_batch:.3f} seconds")
    
    # Method 2: Sequential processing
    print("\n2. Sequential processing...")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    for i in range(batch_size):
        results_seq = stats.compute_all_statistics(
            images[i],
            noise_sigma,
            compute_mono=False,
            verbose=False
        )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    time_sequential = time.time() - start
    print(f"   Time: {time_sequential:.3f} seconds")
    
    # Compare
    speedup = time_sequential / time_batch
    print(f"\n📊 Speedup: {speedup:.2f}x faster with batch processing")
    print(f"   Per-image time (batch): {time_batch/batch_size*1000:.2f} ms")
    print(f"   Per-image time (sequential): {time_sequential/batch_size*1000:.2f} ms")
    
    print("\n✓ Performance comparison complete!\n")


def demo_mixed_inputs():
    """Demonstrate different noise and mask configurations."""
    print("="*80)
    print("DEMO 4: Mixed Noise and Mask Configurations")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 8
    height, width = 128, 64
    images = torch.randn(batch_size, height, width, device=device)
    
    stats = WLStatistics(n_scales=5, device=device)
    
    # Test 1: Scalar noise
    print("\n1. Scalar noise (same for all images)...")
    results = stats.compute_all_statistics(
        images, noise_sigma=0.001, compute_mono=False, verbose=False
    )
    print(f"   ✓ Results shape: {results['wavelet_coeffs'].shape}")
    
    # Test 2: Shared noise map
    print("\n2. Shared noise map (H, W) - same spatial pattern...")
    noise_map = torch.rand(height, width, device=device) * 0.002 + 0.001
    results = stats.compute_all_statistics(
        images, noise_sigma=noise_map, compute_mono=False, verbose=False
    )
    print(f"   ✓ Results shape: {results['wavelet_coeffs'].shape}")
    
    # Test 3: Per-sample noise maps
    print("\n3. Per-sample noise maps (B, H, W) - different for each...")
    noise_batch = torch.rand(batch_size, height, width, device=device) * 0.002 + 0.001
    results = stats.compute_all_statistics(
        images, noise_sigma=noise_batch, compute_mono=False, verbose=False
    )
    print(f"   ✓ Results shape: {results['wavelet_coeffs'].shape}")
    
    # Test 4: Shared mask
    print("\n4. Shared mask (H, W) - same for all images...")
    mask = torch.ones(height, width, device=device)
    mask[:10, :] = 0  # Mask out top region
    results = stats.compute_all_statistics(
        images, noise_sigma=0.001, mask=mask, compute_mono=False, verbose=False
    )
    print(f"   ✓ Results shape: {results['wavelet_coeffs'].shape}")
    
    # Test 5: Per-sample masks
    print("\n5. Per-sample masks (B, H, W) - different for each...")
    mask_batch = torch.ones(batch_size, height, width, device=device)
    for i in range(batch_size):
        mask_batch[i, :10+i*2, :] = 0  # Different mask per image
    results = stats.compute_all_statistics(
        images, noise_sigma=0.001, mask=mask_batch, compute_mono=False, verbose=False
    )
    print(f"   ✓ Results shape: {results['wavelet_coeffs'].shape}")
    
    print("\n✓ All input configurations work correctly!\n")


def demo_backward_compatibility():
    """Show that single image processing still works as before."""
    print("="*80)
    print("DEMO 5: Backward Compatibility (Single Images)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Single image (original API)
    print("\nProcessing single image (H, W)...")
    image = torch.randn(256, 128, device=device)
    noise_sigma = 0.001
    
    stats = WLStatistics(n_scales=6, device=device)
    results = stats.compute_all_statistics(
        image,
        noise_sigma,
        compute_mono=True,
        verbose=False
    )
    
    print(f"  Wavelet coefficients: {results['wavelet_coeffs'].shape}")
    print(f"  Peak counts (scale 0): {results['wavelet_peak_counts'][0].shape}")
    print(f"  L1-norms (scale 0): {results['wavelet_l1_norms'][0].shape}")
    print(f"  Mono-scale peaks: {results['mono_peak_counts'].shape}")
    
    print("\n✓ Single image API remains unchanged!\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("wl_stats_torch: Batch Processing Demonstrations")
    print("="*80 + "\n")
    
    # Run all demos
    demo_basic_batch_usage()
    demo_feature_extraction()
    demo_performance_comparison()
    demo_mixed_inputs()
    demo_backward_compatibility()
    
    print("="*80)
    print("All demonstrations complete!")
    print("="*80)
