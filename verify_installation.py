"""
Verification script to test installation and basic functionality.

Run this after installation to verify everything works correctly.
"""

import sys
import torch


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 70)
    print("PyTorch Installation")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    print("✓ PyTorch OK\n")


def check_package_import():
    """Check if package can be imported."""
    print("=" * 70)
    print("Package Import")
    print("=" * 70)
    
    try:
        import wl_stats_torch
        print(f"wl_stats_torch version: {wl_stats_torch.__version__}")
        print("✓ Package import OK\n")
        return True
    except ImportError as e:
        print(f"✗ Failed to import wl_stats_torch: {e}")
        print("\nPlease install the package:")
        print("  pip install -e .")
        return False


def test_starlet():
    """Test Starlet transform."""
    print("=" * 70)
    print("Testing Starlet Transform")
    print("=" * 70)
    
    try:
        from wl_stats_torch.starlet import Starlet2D
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create test image
        image = torch.randn(128, 128, device=device)
        
        # Initialize and transform
        starlet = Starlet2D(n_scales=5, device=device)
        coeffs = starlet(image)
        
        print(f"  Input shape: {image.shape}")
        print(f"  Output shape: {coeffs.shape}")
        
        # Test reconstruction
        reconstructed = starlet.reconstruct(coeffs)
        error = torch.abs(image - reconstructed.squeeze()).max().item()
        print(f"  Reconstruction error: {error:.2e}")
        
        if error < 1e-5:
            print("✓ Starlet transform OK\n")
            return True
        else:
            print(f"✗ Reconstruction error too large: {error}\n")
            return False
            
    except Exception as e:
        print(f"✗ Starlet test failed: {e}\n")
        return False


def test_peaks():
    """Test peak detection."""
    print("=" * 70)
    print("Testing Peak Detection")
    print("=" * 70)
    
    try:
        from wl_stats_torch.peaks import find_peaks_2d
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create image with known peaks
        image = torch.zeros(64, 64, device=device)
        image[10, 10] = 5.0
        image[30, 30] = 3.0
        image[50, 50] = 4.0
        
        # Find peaks
        positions, heights = find_peaks_2d(image, threshold=2.0)
        
        print(f"  Found {len(positions)} peaks")
        print(f"  Expected: 3 peaks")
        
        if len(positions) == 3:
            print("✓ Peak detection OK\n")
            return True
        else:
            print(f"✗ Expected 3 peaks, found {len(positions)}\n")
            return False
            
    except Exception as e:
        print(f"✗ Peak detection test failed: {e}\n")
        return False


def test_statistics():
    """Test full statistics pipeline."""
    print("=" * 70)
    print("Testing WLStatistics Pipeline")
    print("=" * 70)
    
    try:
        from wl_stats_torch import WLStatistics
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create test data
        kappa = torch.randn(128, 128, device=device) * 0.01
        sigma = torch.ones(128, 128, device=device) * 0.02
        
        # Initialize statistics
        stats = WLStatistics(n_scales=5, device=device)
        
        # Compute all statistics
        results = stats.compute_all_statistics(
            kappa,
            sigma,
            verbose=False
        )
        
        # Check results
        required_keys = [
            'wavelet_coeffs',
            'wavelet_peak_counts',
            'wavelet_l1_norms',
            'mono_peak_counts'
        ]
        
        all_present = all(key in results for key in required_keys)
        
        if all_present:
            print("  All required result keys present")
            print(f"  Number of scales: {len(results['wavelet_peak_counts'])}")
            print(f"  Peak bins: {len(results['peak_bins'])}")
            print("✓ Statistics pipeline OK\n")
            return True
        else:
            missing = [key for key in required_keys if key not in results]
            print(f"✗ Missing result keys: {missing}\n")
            return False
            
    except Exception as e:
        print(f"✗ Statistics test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_checks():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print(" WL-STATS-TORCH INSTALLATION VERIFICATION")
    print("=" * 70 + "\n")
    
    # Check PyTorch
    check_pytorch()
    
    # Check package import
    if not check_package_import():
        print("\n" + "=" * 70)
        print("FAILED: Package not installed correctly")
        print("=" * 70)
        return False
    
    # Run tests
    results = {
        'Starlet': test_starlet(),
        'Peak Detection': test_peaks(),
        'Statistics Pipeline': test_statistics()
    }
    
    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("=" * 70)
    
    if all(results.values()):
        print("\n🎉 All checks passed! Installation successful.")
        print("\nNext steps:")
        print("  1. Try examples: cd examples && python basic_usage.py")
        print("  2. Run tests: pytest tests/ -v")
        print("  3. Read documentation: cat QUICKSTART.md")
        return True
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  1. Reinstall: pip install -e .")
        print("  2. Check dependencies: pip list | grep torch")
        print("  3. Try on CPU: device = torch.device('cpu')")
        return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
