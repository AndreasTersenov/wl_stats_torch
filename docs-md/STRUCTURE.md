# Project Structure

```
wl_summary_stats_torch/
│
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── INSTALL.md               # Installation instructions
├── API.md                   # Complete API reference
├── LICENSE                  # MIT License
├── pyproject.toml          # Package configuration
├── .gitignore              # Git ignore file
├── verify_installation.py  # Installation verification script
│
├── wl_stats_torch/         # Main package
│   ├── __init__.py         # Package initialization
│   ├── starlet.py          # Starlet wavelet transform
│   ├── peaks.py            # Peak detection functions
│   ├── statistics.py       # Main WLStatistics class
│   └── visualization.py    # Plotting utilities
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_starlet.py     # Starlet transform tests
│   ├── test_peaks.py       # Peak detection tests
│   └── test_statistics.py  # Statistics pipeline tests
│
└── examples/                # Usage examples
    ├── basic_usage.py       # Simple example
    ├── cfis_example.py      # Realistic CFIS simulation
    └── batch_processing.py  # Batch processing demo
```

## Core Modules

### `starlet.py`
- **Starlet2D**: 2D à trous wavelet transform
- GPU-accelerated convolutions
- Noise level propagation
- SNR computation
- Perfect reconstruction

### `peaks.py`
- **find_peaks_2d**: Find local maxima in 2D
- **find_peaks_batch**: Batch peak detection
- **peaks_to_histogram**: Histogram computation
- **mono_scale_peaks_smoothed**: Smoothed peak analysis

### `statistics.py`
- **WLStatistics**: Main statistics calculator
- Wavelet peak counts at all scales
- Wavelet L1-norms
- Mono-scale peak counts
- Complete pipeline with masking support

### `visualization.py`
- **plot_peak_histograms**: Plot peak count distributions
- **plot_l1_norms**: Plot L1-norm statistics
- **plot_wavelet_scales**: Visualize wavelet decomposition
- **plot_snr_map**: Display SNR maps
- **plot_comparison**: Compare multiple results

## Examples

### `basic_usage.py`
Simple introduction with synthetic data:
- Generate convergence map with Gaussian peaks
- Add noise and create mask
- Compute all statistics
- Visualize results

### `cfis_example.py`
Realistic CFIS survey simulation:
- Power-law power spectrum
- CFIS noise parameters
- Survey footprint mask
- Full analysis pipeline

### `batch_processing.py`
Efficient processing of multiple maps:
- Batch generation
- GPU vs CPU benchmarking
- Statistical analysis across realizations
- Memory management tips

## Tests

### `test_starlet.py`
- Transform correctness
- Reconstruction accuracy
- Noise propagation
- Device handling
- Edge cases

### `test_peaks.py`
- Peak detection accuracy
- Threshold filtering
- Masking behavior
- Histogram computation
- Batch processing

### `test_statistics.py`
- Pipeline integration
- Result completeness
- Masking support
- Device compatibility
- Error handling

## Documentation

### README.md
- Overview and features
- Installation quickstart
- Basic usage example
- Citation information

### QUICKSTART.md
- Minimal examples
- Common use cases
- Performance tips
- Visualization guide

### INSTALL.md
- Detailed installation
- Requirements
- GPU setup
- Troubleshooting

### API.md
- Complete API reference
- All classes and functions
- Parameters and returns
- Usage examples

## Development

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=wl_stats_torch
```

Format code:
```bash
black wl_stats_torch/
isort wl_stats_torch/
```

Verify installation:
```bash
python verify_installation.py
```

## Package Configuration

The `pyproject.toml` defines:
- Package metadata
- Dependencies
- Development dependencies
- Build system
- Tool configurations (black, isort, pytest)

## Git Configuration

The `.gitignore` excludes:
- Python cache files
- Build artifacts
- Test outputs
- Generated images
- IDE files
- Environment files
