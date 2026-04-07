# Copilot Instructions for `wl_stats_torch`

## Build, test, and lint commands

- Install project + dev tools:
  - `pip install -e ".[dev]"`
  - or `make install-dev`
- Run lint checks:
  - `make lint`
  - equivalent: `flake8 wl_stats_torch tests`
- Run formatting checks (without modifying files):
  - `make format-check`
  - equivalent: `isort --check-only wl_stats_torch tests examples && black --check wl_stats_torch tests examples`
- Run full tests:
  - `make test`
  - equivalent: `pytest`
- Run tests with coverage:
  - `make test-cov`
- Run a single test file:
  - `pytest tests/test_peaks.py`
- Run a single test:
  - `pytest tests/test_peaks.py::TestPeakDetection::test_find_peaks_simple`
- Build docs:
  - `make docs`
- Build distributions:
  - `make build`

## High-level architecture

- `wl_stats_torch/statistics.py` is the orchestration layer.
  - `WLStatistics.compute_wavelet_transform()` computes Starlet coefficients, propagated noise levels, and SNR maps.
  - `compute_wavelet_peak_counts()` and `compute_wavelet_l1_norms()` operate on cached `self.snr_coeffs` from the transform step.
  - `compute_mono_scale_peaks()` provides Gaussian-smoothed mono-scale peak counts.
  - `compute_all_statistics()` wires all steps into one pipeline and returns a unified results dictionary.

- `wl_stats_torch/starlet.py` implements core wavelet math.
  - `Starlet2D` uses fixed B3-spline kernels with dilated `Conv2d` layers (`a trous`).
  - Noise propagation uses impulse-response convolution (`fft_convolve2d`) in PyTorch to stay on-device.
  - Reconstruction and scale-resolution helpers live here as well.

- `wl_stats_torch/peaks.py` handles peak and histogram logic.
  - `find_peaks_2d` / `find_peaks_batch` perform local-max detection.
  - `peaks_to_histogram` implements binning semantics aligned with CosmoStat/pycs.
  - `mono_scale_peaks_smoothed` is the mono-scale statistics path.

- `wl_stats_torch/visualization.py` is plotting-only and intentionally converts tensors to NumPy (`.cpu().numpy()`) for matplotlib.

- Test layout mirrors architecture:
  - `tests/test_starlet.py`, `tests/test_peaks.py`, `tests/test_statistics.py`
  - `tests/test_batch_processing.py` validates batch vectorization, broadcasting, and sequential-vs-batch consistency.

## Key repository conventions

- Shape contract is strict and preserved across APIs:
  - single image input: `(H, W)`
  - batched input: `(B, H, W)`
  - outputs mirror this choice (single outputs are unbatched; batched outputs include `B`).

- Noise and mask broadcasting behavior is centralized in `WLStatistics._broadcast_noise_sigma()` and `_broadcast_mask()`. Reuse/extend these helpers rather than adding ad-hoc shape handling.

- Histogram behavior intentionally matches CosmoStat/pycs:
  - default `clamp_overflow=False` excludes out-of-range values
  - right-edge bin handling is explicitly implemented in `peaks_to_histogram()`.

- Peak definition is strict local maximum vs all 8 neighbors (`>`), with border pixels excluded by default (`include_border=False`).

- `compute_wavelet_peak_counts()` and `compute_wavelet_l1_norms()` require `compute_wavelet_transform()` first (they consume cached class state). Keep this call ordering invariant unless the API is redesigned end-to-end.

- Numerical/device convention:
  - `WLStatistics` defaults to `torch.float64` (NumPy/CosmoStat-aligned precision).
  - `Starlet2D` receives dtype/device from `WLStatistics`.
  - Compute paths should stay in PyTorch tensors on `self.device`; avoid unnecessary CPU transfers.

- Coarse-scale handling is deliberate: `subtract_coarse_mean=True` by default in `compute_wavelet_transform()` and `compute_all_statistics()`.

- CI and local tooling expectations:
  - Python compatibility target in CI: 3.8 through 3.12.
  - Formatting/linting: Black + isort + flake8, line length 100.
  - flake8 ignores include `E203`, `E501`, `W503`; `__init__.py` allows `F401` for API re-exports.
Nice .