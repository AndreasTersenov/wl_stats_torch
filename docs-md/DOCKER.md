# Docker Usage Guide

This guide covers how to use `wl-stats-torch` with Docker for both CPU and GPU environments.

## Quick Start

### CPU Version

```bash
# Build the image
docker build -t wl-stats-torch:cpu .

# Run interactive Python shell
docker run -it --rm -v $(pwd)/data:/data wl-stats-torch:cpu

# Run a script
docker run -it --rm -v $(pwd)/data:/data wl-stats-torch:cpu python /data/my_script.py
```

### GPU Version (CUDA)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Build the image
docker build -t wl-stats-torch:cuda -f Dockerfile.cuda .

# Run interactive Python shell with GPU access
docker run -it --rm --gpus all -v $(pwd)/data:/data wl-stats-torch:cuda

# Run a script with GPU
docker run -it --rm --gpus all -v $(pwd)/data:/data wl-stats-torch:cuda python /data/my_script.py
```

## Building Images

### CPU Image

```bash
docker build -t wl-stats-torch:cpu .
```

The CPU image uses a multi-stage build to minimize size:
- Build stage: compiles the package
- Runtime stage: contains only production dependencies

### GPU Image

```bash
docker build -t wl-stats-torch:cuda -f Dockerfile.cuda .
```

The GPU image is based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04` and includes:
- CUDA 12.4 runtime
- Python 3.11
- PyTorch with CUDA support

## Using Docker Compose

Docker Compose provides an easy way to manage both CPU and GPU services.

### Start CPU Service

```bash
# Interactive shell
docker compose run --rm wl-stats-cpu

# Run specific command
docker compose run --rm wl-stats-cpu python -c "import wl_stats_torch; print('OK')"
```

### Start GPU Service

```bash
# Interactive shell with GPU
docker compose run --rm wl-stats-gpu

# Run specific command
docker compose run --rm wl-stats-gpu python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Run Tests

```bash
# Run tests in CPU container
docker compose run --rm test-cpu

# Run tests in GPU container
docker compose run --rm test-gpu
```

### Build All Images

```bash
docker compose build
```

## Volume Mounting

### Data Directory

Mount your data directory to `/data` inside the container:

```bash
docker run -it --rm -v /path/to/your/data:/data wl-stats-torch:cpu
```

### Scripts and Examples

Mount scripts for execution:

```bash
docker run -it --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/scripts:/scripts:ro \
  wl-stats-torch:cpu python /scripts/analyze.py
```

### Multiple Mounts

```bash
docker run -it --rm \
  -v $(pwd)/input:/data/input:ro \
  -v $(pwd)/output:/data/output \
  wl-stats-torch:cpu python /data/input/process.py
```

## Example Usage

### Basic Analysis

Create a script `analyze.py`:

```python
import torch
from wl_stats_torch import WLStatistics

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize statistics calculator
stats = WLStatistics(n_scales=5, device=device)

# Load your data (mounted at /data)
kappa_map = torch.load('/data/kappa_map.pt')
sigma_map = torch.load('/data/sigma_map.pt')

# Compute statistics
results = stats.compute_all_statistics(
    kappa_map.to(device),
    sigma_map.to(device),
    min_snr=-2,
    max_snr=6,
    nbins=31
)

# Save results
torch.save(results, '/data/results.pt')
print("Results saved to /data/results.pt")
```

Run with Docker:

```bash
# CPU
docker run -it --rm -v $(pwd):/data wl-stats-torch:cpu python /data/analyze.py

# GPU
docker run -it --rm --gpus all -v $(pwd):/data wl-stats-torch:cuda python /data/analyze.py
```

### Batch Processing

```python
import torch
from wl_stats_torch import WLStatistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stats = WLStatistics(n_scales=5, device=device)

# Process multiple maps
for i in range(100):
    kappa = torch.load(f'/data/maps/kappa_{i:04d}.pt')
    sigma = torch.load(f'/data/maps/sigma_{i:04d}.pt')

    results = stats.compute_all_statistics(
        kappa.to(device),
        sigma.to(device)
    )

    torch.save(results, f'/data/output/results_{i:04d}.pt')
```

## GPU Configuration

### Specify GPU Device

```bash
# Use specific GPU
docker run -it --rm --gpus '"device=0"' -v $(pwd)/data:/data wl-stats-torch:cuda

# Use multiple GPUs
docker run -it --rm --gpus '"device=0,1"' -v $(pwd)/data:/data wl-stats-torch:cuda
```

### Memory Limits

```bash
# Limit container memory
docker run -it --rm --gpus all --memory=16g -v $(pwd)/data:/data wl-stats-torch:cuda
```

### Check GPU Availability

```bash
docker run --rm --gpus all wl-stats-torch:cuda python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

## Troubleshooting

### GPU Not Detected

1. Ensure NVIDIA drivers are installed on the host
2. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. Test with: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run as current user
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd)/data:/data wl-stats-torch:cpu
```

### Out of Memory

For large datasets, increase Docker memory limits or process data in chunks.

## Image Sizes

Approximate sizes:
- CPU image: ~1.5 GB
- GPU image: ~8 GB (includes CUDA runtime)

To reduce image size:
- CPU image uses multi-stage build
- GPU image uses runtime (not devel) CUDA base
