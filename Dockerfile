# Dockerfile for wl-stats-torch (CPU variant)
# Multi-stage build for minimal image size

# ============================================
# Build stage
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY wl_stats_torch/ ./wl_stats_torch/

# Install pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Build wheel
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels .

# Install CPU-only PyTorch (smaller than default)
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels \
    torch --index-url https://download.pytorch.org/whl/cpu

# ============================================
# Runtime stage
# ============================================
FROM python:3.11-slim

LABEL maintainer="Andreas Tersenov <andrewtersenov@gmail.com>"
LABEL description="GPU-accelerated weak lensing summary statistics with PyTorch (CPU variant)"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels

# Install CPU-only PyTorch first, then the package
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /wheels/torch*.whl \
    && pip install --no-cache-dir /wheels/wl_stats_torch*.whl \
    && rm -rf /wheels

# Create directory for user data
RUN mkdir -p /data

# Set working directory for data operations
WORKDIR /data

# Default command: Python interpreter
CMD ["python"]
