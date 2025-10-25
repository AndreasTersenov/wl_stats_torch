"""
Visualization utilities for weak lensing statistics.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_peak_histograms(
    bin_centers: torch.Tensor,
    peak_counts: List[torch.Tensor],
    scale_labels: Optional[List[str]] = None,
    title: str = "Wavelet Peak Counts",
    xlabel: str = "SNR",
    ylabel: str = "Peak Counts",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot peak count histograms for multiple scales.

    Args:
        bin_centers: Bin centers, shape (n_bins,)
        peak_counts: List of peak counts per scale
        scale_labels: Optional labels for each scale
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic y-axis
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=figsize)

    # Convert to numpy for plotting
    bins_np = bin_centers.cpu().numpy()

    n_scales = len(peak_counts)
    if scale_labels is None:
        scale_labels = [f"Scale {i+1}" for i in range(n_scales)]

    for i, counts in enumerate(peak_counts):
        counts_np = counts.cpu().numpy()
        plt.plot(bins_np, counts_np, label=scale_labels[i], linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_l1_norms(
    l1_bins: List[torch.Tensor],
    l1_norms: List[torch.Tensor],
    scale_labels: Optional[List[str]] = None,
    title: str = "Wavelet L1-Norms",
    xlabel: str = "SNR",
    ylabel: str = "L1-Norm",
    log_scale: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot L1-norm as a function of SNR for multiple scales.

    Args:
        l1_bins: List of bin centers per scale
        l1_norms: List of L1-norms per scale
        scale_labels: Optional labels for each scale
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic y-axis
        xlim: X-axis limits (min, max)
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=figsize)

    n_scales = len(l1_norms)
    if scale_labels is None:
        scale_labels = [f"Scale {i+1}" for i in range(n_scales)]

    for i in range(n_scales):
        bins_np = l1_bins[i].cpu().numpy()
        norms_np = l1_norms[i].cpu().numpy()
        plt.plot(bins_np, norms_np, label=scale_labels[i], linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    if xlim:
        plt.xlim(xlim)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_wavelet_scales(
    wavelet_coeffs: torch.Tensor,
    peak_positions: Optional[List[torch.Tensor]] = None,
    titles: Optional[List[str]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (15, 10),
    mark_peaks: bool = True,
    save_path: Optional[str] = None,
):
    """
    Visualize wavelet scales with optional peak markers.

    Args:
        wavelet_coeffs: Wavelet coefficients (n_scales, H, W)
        peak_positions: Optional list of peak positions per scale
        titles: Optional titles for each scale
        cmap: Colormap name
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
        figsize: Figure size
        mark_peaks: Whether to mark peak positions
        save_path: If provided, save figure to this path
    """
    n_scales = wavelet_coeffs.shape[0]

    # Determine grid layout
    n_cols = min(3, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_scales == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_scales):
        ax = axes[i]
        scale_data = wavelet_coeffs[i].cpu().numpy()

        # Plot scale
        im = ax.imshow(scale_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

        # Mark peaks if provided
        if mark_peaks and peak_positions is not None and i < len(peak_positions):
            if len(peak_positions[i]) > 0:
                peaks = peak_positions[i].cpu().numpy()
                ax.scatter(peaks[:, 1], peaks[:, 0], c="red", s=10, alpha=0.5)

        # Set title
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Scale {i+1}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(n_scales, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_snr_map(
    snr_coeffs: torch.Tensor,
    scale_idx: int = 0,
    peak_positions: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: float = -5,
    vmax: float = 5,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot SNR map for a specific scale with optional peak markers.

    Args:
        snr_coeffs: SNR coefficients (n_scales, H, W)
        scale_idx: Which scale to plot
        peak_positions: Optional peak positions (N, 2)
        title: Plot title
        cmap: Colormap name
        vmin: Minimum SNR for colorscale
        vmax: Maximum SNR for colorscale
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=figsize)

    snr_data = snr_coeffs[scale_idx].cpu().numpy()

    plt.imshow(snr_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar(label="SNR", fraction=0.046, pad=0.04)

    # Mark peaks
    if peak_positions is not None and len(peak_positions) > 0:
        peaks = peak_positions.cpu().numpy()
        plt.scatter(
            peaks[:, 1],
            peaks[:, 0],
            c="black",
            s=20,
            marker="x",
            alpha=0.7,
            label=f"{len(peaks)} peaks",
        )
        plt.legend()

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f"SNR Map - Scale {scale_idx + 1}", fontsize=14)

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_comparison(
    results_list: List[dict],
    labels: List[str],
    statistic: str = "wavelet_peak_counts",
    scale_idx: int = 0,
    title: Optional[str] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Compare the same statistic across multiple result sets.

    Args:
        results_list: List of result dictionaries from compute_all_statistics
        labels: Labels for each result set
        statistic: Which statistic to compare ('wavelet_peak_counts' or 'wavelet_l1_norms')
        scale_idx: Which scale to plot (for multi-scale statistics)
        title: Plot title
        log_scale: Use logarithmic y-axis
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=figsize)

    for results, label in zip(results_list, labels):
        if statistic == "wavelet_peak_counts":
            bins = results["peak_bins"].cpu().numpy()
            data = results["wavelet_peak_counts"][scale_idx].cpu().numpy()
            ylabel = "Peak Counts"
        elif statistic == "wavelet_l1_norms":
            bins = results["l1_bins"][scale_idx].cpu().numpy()
            data = results["wavelet_l1_norms"][scale_idx].cpu().numpy()
            ylabel = "L1-Norm"
        elif statistic == "mono_peak_counts":
            bins = results["mono_peak_bins"].cpu().numpy()
            data = results["mono_peak_counts"].cpu().numpy()
            ylabel = "Peak Counts"
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        plt.plot(bins, data, label=label, linewidth=2)

    plt.xlabel("SNR", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f"Comparison: {statistic} (Scale {scale_idx + 1})", fontsize=14)

    plt.legend()
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
