"""
Visualization utilities for NeuronBrain.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_raster(
    spike_times: List[float],
    neuron_ids: List[int],
    filename: Optional[str] = None,
    title: str = "Spike Raster Plot",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    marker_size: float = 0.5
) -> plt.Figure:
    """
    Create a spike raster plot.
    
    Args:
        spike_times: List of spike times
        neuron_ids: Corresponding neuron IDs
        filename: Optional file to save plot
        title: Plot title
        xlim: X-axis limits (time)
        ylim: Y-axis limits (neuron IDs)
        marker_size: Size of spike markers
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(spike_times, neuron_ids, s=marker_size, c='black', alpha=0.7)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_firing_rates(
    rates: np.ndarray,
    neuron_ids: Optional[List[int]] = None,
    filename: Optional[str] = None,
    title: str = "Firing Rates",
    bins: int = 50
) -> plt.Figure:
    """
    Plot distribution of firing rates.
    
    Args:
        rates: Array of firing rates
        neuron_ids: Optional neuron IDs for labeling
        filename: Optional file to save plot
        title: Plot title
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(rates, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Firing Rate (Hz)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Firing Rate Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Sorted rates
    sorted_rates = np.sort(rates)[::-1]
    if neuron_ids:
        sorted_ids = np.argsort(rates)[::-1]
        axes[1].scatter(sorted_ids, sorted_rates, s=5, c='darkred', alpha=0.7)
        axes[1].set_xlabel('Neuron Rank', fontsize=12)
    else:
        axes[1].plot(sorted_rates, color='darkred', linewidth=1)
        axes[1].set_xlabel('Neuron Rank', fontsize=12)
    
    axes[1].set_ylabel('Firing Rate (Hz)', fontsize=12)
    axes[1].set_title('Sorted Firing Rates', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_connectivity(
    connectivity_matrix: np.ndarray,
    filename: Optional[str] = None,
    title: str = "Connectivity Matrix",
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Plot network connectivity matrix.
    
    Args:
        connectivity_matrix: N x N connectivity matrix
        filename: Optional file to save plot
        title: Plot title
        cmap: Colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(
        connectivity_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )
    
    ax.set_xlabel('Presynaptic Neuron', fontsize=12)
    ax.set_ylabel('Postsynaptic Neuron', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Weight', fontsize=12)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_membrane_potential(
    time: np.ndarray,
    voltage: np.ndarray,
    threshold: float = -55.0,
    filename: Optional[str] = None,
    title: str = "Membrane Potential"
) -> plt.Figure:
    """
    Plot membrane potential trace.
    
    Args:
        time: Time array (ms)
        voltage: Voltage array (mV)
        threshold: Spike threshold
        filename: Optional file to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(time, voltage, color='steelblue', linewidth=0.5, alpha=0.8)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label='Threshold')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ISI_distribution(
    isis: np.ndarray,
    filename: Optional[str] = None,
    title: str = "Inter-Spike Interval Distribution",
    bins: int = 50
) -> plt.Figure:
    """
    Plot ISI distribution.
    
    Args:
        isis: Array of inter-spike intervals (ms)
        filename: Optional file to save plot
        title: Plot title
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(isis, bins=bins, color='forestgreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Inter-Spike Interval (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_network_activity(
    network_stats: Dict[str, Any],
    filename: Optional[str] = None,
    title: str = "Network Activity"
) -> plt.Figure:
    """
    Plot comprehensive network activity summary.
    
    Args:
        network_stats: Dictionary of network statistics
        filename: Optional file to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rate distribution
    if 'rates' in network_stats:
        axes[0, 0].hist(network_stats['rates'], bins=30, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Firing Rate (Hz)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Firing Rate Distribution')
    
    # Raster plot
    if 'spike_times' in network_stats and 'neuron_ids' in network_stats:
        axes[0, 1].scatter(
            network_stats['spike_times'],
            network_stats['neuron_ids'],
            s=0.1, c='black', alpha=0.5
        )
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Neuron ID')
        axes[0, 1].set_title('Spike Raster')
    
    # Connectivity
    if 'connectivity' in network_stats:
        axes[1, 0].imshow(
            network_stats['connectivity'],
            cmap='viridis',
            aspect='auto'
        )
        axes[1, 0].set_xlabel('Presynaptic')
        axes[1, 0].set_ylabel('Postsynaptic')
        axes[1, 0].set_title('Connectivity')
    
    # Summary stats
    if 'summary' in network_stats:
        summary = network_stats['summary']
        stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in summary.items()])
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig
