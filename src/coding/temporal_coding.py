"""
Temporal Coding Implementation.

Temporal coding represents information in the precise timing of spikes,
rather than just the average firing rate. This module implements various
temporal coding schemes including:
- Phase coding
- Time-to-first-spike coding
- Temporal patterns
- Synchrony coding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TemporalCodingConfig:
    """Configuration for temporal coding."""
    time_precision: float = 1.0  # ms
    phase_bins: int = 32
    synchronization_window: float = 5.0  # ms


class TemporalCoder:
    """
    Temporal neural encoder/decoder.
    
    This class handles encoding and decoding of temporally coded signals.
    
    Example:
        >>> coder = TemporalCoder()
        >>> 
        >>> # Encode using time-to-first-spike
        >>> first_spikes = coder.time_to_first_spike(signal, num_neurons=100)
        >>> 
        >>> # Decode from spike times
        >>> signal_estimate = coder.decode_from_times(spike_times, num_bins=100)
    """
    
    def __init__(self, config: Optional[TemporalCodingConfig] = None):
        """
        Initialize temporal coder.
        
        Args:
            config: Coding configuration
        """
        self.config = config or TemporalCodingConfig()
    
    def time_to_first_spike(
        self,
        signal: np.ndarray,
        num_neurons: int = 100,
        t_max: float = 100.0
    ) -> np.ndarray:
        """
        Encode signal using time-to-first-spike coding.
        
        Earlier spikes encode higher values.
        
        Args:
            signal: Input signal (normalized 0-1)
            num_neurons: Number of neurons
            t_max: Maximum spike time (ms)
            
        Returns:
            First spike times for each neuron
        """
        signal_normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
        
        first_spikes = np.zeros((len(signal), num_neurons))
        
        for t, val in enumerate(signal_normalized):
            # Assign spike times inversely proportional to signal
            spike_times = t_max * (1 - val) + np.linspace(0, t_max/10, num_neurons)
            first_spikes[t] = spike_times
        
        return first_spikes
    
    def latency_code(
        self,
        signal: np.ndarray,
        num_neurons: int = 100,
        t_min: float = 5.0,
        t_max: float = 50.0
    ) -> np.ndarray:
        """
        Encode using latency coding (synchronized onset).
        
        Args:
            signal: Input signal
            num_neurons: Number of neurons
            t_min: Minimum latency
            t_max: Maximum latency
            
        Returns:
            Spike times
        """
        signal_normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
        
        # All neurons fire within a window, with timing based on signal
        latencies = t_min + (t_max - t_min) * (1 - signal_normalized.reshape(-1, 1))
        
        # Add some noise
        noise = np.random.uniform(-2, 2, latencies.shape)
        latencies += noise
        
        return latencies
    
    def phase_code(
        self,
        signal: np.ndarray,
        oscillation_frequency: float = 8.0,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode signal as spike phase relative to oscillation.
        
        Args:
            signal: Input signal
            oscillation_frequency: Frequency of reference oscillation (Hz)
            dt: Time step (ms)
            
        Returns:
            Tuple of (spike_times, spike_phases)
        """
        n_timesteps = len(signal)
        
        # Create reference oscillation
        t = np.arange(n_timesteps) * dt
        oscillation = np.sin(2 * np.pi * oscillation_frequency * t / 1000)
        
        # Find peaks and troughs
        peaks = np.where(np.diff(oscillation) < 0)[0]
        
        # Assign phases to spikes (simplified)
        # Higher signal -> spike earlier in cycle
        spike_phases = np.zeros(n_timesteps)
        spike_times = []
        
        for i, val in enumerate(signal):
            # Find current phase
            phase = np.arctan2(oscillation[i], np.sqrt(1 - oscillation[i]**2 + 1e-10))
            spike_phases[i] = phase
            
            # Spike if signal crosses threshold relative to phase
            if val > 0.7 and phase > 0:
                spike_times.append(i * dt)
        
        return np.array(spike_times), spike_phases
    
    def decode_from_times(
        self,
        spike_times: np.ndarray,
        num_bins: int = 100,
        t_total: float = 1000.0
    ) -> np.ndarray:
        """
        Decode signal from spike times.
        
        Args:
            spike_times: Spike times in ms
            num_bins: Number of output bins
            t_total: Total recording time
            
        Returns:
            Decoded signal
        """
        bins = np.linspace(0, t_total, num_bins + 1)
        spike_counts, _ = np.histogram(spike_times, bins=bins)
        
        return spike_counts.astype(float) / max(spike_counts.max(), 1)
    
    def compute_synchrony(
        self,
        spike_trains: np.ndarray,
        window_ms: float = 5.0,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute synchrony across neuron population.
        
        Args:
            spike_trains: Binary spike trains (n_neurons x n_timesteps)
            window_ms: Synchrony window
            dt: Time step
            
        Returns:
            Synchrony measure over time
        """
        n_neurons, n_timesteps = spike_trains.shape
        window = int(window_ms / dt)
        
        synchrony = np.zeros(n_timesteps)
        
        for t in range(window, n_timesteps):
            window_spikes = spike_trains[:, t-window:t]
            # Count neurons that spiked in window
            spikes_in_window = np.sum(window_spikes, axis=1)
            # Synchrony = number of neurons that spiked together
            synchrony[t] = np.mean(spikes_in_window > 0)
        
        return synchrony
    
    def detect_temporal_patterns(
        self,
        spike_trains: np.ndarray,
        pattern_length_ms: float = 20.0,
        dt: float = 1.0,
        min_occurrences: int = 3
    ) -> Dict[str, List]:
        """
        Detect recurring temporal patterns in spike trains.
        
        Args:
            spike_trains: Binary spike trains
            pattern_length_ms: Length of patterns to detect
            dt: Time step
            min_occurrences: Minimum pattern occurrences
            
        Returns:
            Dictionary of detected patterns
        """
        pattern_length = int(pattern_length_ms / dt)
        n_neurons, n_timesteps = spike_trains.shape
        
        patterns = []
        
        # Slide window and find patterns
        for start in range(0, n_timesteps - pattern_length, pattern_length // 2):
            window = spike_trains[:, start:start + pattern_length]
            
            # Check if pattern is "interesting" (not all zeros)
            if np.sum(window) > n_neurons * 0.1:
                patterns.append(window)
        
        # Find similar patterns (simplified clustering)
        unique_patterns = self._cluster_patterns(patterns, threshold=0.8)
        
        return {
            'patterns': unique_patterns,
            'num_unique': len(unique_patterns),
            'pattern_length_ms': pattern_length_ms
        }
    
    def _cluster_patterns(
        self, 
        patterns: List, 
        threshold: float
    ) -> List:
        """Cluster similar patterns."""
        if len(patterns) < 2:
            return patterns
        
        unique = [patterns[0]]
        
        for p in patterns[1:]:
            is_unique = True
            for u in unique:
                similarity = self._pattern_similarity(p, u)
                if similarity > threshold:
                    is_unique = False
                    break
            if is_unique:
                unique.append(p)
        
        return unique
    
    def _pattern_similarity(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute similarity between two patterns."""
        if p1.shape != p2.shape:
            return 0.0
        
        intersection = np.sum((p1 > 0) & (p2 > 0))
        union = np.sum((p1 > 0) | (p2 > 0))
        
        return intersection / max(union, 1)


class PhaseCoder:
    """
    Phase coding with oscillatory population dynamics.
    
    Encodes information in the phase of spikes relative to a
    population oscillation (e.g., theta-gamma coupling).
    """
    
    def __init__(
        self,
        theta_frequency: float = 8.0,
        gamma_frequency: float = 40.0
    ):
        """
        Initialize phase coder.
        
        Args:
            theta_frequency: Theta oscillation frequency (Hz)
            gamma_frequency: Gamma oscillation frequency (Hz)
        """
        self.theta_freq = theta_frequency
        self.gamma_freq = gamma_frequency
    
    def compute_phase_histogram(
        self,
        spike_times: np.ndarray,
        oscillation_frequency: float,
        num_cycles: int = 100,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute phase histogram of spikes.
        
        Args:
            spike_times: Spike times
            oscillation_frequency: Reference frequency (Hz)
            num_cycles: Number of oscillation cycles
            dt: Time step
            
        Returns:
            Phase histogram (num_bins)
        """
        num_bins = 36  # 10-degree bins
        phases = np.zeros(num_bins)
        
        for spike in spike_times:
            phase = (spike * oscillation_frequency * 360 / 1000) % 360
            bin_idx = int(phase / 10) % num_bins
            phases[bin_idx] += 1
        
        return phases / max(np.sum(phases), 1)
    
    def theta_gamma_coupling(
        self,
        spike_times: np.ndarray,
        theta_cycles: np.ndarray,
        t_total: float
    ) -> np.ndarray:
        """
        Analyze theta-gamma coupling.
        
        Args:
            spike_times: Spike times
            theta_cycles: Theta cycle boundaries
            t_total: Total time
            
        Returns:
            Array of gamma phases within each theta cycle
        """
        gamma_spikes_per_theta = []
        
        for i in range(len(theta_cycles) - 1):
            t_start = theta_cycles[i]
            t_end = theta_cycles[i + 1]
            
            spikes_in_cycle = spike_times[(spike_times >= t_start) & (spike_times < t_end)]
            
            if len(spikes_in_cycle) > 0:
                # Compute phase within theta cycle
                relative_times = (spikes_in_cycle - t_start) / (t_end - t_start)
                gamma_phases = relative_times * 360 * (self.gamma_freq / self.theta_freq)
                gamma_spikes_per_theta.append(gamma_phases)
        
        return gamma_spikes_per_theta
