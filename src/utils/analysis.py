"""
Analysis utilities for NeuronBrain simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_ISI_distribution(
    spike_times: List[float],
    num_bins: int = 50,
    max_isi: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inter-spike interval distribution.
    
    Args:
        spike_times: Sorted spike times
        num_bins: Number of histogram bins
        max_isi: Maximum ISI to consider
        
    Returns:
        Tuple of (bin_centers, histogram)
    """
    if len(spike_times) < 2:
        return np.array([]), np.array([])
    
    isis = np.diff(spike_times)
    
    if max_isi:
        isis = isis[isis < max_isi]
    
    hist, edges = np.histogram(isis, bins=num_bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    return bin_centers, hist


def compute_cv(spike_times: List[float]) -> float:
    """
    Compute coefficient of variation of ISI.
    
    CV = std(ISI) / mean(ISI)
    CV = 1 for Poisson, < 1 for regular, > 1 for bursty
    
    Args:
        spike_times: Sorted spike times
        
    Returns:
        Coefficient of variation
    """
    if len(spike_times) < 2:
        return 0.0
    
    isis = np.diff(spike_times)
    
    if np.mean(isis) == 0:
        return 0.0
    
    return np.std(isis) / np.mean(isis)


def compute_cv2(spike_times: List[float]) -> float:
    """
    Compute local coefficient of variation (CV2).
    
    CV2 = 2 * |ISI_n+1 - ISI_n| / (ISI_n+1 + ISI_n)
    
    More robust to slow rate changes than CV.
    
    Args:
        spike_times: Sorted spike times
        
    Returns:
        CV2 value
    """
    if len(spike_times) < 3:
        return 0.0
    
    isis = np.diff(spike_times)
    
    cv2_values = []
    for i in range(len(isis) - 1):
        isi_sum = isis[i] + isis[i + 1]
        if isi_sum > 0:
            cv2 = 2 * abs(isis[i + 1] - isis[i]) / isi_sum
            cv2_values.append(cv2)
    
    return np.mean(cv2_values) if cv2_values else 0.0


def detect_bursts(
    spike_times: List[float],
    burst_threshold: float = 10.0
) -> List[List[float]]:
    """
    Detect bursts in spike train.
    
    A burst is a sequence of spikes with short inter-spike intervals.
    
    Args:
        spike_times: Sorted spike times
        burst_threshold: ISI threshold for burst detection (ms)
        
    Returns:
        List of bursts, each burst is a list of spike times
    """
    if len(spike_times) < 3:
        return []
    
    isis = np.diff(spike_times)
    bursts = []
    current_burst = [spike_times[0]]
    
    for i, isi in enumerate(isis):
        if isi < burst_threshold:
            current_burst.append(spike_times[i + 1])
        else:
            if len(current_burst) >= 3:  # Minimum 3 spikes for burst
                bursts.append(current_burst)
            current_burst = [spike_times[i + 1]]
    
    # Don't forget the last burst
    if len(current_burst) >= 3:
        bursts.append(current_burst)
    
    return bursts


def compute_population_rate(
    spike_times: List[float],
    neuron_ids: List[int],
    time_window: float = 10.0,
    t_start: float = 0.0,
    t_end: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute population firing rate over time.
    
    Args:
        spike_times: Spike times
        neuron_ids: Corresponding neuron IDs
        time_window: Window size (ms)
        t_start: Start time
        t_end: End time
        
    Returns:
        Tuple of (time_centers, population_rates)
    """
    time_bins = np.arange(t_start, t_end, time_window)
    rates = np.zeros(len(time_bins) - 1)
    
    for i in range(len(time_bins) - 1):
        t0, t1 = time_bins[i], time_bins[i + 1]
        spikes_in_window = np.sum((spike_times >= t0) & (spike_times < t1))
        rates[i] = spikes_in_window / (time_window / 1000)  # Hz
    
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    return time_centers, rates


def compute_synchrony(
    spike_trains: List[List[float]],
    window_ms: float = 5.0
) -> float:
    """
    Compute population synchrony (Kuramoto order parameter).
    
    Args:
        spike_trains: List of spike trains (one per neuron)
        window_ms: Synchrony window
        
    Returns:
        Order parameter r (0 = async, 1 = sync)
    """
    if not spike_trains or not spike_trains[0]:
        return 0.0
    
    # Convert to phases
    all_times = []
    for train in spike_trains:
        all_times.extend(train)
    
    if not all_times:
        return 0.0
    
    t_min, t_max = min(all_times), max(all_times)
    duration = t_max - t_min
    
    if duration < window_ms:
        return 0.0
    
    # Compute phase coherence
    phases = []
    for train in spike_trains:
        for spike in train:
            phase = 2 * np.pi * spike / window_ms
            phases.append(phase)
    
    if not phases:
        return 0.0
    
    # Order parameter
    mean_vector = np.mean(np.exp(1j * np.array(phases)))
    r = np.abs(mean_vector)
    
    return r


def compute_information(
    spike_train: List[float],
    signal: np.ndarray,
    time_resolution: float = 1.0
) -> Dict[str, float]:
    """
    Compute information-theoretic measures.
    
    Args:
        spike_train: Binary spike train
        signal: Stimulus/behavior signal
        time_resolution: Time resolution (ms)
        
    Returns:
        Dictionary of information measures
    """
    # Fisher information (simplified)
    signal_diff = np.diff(signal)
    signal_std = np.std(signal_diff)
    
    # Spike-triggered average
    if len(spike_train) < 10:
        return {'fisher_info': 0.0, 'mutual_info': 0.0}
    
    spike_indices = np.where(np.array(spike_train) > 0)[0]
    
    if len(spike_indices) < 2:
        return {'fisher_info': 0.0, 'mutual_info': 0.0}
    
    # Simplified Fisher information
    sta = compute_spike_triggered_average(spike_train, signal)
    fisher_info = np.sum(sta**2) * time_resolution
    
    # Mutual information (simplified)
    rate = np.mean(spike_train)
    if rate > 0 and rate < 1:
        p_spike = rate
        p_no_spike = 1 - rate
        
        # Assume Gaussian signal
        signal_entropy = 0.5 * np.log(2 * np.pi * np.e * np.var(signal))
        
        # Simplified mutual info
        mutual_info = signal_entropy * p_spike
    else:
        mutual_info = 0.0
    
    return {
        'fisher_info': fisher_info,
        'mutual_info': mutual_info,
        'spike_rate': np.mean(spike_train) / time_resolution * 1000,
    }


def compute_spike_triggered_average(
    spike_train: List[float],
    signal: np.ndarray,
    window_ms: float = 50.0,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute spike-triggered average.
    
    Args:
        spike_train: Binary spike train
        signal: Stimulus signal
        window_ms: STA window
        dt: Time step
        
    Returns:
        Spike-triggered average
    """
    window_size = int(window_ms / dt)
    spike_indices = np.where(np.array(spike_train) > 0)[0]
    
    if len(spike_indices) < 5:
        return np.zeros(window_size)
    
    sta = np.zeros(window_size)
    count = 0
    
    for idx in spike_indices:
        if idx >= window_size:
            window = signal[idx - window_size:idx]
            if len(window) == window_size:
                sta += window
                count += 1
    
    return sta / max(count, 1)


def compute_correlation_coefficient(
    spike_train_1: List[float],
    spike_train_2: List[float],
    dt: float = 1.0,
    max_lag_ms: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two spike trains.
    
    Args:
        spike_train_1: First spike train (binary)
        spike_train_2: Second spike train (binary)
        dt: Time step
        max_lag_ms: Maximum lag
        
    Returns:
        Tuple of (lags, correlations)
    """
    n1 = np.array(spike_train_1)
    n2 = np.array(spike_train_2)
    
    max_lag = int(max_lag_ms / dt)
    
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            c = np.correlate(n1[:lag], n2[-lag:], mode='valid')[0]
        elif lag > 0:
            c = np.correlate(n1[lag:], n2[: -lag if lag > 0 else None], mode='valid')[0]
        else:
            c = np.correlate(n1, n2, mode='valid')[0]
        
        correlations.append(c)
    
    return np.array(list(lags)) * dt, np.array(correlations)


def compute_autocorrelation(
    spike_train: List[float],
    dt: float = 1.0,
    max_lag_ms: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spike train autocorrelation.
    
    Args:
        spike_train: Binary spike train
        dt: Time step
        max_lag_ms: Maximum lag
        
    Returns:
        Tuple of (lags, autocorrelations)
    """
    return compute_correlation_coefficient(spike_train, spike_train, dt, max_lag_ms)
