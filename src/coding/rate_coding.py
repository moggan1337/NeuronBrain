"""
Rate Coding Implementation.

Rate coding is one of the simplest and most widely used neural coding
schemes. It represents information as the average firing rate of neurons
over a time window.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class RateCodingConfig:
    """Configuration for rate coding."""
    time_window: float = 100.0  # ms for rate calculation
    smooth_kernel_size: int = 5
    baseline_rate: float = 1.0  # Hz
    max_rate: float = 100.0  # Hz


class RateCoder:
    """
    Rate-based neural encoder/decoder.
    
    This class converts between neural spike trains and firing rates.
    It implements several rate estimation methods.
    
    Example:
        >>> coder = RateCoder(time_window=50.0)
        >>> 
        >>> # Encode: convert signal to firing rates
        >>> rates = coder.encode(signal, num_neurons=100)
        >>> 
        >>> # Decode: convert spikes to signal estimate
        >>> signal_estimate = coder.decode(spike_train, num_bins=100)
    """
    
    def __init__(self, config: Optional[RateCodingConfig] = None):
        """
        Initialize rate coder.
        
        Args:
            config: Coding configuration
        """
        self.config = config or RateCodingConfig()
    
    def encode(
        self, 
        signal: np.ndarray, 
        num_neurons: int = 100,
        tuning_curves: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Encode a signal as firing rates.
        
        Args:
            signal: Input signal values
            num_neurons: Number of neurons to use
            tuning_curves: Optional custom tuning curve function
            
        Returns:
            Array of firing rates (num_neurons x signal_length)
        """
        n_timesteps = len(signal)
        
        if tuning_curves is None:
            # Use linear tuning curves
            signal_normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
            rates = np.outer(signal_normalized, np.ones(num_neurons)) * self.config.max_rate
        else:
            rates = np.zeros((n_timesteps, num_neurons))
            for i, val in enumerate(signal):
                rates[i] = tuning_curves(val, num_neurons)
        
        return rates
    
    def decode(
        self, 
        spike_train: np.ndarray,
        num_bins: Optional[int] = None,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Decode firing rates from spike train.
        
        Args:
            spike_train: Binary spike train (n_neurons x n_timesteps)
            num_bins: Number of output bins (defaults to input length)
            method: Decoding method ('mean', 'pop_vector', 'optimal')
            
        Returns:
            Decoded signal
        """
        if num_bins is None:
            num_bins = spike_train.shape[1]
        
        if method == 'mean':
            # Simple mean rate
            rates = np.mean(spike_train, axis=0) * 1000.0 / self.config.time_window
            # Interpolate to desired bins
            return self._interpolate_signal(rates, num_bins)
        
        elif method == 'pop_vector':
            # Population vector coding
            return self._population_vector_decode(spike_train)
        
        elif method == 'optimal':
            # Optimal linear decoder
            return self._optimal_decode(spike_train)
        
        return np.zeros(num_bins)
    
    def _interpolate_signal(self, signal: np.ndarray, num_bins: int) -> np.ndarray:
        """Interpolate signal to desired length."""
        if len(signal) == num_bins:
            return signal
        
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, num_bins)
        
        return np.interp(x_new, x_old, signal)
    
    def _population_vector_decode(self, spike_train: np.ndarray) -> np.ndarray:
        """Population vector decoding."""
        # Weight each neuron by its preferred direction
        n_neurons = spike_train.shape[0]
        preferred_angles = np.linspace(0, 2*np.pi, n_neurons)
        
        rates = np.sum(spike_train, axis=1) / spike_train.shape[1]
        
        x = np.sum(rates * np.cos(preferred_angles))
        y = np.sum(rates * np.sin(preferred_angles))
        
        angle = np.arctan2(y, x)
        return np.array([np.cos(angle), np.sin(angle)])
    
    def _optimal_decode(self, spike_train: np.ndarray) -> np.ndarray:
        """Optimal linear decoder (Wiener filter)."""
        # Simplified version
        rates = np.mean(spike_train, axis=0)
        return rates / (np.max(rates) + 1e-10)


class PoissonCoder:
    """
    Poisson spike generator for rate coding.
    
    Generates spike trains with Poisson statistics based on firing rates.
    
    Reference:
        Dayan, P. and Abbott, L.F. (2001). Theoretical Neuroscience.
        MIT Press.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        refractory_period: float = 0.0
    ):
        """
        Initialize Poisson coder.
        
        Args:
            seed: Random seed for reproducibility
            refractory_period: Minimum ISI (ms)
        """
        self.rng = np.random.RandomState(seed)
        self.refractory_period = refractory_period
        self.last_spike_times: Dict[int, float] = {}
    
    def generate_spikes(
        self,
        rates: np.ndarray,
        dt: float = 1.0,
        neuron_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Poisson spike train from rates.
        
        Args:
            rates: Firing rates in Hz (array)
            dt: Time step in ms
            neuron_id: Optional ID for refractory tracking
            
        Returns:
            Binary spike train
        """
        n_timesteps = len(rates)
        spikes = np.zeros(n_timesteps)
        
        # Convert to spike probability per timestep
        # P(spike) = 1 - exp(-r*dt/1000) for dt in ms, r in Hz
        probs = 1 - np.exp(-rates * dt / 1000.0)
        
        # Generate spikes
        spikes = (self.rng.random(n_timesteps) < probs).astype(float)
        
        # Apply refractory period
        if self.refractory_period > 0 and neuron_id is not None:
            spikes = self._apply_refractory(spikes, dt, neuron_id)
        
        return spikes
    
    def _apply_refractory(
        self, 
        spikes: np.ndarray, 
        dt: float,
        neuron_id: int
    ) -> np.ndarray:
        """Apply refractory period to spike train."""
        last_time = self.last_spike_times.get(neuron_id, -self.refractory_period)
        
        refractory_steps = int(self.refractory_period / dt)
        result = spikes.copy()
        
        for i in range(len(spikes)):
            if i > 0 and spikes[i] > 0:
                if i * dt - last_time < self.refractory_period:
                    result[i] = 0.0
                else:
                    last_time = i * dt
        
        self.last_spike_times[neuron_id] = last_time
        return result
    
    def generate_population_spikes(
        self,
        rates: np.ndarray,
        num_neurons: int,
        dt: float = 1.0,
        correlations: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate spike trains for a population of neurons.
        
        Args:
            rates: Firing rates (n_timesteps x num_neurons)
            num_neurons: Number of neurons
            dt: Time step (ms)
            correlations: Optional correlation coefficient
            
        Returns:
            Population spike train (num_neurons x n_timesteps)
        """
        if rates.ndim == 1:
            rates = np.tile(rates.reshape(-1, 1), (1, num_neurons))
        
        spike_trains = np.zeros((num_neurons, rates.shape[0]))
        
        for n in range(num_neurons):
            spike_trains[n] = self.generate_spikes(rates[:, n], dt, neuron_id=n)
        
        # Add correlations if specified
        if correlations is not None and correlations > 0:
            spike_trains = self._add_correlations(spike_trains, correlations)
        
        return spike_trains
    
    def _add_correlations(
        self, 
        spike_trains: np.ndarray, 
        corr: float
    ) -> np.ndarray:
        """Add correlations to spike trains."""
        n_neurons, n_timesteps = spike_trains.shape
        
        # Generate common input
        common_input = (self.rng.random(n_timesteps) < corr).astype(float)
        
        # Add to all neurons with some probability
        for n in range(n_neurons):
            add_common = self.rng.random(n_timesteps) < 0.3
            spike_trains[n] = np.maximum(spike_trains[n], common_input * add_common)
        
        return spike_trains
    
    def compute_spike_triggered_average(
        self,
        spikes: np.ndarray,
        stimulus: np.ndarray,
        window_ms: float = 50.0,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute spike-triggered average (STA).
        
        Args:
            spikes: Binary spike train
            stimulus: Stimulus values
            window_ms: Window size for STA
            dt: Time step
            
        Returns:
            Spike-triggered average
        """
        window_size = int(window_ms / dt)
        spike_times = np.where(spikes > 0)[0]
        
        if len(spike_times) < 10:
            return np.zeros(window_size)
        
        sta = np.zeros(window_size)
        count = 0
        
        for t in spike_times:
            if t >= window_size:
                window = stimulus[t-window_size:t]
                sta += window
                count += 1
        
        return sta / max(count, 1)
