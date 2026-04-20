"""
Population Coding Implementation.

Population coding uses the coordinated activity of many neurons
to encode information. This module implements various population
coding schemes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PopulationCodingConfig:
    """Configuration for population coding."""
    num_neurons: int = 100
    tuning_curve_width: float = 30.0  # degrees or arbitrary units
    noise_std: float = 2.0
    max_firing_rate: float = 100.0  # Hz


class PopulationCoder:
    """
    Population coding encoder/decoder.
    
    This class implements population coding where each neuron has
    a preferred stimulus value (tuning curve) and activity represents
    similarity to the stimulus.
    
    Example:
        >>> coder = PopulationCoder(num_neurons=100, tuning_width=30)
        >>> 
        >>> # Encode: compute population response
        >>> response = coder.encode(stimulus_value=45.0)
        >>> 
        >>> # Decode: estimate stimulus from population response
        >>> estimate = coder.decode(response)
    """
    
    def __init__(self, config: Optional[PopulationCodingConfig] = None):
        """
        Initialize population coder.
        
        Args:
            config: Coding configuration
        """
        self.config = config or PopulationCodingConfig()
        
        # Generate tuning curves
        self.preferred_values = self._generate_preferred_values()
        self.tuning_curves = self._generate_tuning_functions()
    
    def _generate_preferred_values(self) -> np.ndarray:
        """Generate preferred stimulus values for each neuron."""
        n = self.config.num_neurons
        
        # Distribute uniformly over the range
        return np.linspace(0, 360, n) if n > 0 else np.array([])
    
    def _generate_tuning_functions(self) -> np.ndarray:
        """Generate Gaussian tuning curve parameters."""
        n = self.config.num_neurons
        
        # For each neuron: (preferred, amplitude, width)
        tuning = np.zeros((n, 3))
        tuning[:, 0] = self.preferred_values  # preferred value
        tuning[:, 1] = self.config.max_firing_rate  # amplitude
        tuning[:, 2] = self.config.tuning_curve_width  # width
        
        return tuning
    
    def encode(
        self,
        stimulus_value: float,
        noise: bool = True,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Encode a stimulus value as population activity.
        
        Args:
            stimulus_value: The value to encode
            noise: Whether to add noise
            rng: Random number generator
            
        Returns:
            Firing rates for each neuron
        """
        n = self.config.num_neurons
        
        # Compute tuning curve response
        preferred = self.tuning_curves[:, 0]
        amplitude = self.tuning_curves[:, 1]
        width = self.tuning_curves[:, 2]
        
        # Gaussian tuning curve
        diff = self._circular_diff(stimulus_value, preferred, 360)
        rates = amplitude * np.exp(-diff**2 / (2 * width**2))
        
        # Add noise
        if noise:
            if rng is None:
                rng = np.random.RandomState()
            rates += rng.normal(0, self.config.noise_std, n)
            rates = np.maximum(rates, 0)
        
        return rates
    
    def _circular_diff(self, x: float, y: np.ndarray, period: float) -> np.ndarray:
        """Compute circular difference."""
        diff = y - x
        return np.mod(diff + period / 2, period) - period / 2
    
    def decode(
        self,
        population_response: np.ndarray,
        method: str = 'population_vector'
    ) -> float:
        """
        Decode stimulus estimate from population response.
        
        Args:
            population_response: Firing rates of neurons
            method: Decoding method
            
        Returns:
            Estimated stimulus value
        """
        if method == 'population_vector':
            return self._population_vector_decode(population_response)
        elif method == 'optimal':
            return self._optimal_decode(population_response)
        elif method == 'center_of_mass':
            return self._center_of_mass_decode(population_response)
        elif method == 'winner_take_all':
            return self._winner_take_all_decode(population_response)
        
        return 0.0
    
    def _population_vector_decode(self, response: np.ndarray) -> float:
        """
        Population vector decoding.
        
        Computes the weighted average of preferred values.
        """
        preferred = self.preferred_values
        
        # Subtract baseline
        baseline = np.percentile(response, 20)
        response_centered = np.maximum(response - baseline, 0)
        
        # Compute vector sum
        x = np.sum(response_centered * np.cos(2 * np.pi * preferred / 360))
        y = np.sum(response_centered * np.sin(2 * np.pi * preferred / 360))
        
        angle = np.arctan2(y, x) * 180 / np.pi
        return np.mod(angle, 360)
    
    def _optimal_decode(self, response: np.ndarray) -> float:
        """
        Optimal linear decoding (Wiener filter).
        
        Uses pseudo-inverse to find optimal weights.
        """
        # This would use precomputed optimal weights
        # Simplified version using least squares
        preferred = self.preferred_values / 360  # Normalize
        
        # Weighted average
        response_pos = np.maximum(response, 0)
        if np.sum(response_pos) == 0:
            return 0.0
        
        estimate = np.sum(response_pos * preferred) / np.sum(response_pos)
        return estimate * 360
    
    def _center_of_mass_decode(self, response: np.ndarray) -> float:
        """Center of mass decoding."""
        baseline = np.percentile(response, 20)
        response_centered = np.maximum(response - baseline, 0)
        
        if np.sum(response_centered) == 0:
            return 0.0
        
        return np.sum(response_centered * self.preferred_values) / np.sum(response_centered)
    
    def _winner_take_all_decode(self, response: np.ndarray) -> float:
        """Winner-take-all decoding."""
        winner = np.argmax(response)
        return self.preferred_values[winner]
    
    def decode_uncertainty(self, population_response: np.ndarray) -> Tuple[float, float]:
        """
        Decode with uncertainty estimate.
        
        Args:
            population_response: Population response
            
        Returns:
            Tuple of (estimate, uncertainty)
        """
        estimate = self.decode(population_response)
        
        # Uncertainty from population spread
        baseline = np.percentile(population_response, 20)
        response_centered = np.maximum(population_response - baseline, 0)
        
        if np.sum(response_centered) == 0:
            return estimate, 180.0
        
        # Weighted circular variance
        preferred = self.preferred_values
        diff = self._circular_diff(preferred, estimate, 360)
        variance = np.sqrt(np.sum(response_centered * diff**2) / np.sum(response_centered))
        
        return estimate, variance
    
    def get_tuning_curve(self, neuron_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tuning curve for a neuron.
        
        Args:
            neuron_id: Neuron index
            
        Returns:
            Tuple of (stimulus_values, firing_rates)
        """
        if neuron_id >= self.config.num_neurons:
            raise ValueError(f"Neuron {neuron_id} out of range")
        
        stimulus_values = np.linspace(0, 360, 100)
        rates = np.zeros(100)
        
        for i, val in enumerate(stimulus_values):
            diff = self._circular_diff(val, self.tuning_curves[neuron_id, 0], 360)
            rates[i] = self.tuning_curves[neuron_id, 1] * np.exp(
                -diff**2 / (2 * self.tuning_curves[neuron_id, 2]**2)
            )
        
        return stimulus_values, rates


class VectorCoder:
    """
    Vector population coding for multi-dimensional stimuli.
    
    Encodes direction vectors or positions in 2D/3D space.
    """
    
    def __init__(self, dimensions: int = 2, num_neurons: int = 100):
        """
        Initialize vector coder.
        
        Args:
            dimensions: Number of spatial dimensions (2 or 3)
            num_neurons: Number of neurons
        """
        self.dimensions = dimensions
        self.num_neurons = num_neurons
        
        # Generate preferred directions
        if dimensions == 2:
            self.preferred_directions = np.random.uniform(0, 2*np.pi, num_neurons)
        else:
            # Uniform sampling on sphere
            phi = np.random.uniform(0, 2*np.pi, num_neurons)
            theta = np.arccos(2 * np.random.uniform(0, 1, num_neurons) - 1)
            self.preferred_directions = np.stack([theta, phi], axis=1)
        
        # Tuning parameters
        self.tuning_width = 0.5  # radians
        self.max_rate = 100.0
    
    def encode_direction(
        self,
        direction: np.ndarray,
        noise: bool = True
    ) -> np.ndarray:
        """
        Encode a direction vector.
        
        Args:
            direction: Direction vector (2D or 3D)
            noise: Add noise
            
        Returns:
            Firing rates
        """
        direction = np.array(direction)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        if self.dimensions == 2:
            angle = np.arctan2(direction[1], direction[0])
            cos_diff = np.cos(self.preferred_directions - angle)
            rates = self.max_rate * np.exp((cos_diff - 1) / self.tuning_width**2)
        else:
            # 3D dot product
            cos_diff = self._cos_diff_3d(direction)
            rates = self.max_rate * np.exp((cos_diff - 1) / self.tuning_width**2)
        
        if noise:
            rates += np.random.normal(0, 5, rates.shape)
            rates = np.maximum(rates, 0)
        
        return rates
    
    def _cos_diff_3d(self, direction: np.ndarray) -> np.ndarray:
        """Compute cosine difference for 3D directions."""
        x = np.sin(self.preferred_directions[:, 0]) * np.cos(self.preferred_directions[:, 1])
        y = np.sin(self.preferred_directions[:, 0]) * np.sin(self.preferred_directions[:, 1])
        z = np.cos(self.preferred_directions[:, 0])
        
        preferred = np.stack([x, y, z], axis=1)
        
        # Dot product
        dot = np.sum(preferred * direction, axis=1)
        return np.clip(dot, -1, 1)
    
    def decode_direction(self, rates: np.ndarray) -> np.ndarray:
        """
        Decode direction from population response.
        
        Args:
            rates: Firing rates
            
        Returns:
            Direction vector
        """
        if self.dimensions == 2:
            x = np.sum(rates * np.cos(self.preferred_directions))
            y = np.sum(rates * np.sin(self.preferred_directions))
            
            norm = np.sqrt(x**2 + y**2) + 1e-10
            return np.array([x, y]) / norm
        else:
            x = np.sum(rates * np.sin(self.preferred_directions[:, 0]) * 
                      np.cos(self.preferred_directions[:, 1]))
            y = np.sum(rates * np.sin(self.preferred_directions[:, 0]) * 
                      np.sin(self.preferred_directions[:, 1]))
            z = np.sum(rates * np.cos(self.preferred_directions[:, 0]))
            
            direction = np.array([x, y, z])
            norm = np.linalg.norm(direction) + 1e-10
            return direction / norm
    
    def encode_position(
        self,
        position: np.ndarray,
        field_centers: Optional[np.ndarray] = None,
        field_width: float = 50.0
    ) -> np.ndarray:
        """
        Encode position (place fields).
        
        Args:
            position: Position vector
            field_centers: Place field centers (use default if None)
            field_width: Width of place fields
            
        Returns:
            Firing rates
        """
        if field_centers is None:
            field_centers = np.random.uniform(0, 100, (self.num_neurons, self.dimensions))
        
        # Compute distance from each field center
        distances = np.linalg.norm(field_centers - position, axis=1)
        
        # Gaussian place fields
        rates = self.max_rate * np.exp(-distances**2 / (2 * field_width**2))
        
        return rates
