"""
Cortical Column Model.

This module implements a simplified cortical column model with
layered structure and realistic connectivity patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CorticalColumnConfig:
    """Configuration for a cortical column."""
    # Layer sizes (neurons per layer)
    L2_3_size: int = 50
    L4_size: int = 40
    L5_size: int = 30
    L6_size: int = 30
    
    # Synaptic parameters
    AMPA_conductance: float = 1.0
    NMDA_conductance: float = 0.8
    GABA_A_conductance: float = 1.0
    gap_junction_conductance: float = 0.1
    
    # Connection probabilities
    intra_layer_prob: float = 0.1
    inter_layer_prob: float = 0.05
    
    # Neuron parameters
    neuron_type: str = "izhikevich"
    exc_ratio: float = 0.85


class Layer:
    """Represents a cortical layer."""
    
    def __init__(self, name: str, size: int, neuron_indices: List[int]):
        self.name = name
        self.size = size
        self.neuron_indices = neuron_indices
        self.input_conductance: float = 0.0
        self.output_conductance: float = 0.0
    
    @property
    def range(self) -> Tuple[int, int]:
        """Return (start, end) indices for this layer."""
        return (min(self.neuron_indices), max(self.neuron_indices) + 1)


class CorticalColumn:
    """
    Cortical Column Model.
    
    This class implements a simplified cortical column with:
    - 4 cortical layers (L2/3, L4, L5, L6)
    - Realistic connectivity patterns between layers
    - Excitatory and inhibitory neurons
    - Thalamic input and cortical output
    
    Connectivity patterns:
    - L4 receives sensory input (thalamus)
    - L2/3 processes intracortical signals
    - L5 projects to other areas
    - L6 modulates thalamic feedback
    
    Example:
        >>> config = CorticalColumnConfig()
        >>> column = CorticalColumn(config)
        >>> 
        >>> # Inject sensory input
        >>> column.inject_thalamic_input(L4_neurons, intensity=5.0)
        >>> 
        >>> # Run simulation
        >>> for t in range(10000):
        ...     column.step(0.1)
    """
    
    LAYER_ORDER = ['L2_3', 'L4', 'L5', 'L6']
    
    def __init__(self, config: Optional[CorticalColumnConfig] = None):
        """
        Initialize cortical column.
        
        Args:
            config: Column configuration
        """
        self.config = config or CorticalColumnConfig()
        
        # Calculate total size
        self.total_size = (
            self.config.L2_3_size +
            self.config.L4_size +
            self.config.L5_size +
            self.config.L6_size
        )
        
        # Create layers
        self.layers: Dict[str, Layer] = {}
        self._create_layers()
        
        # Initialize network (will be set externally)
        self.network = None
        
        # Connectivity matrices
        self.connectivity: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Thalamic input
        self.thalamic_input: float = 0.0
        self.thalamic_target: str = 'L4'
        
        # State
        self.time: float = 0.0
        
    def _create_layers(self):
        """Create layer objects."""
        idx = 0
        
        for layer_name in self.LAYER_ORDER:
            size = getattr(self.config, f"{layer_name}_size")
            indices = list(range(idx, idx + size))
            self.layers[layer_name] = Layer(layer_name, size, indices)
            idx += size
    
    def get_layer_neurons(self, layer_name: str) -> List[int]:
        """
        Get neuron indices for a layer.
        
        Args:
            layer_name: Layer name (L2_3, L4, L5, L6)
            
        Returns:
            List of neuron indices
        """
        if layer_name not in self.layers:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self.layers[layer_name].neuron_indices
    
    def get_layer_properties(self, layer_name: str) -> Dict:
        """
        Get properties of a layer.
        
        Args:
            layer_name: Layer name
            
        Returns:
            Dictionary of layer properties
        """
        layer = self.layers.get(layer_name)
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        return {
            'name': layer.name,
            'size': layer.size,
            'neuron_range': layer.range,
            'input_conductance': layer.input_conductance,
            'output_conductance': layer.output_conductance,
        }
    
    def inject_thalamic_input(
        self,
        target_layer: str = 'L4',
        intensity: float = 5.0,
        pattern: Optional[np.ndarray] = None
    ):
        """
        Inject thalamic input to the column.
        
        Args:
            target_layer: Target layer (default: L4)
            intensity: Input intensity
            pattern: Optional input pattern (firing rates)
        """
        self.thalamic_target = target_layer
        self.thalamic_input = intensity
        
        if pattern is not None:
            # Apply pattern to target layer neurons
            target_indices = self.get_layer_neurons(target_layer)
            for i, idx in enumerate(target_indices):
                if self.network and idx < len(self.network.current_input):
                    if i < len(pattern):
                        self.network.current_input[idx] += pattern[i]
                    else:
                        self.network.current_input[idx] += intensity
    
    def get_layer_activity(self, layer_name: str) -> Dict[str, float]:
        """
        Get activity statistics for a layer.
        
        Args:
            layer_name: Layer name
            
        Returns:
            Dictionary with activity statistics
        """
        if layer_name not in self.layers:
            return {}
        
        if self.network is None:
            return {}
        
        layer_indices = self.get_layer_neurons(layer_name)
        
        # Get membrane potentials
        voltages = [self.network.neurons[i].state.membrane_potential for i in layer_indices]
        
        # Get firing rates
        rates = self.network.get_firing_rates()[layer_indices]
        
        # Count recent spikes
        recent_spikes = 0
        for idx in layer_indices:
            spikes = self.network.spike_history[idx]
            recent = [s for s in spikes if s > self.time - 100]
            recent_spikes += len(recent)
        
        return {
            'mean_voltage': np.mean(voltages),
            'std_voltage': np.std(voltages),
            'mean_rate': np.mean(rates),
            'active_fraction': np.mean(rates > 0),
            'recent_spikes': recent_spikes,
            'activity': recent_spikes / max(1, self.layers[layer_name].size * 100) * 1000,
        }
    
    def get_column_activity(self) -> Dict[str, Dict]:
        """
        Get activity for all layers.
        
        Returns:
            Dictionary of layer activities
        """
        activities = {}
        for layer_name in self.LAYER_ORDER:
            activities[layer_name] = self.get_layer_activity(layer_name)
        return activities
    
    def get_feature_map(self, layer_name: str, bin_size: float = 10.0) -> np.ndarray:
        """
        Get spatial feature map for a layer.
        
        Args:
            layer_name: Layer name
            bin_size: Spatial bin size (µm)
            
        Returns:
            2D array representing activity map
        """
        if self.network is None:
            return np.zeros((10, 10))
        
        layer_indices = self.get_layer_neurons(layer_name)
        
        # Get positions
        positions = [self.network.neurons[i].position[:2] for i in layer_indices]
        
        # Create bins
        if len(positions) > 0:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            x_bins = int((max(x_coords) - min(x_coords)) / bin_size) + 1
            y_bins = int((max(y_coords) - min(y_coords)) / bin_size) + 1
        else:
            x_bins, y_bins = 10, 10
        
        # Create activity map
        activity_map = np.zeros((max(x_bins, 1), max(y_bins, 1)))
        
        # Get firing rates
        rates = self.network.get_firing_rates()[layer_indices]
        
        # Bin activity
        for i, idx in enumerate(layer_indices):
            x, y = positions[i]
            xi = min(int(x / bin_size), x_bins - 1)
            yi = min(int(y / bin_size), y_bins - 1)
            activity_map[xi, yi] += rates[i]
        
        return activity_map
    
    def get_statistics(self) -> Dict:
        """
        Get column statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_neurons': self.total_size,
            'time': self.time,
            'thalamic_input': self.thalamic_input,
        }
        
        # Layer statistics
        for layer_name in self.LAYER_ORDER:
            layer_stats = self.get_layer_activity(layer_name)
            for key, value in layer_stats.items():
                stats[f'{layer_name}_{key}'] = value
        
        return stats
    
    def __repr__(self) -> str:
        return (f"CorticalColumn(size={self.total_size}, "
                f"layers={[l.name for l in self.layers.values()]})")
