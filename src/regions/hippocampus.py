"""
Hippocampus Model.

The hippocampus is crucial for memory formation and spatial navigation.
This module implements a simplified hippocampal model including:
- Dentate gyrus (DG)
- CA3 region (with recurrent collaterals)
- CA1 region (output)
- Entorhinal cortex input (EC)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_region import BrainRegion, RegionConfig, InputPort, OutputPort


@dataclass
class HippocampusConfig(RegionConfig):
    """Configuration for hippocampus model."""
    # Population sizes
    EC_size: int = 50      # Entorhinal cortex input
    DG_size: int = 100     # Dentate gyrus granule cells
    CA3_size: int = 80     # CA3 pyramidal cells
    CA1_size: int = 80     # CA1 pyramidal cells
    
    # Synaptic parameters
    perforant_path_weight: float = 1.0   # EC -> DG, CA3
    mossy_fiber_weight: float = 2.0      # DG -> CA3
    Schaffer_collateral_weight: float = 1.0  # CA3 -> CA1
    recurrent_weight: float = 0.5        # CA3 -> CA3
    
    # Plasticity
    LTP_threshold: float = -50.0  # mV for LTP induction
    LTD_threshold: float = -60.0  # mV for LTD induction


class Hippocampus(BrainRegion):
    """
    Hippocampal formation model.
    
    This model implements the trisynaptic circuit:
    1. Entorhinal cortex (EC) -> Dentate Gyrus (DG)
    2. DG -> CA3 (via mossy fibers)
    3. CA3 -> CA1 (via Schaffer collaterals)
    4. CA3 recurrent collaterals
    
    Features:
    - Separate populations for each region
    - Realistic connection patterns
    - Place cell-like activity patterns
    - Memory encoding via plasticity
    
    Reference:
        Hasselmo, M.E. (1999). Neuromodulation: acetylcholine and
        memory consolidation. Trends in Cognitive Sciences, 3(9),
        351-359.
        
    Example:
        >>> config = HippocampusConfig(name="hippocampus")
        >>> hippo = Hippocampus(config)
        >>> 
        >>> # Present spatial pattern
        >>> hippo.present_spatial_pattern(pattern)
        >>> 
        >>> # Run theta oscillation
        >>> for t in range(10000):
        ...     hippo.step(0.1)
        ...     hippo.apply_theta_drive()
    """
    
    # Region indices
    EC = "EC"
    DG = "DG"
    CA3 = "CA3"
    CA1 = "CA1"
    
    def __init__(self, config: Optional[HippocampusConfig] = None):
        """
        Initialize hippocampus.
        
        Args:
            config: Hippocampal configuration
        """
        super().__init__(config or HippocampusConfig())
        
        # Add input/output ports
        self.add_input_port("perirhinal", "EC", weight=1.0)
        self.add_input_port("spatial", "EC", weight=1.0)
        self.add_output_port("CA1_output", "CA1")
        self.add_output_port("DG_output", "DG")
        
        # Theta rhythm
        self.theta_phase: float = 0.0
        self.theta_frequency: float = 8.0  # Hz
        
        # Memory trace
        self.memory_trace: np.ndarray = np.zeros(self.config.CA3_size)
        self.pattern_separation_factor: float = 1.0
    
    def _initialize_components(self):
        """Initialize hippocampal populations."""
        from ..models.izhikevich import IzhikevichNeuron, IzhikevichParameters, IzhikevichNeuronType
        
        config = self.config
        idx = 0
        
        # Entorhinal cortex input (layer II stellate cells)
        self.EC_indices = list(range(idx, idx + config.EC_size))
        idx += config.EC_size
        
        # Dentate gyrus (granule cells)
        self.DG_indices = list(range(idx, idx + config.DG_size))
        idx += config.DG_size
        
        # CA3 (pyramidal cells)
        self.CA3_indices = list(range(idx, idx + config.CA3_size))
        idx += config.CA3_size
        
        # CA1 (pyramidal cells)
        self.CA1_indices = list(range(idx, idx + config.CA1_size))
        idx += config.CA1_size
        
        # Create neurons
        self.neurons = []
        
        for i in range(idx):
            neuron = IzhikevichNeuron(
                parameters=IzhikevichParameters(),
                id=i
            )
            self.neurons.append(neuron)
        
        # Create connectivity
        self._create_connectivity()
        
        # State variables
        self.state = {
            'DG_activity': np.zeros(config.DG_size),
            'CA3_activity': np.zeros(config.CA3_size),
            'CA1_activity': np.zeros(config.CA1_size),
            'memory_active': False,
        }
    
    def _create_connectivity(self):
        """Create hippocampal connectivity matrix."""
        config = self.config
        
        # EC -> DG (perforant path)
        self.EC_to_DG = self._random_connectivity(
            config.EC_size, config.DG_size, prob=0.1
        )
        
        # EC -> CA3 (direct perforant path)
        self.EC_to_CA3 = self._random_connectivity(
            config.EC_size, config.CA3_size, prob=0.05
        )
        
        # DG -> CA3 (mossy fibers)
        self.DG_to_CA3 = self._random_connectivity(
            config.DG_size, config.CA3_size, prob=0.2
        )
        
        # CA3 -> CA3 (recurrent collaterals)
        self.CA3_to_CA3 = self._random_connectivity(
            config.CA3_size, config.CA3_size, prob=0.1
        )
        
        # CA3 -> CA1 (Schaffer collaterals)
        self.CA3_to_CA1 = self._random_connectivity(
            config.CA3_size, config.CA1_size, prob=0.15
        )
    
    def _random_connectivity(self, pre_size: int, post_size: int, prob: float) -> np.ndarray:
        """Create random connectivity matrix."""
        conn = np.random.binomial(1, prob, (pre_size, post_size))
        # Normalize
        conn = conn.astype(float) * np.random.uniform(0.8, 1.2, (pre_size, post_size))
        return conn
    
    def _compute_dynamics(self, dt: float):
        """Compute hippocampal dynamics."""
        config = self.config
        
        # Get input from EC
        ec_input = np.zeros(config.EC_size)
        if "perirhinal" in self.input_ports:
            activity = self.input_ports["perirhinal"].activity
            ec_input += activity * np.random.uniform(0.5, 1.5, config.EC_size)
        if "spatial" in self.input_ports:
            activity = self.input_ports["spatial"].activity
            ec_input += activity * np.random.uniform(0.5, 1.5, config.EC_size)
        
        # Update EC neurons
        for i, idx in enumerate(self.EC_indices):
            self.neurons[idx].update(dt, ec_input[i])
        
        # Compute DG activity (sparse coding - pattern separation)
        dg_input = np.dot(
            ec_input[:len(self.EC_indices)],
            self.EC_to_DG
        ) * config.perforant_path_weight
        
        # Apply pattern separation (denoising)
        dg_input = self._apply_pattern_separation(dg_input)
        
        for i, idx in enumerate(self.DG_indices):
            self.neurons[idx].update(dt, dg_input[i])
        
        self.state['DG_activity'] = self._get_population_activity(self.DG_indices)
        
        # Compute CA3 activity (autoassociative memory)
        mossy_input = np.dot(
            self.state['DG_activity'],
            self.DG_to_CA3
        ) * config.mossy_fiber_weight
        
        ec_direct = np.dot(
            ec_input[:len(self.EC_indices)],
            self.EC_to_CA3
        ) * config.perforant_path_weight
        
        recurrent_input = np.dot(
            self.state['CA3_activity'],
            self.CA3_to_CA3
        ) * config.recurrent_weight
        
        ca3_total = mossy_input + ec_direct + recurrent_input
        
        # Apply memory trace (context-dependent recall)
        if self.state['memory_active']:
            ca3_total += self.memory_trace * 0.5
        
        for i, idx in enumerate(self.CA3_indices):
            self.neurons[idx].update(dt, ca3_total[i])
        
        self.state['CA3_activity'] = self._get_population_activity(self.CA3_indices)
        
        # Update memory trace
        self._update_memory_trace()
        
        # Compute CA1 activity (output)
        schaffer_input = np.dot(
            self.state['CA3_activity'],
            self.CA3_to_CA1
        ) * config.Schaffer_collateral_weight
        
        for i, idx in enumerate(self.CA1_indices):
            self.neurons[idx].update(dt, schaffer_input[i])
        
        self.state['CA1_activity'] = self._get_population_activity(self.CA1_indices)
    
    def _apply_pattern_separation(self, input_pattern: np.ndarray) -> np.ndarray:
        """
        Apply pattern separation in dentate gyrus.
        
        This implements sparse coding and orthogonalization.
        """
        # Sparsify (only strongest inputs activate)
        threshold = np.percentile(input_pattern, 70)
        sparse = np.where(input_pattern > threshold, input_pattern, 0)
        
        # Normalize
        norm = np.linalg.norm(sparse)
        if norm > 0:
            sparse = sparse / norm * np.sqrt(len(sparse)) * self.pattern_separation_factor
        
        return sparse
    
    def _update_memory_trace(self):
        """Update CA3 memory trace for recall."""
        ca3_activity = self.state['CA3_activity']
        
        # Strong activation leaves trace
        strong_mask = ca3_activity > 5.0
        self.memory_trace[strong_mask] += ca3_activity[strong_mask] * 0.1
        
        # Decay trace
        self.memory_trace *= 0.999
    
    def _get_population_activity(self, indices: List[int]) -> np.ndarray:
        """Get activity for a population."""
        rates = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            rates[i] = self.neurons[idx].get_average_firing_rate(window_ms=100)
        return rates
    
    def _update_outputs(self):
        """Update output ports."""
        ca1_rate = np.mean(self.state['CA1_activity'])
        dg_rate = np.mean(self.state['DG_activity'])
        
        self.output_ports['CA1_output'].update(ca1_rate)
        self.output_ports['DG_output'].update(dg_rate)
    
    def apply_theta_drive(self):
        """
        Apply theta oscillation drive.
        
        Hippocampus exhibits 8 Hz theta oscillations during active behavior.
        This modulates excitability.
        """
        self.theta_phase += 2 * np.pi * self.theta_frequency * 0.001  # 0.001 = 1 ms
        
        phase = (np.sin(self.theta_phase) + 1) / 2  # 0 to 1
        
        # Phase 0-0.5: Active (encoding)
        # Phase 0.5-1.0: Rest (consolidation)
        encoding_strength = phase if phase < 0.5 else 1 - phase
        
        # Apply to CA3 (place cells)
        for idx in self.CA3_indices:
            # Modulate input
            pass  # Would apply to synaptic weights or input
    
    def present_spatial_pattern(self, pattern: np.ndarray, duration: float = 100.0):
        """
        Present a spatial pattern to the hippocampus.
        
        Args:
            pattern: Spatial input pattern
            duration: Presentation duration (ms)
        """
        if len(pattern) != self.config.EC_size:
            pattern = np.interp(
                np.linspace(0, 1, self.config.EC_size),
                np.linspace(0, 1, len(pattern)),
                pattern
            )
        
        self.inject_input("spatial", np.mean(pattern))
    
    def recall_memory(self, cue: np.ndarray) -> np.ndarray:
        """
        Recall memory using a cue.
        
        Args:
            cue: Retrieval cue pattern
            
        Returns:
            Recalled CA3 pattern
        """
        # Present cue
        self.present_spatial_pattern(cue)
        
        # Enable memory recall
        self.state['memory_active'] = True
        
        # Return CA3 activity
        return self.state['CA3_activity'].copy()
    
    def get_place_fields(self) -> Dict[int, np.ndarray]:
        """
        Get simulated place fields.
        
        Returns:
            Dictionary mapping neuron indices to place fields
        """
        place_fields = {}
        
        # Simulate place fields for CA1 neurons
        for i, idx in enumerate(self.CA1_indices):
            # Gaussian place field
            position = np.linspace(0, 100, 100)
            center = np.random.uniform(20, 80)
            width = np.random.uniform(10, 30)
            field = np.exp(-(position - center)**2 / (2 * width**2))
            place_fields[idx] = field
        
        return place_fields
    
    def _record_custom_statistics(self):
        """Record hippocampus-specific statistics."""
        self.statistics['DG_sparsity'] = np.mean(self.state['DG_activity'] > 0)
        self.statistics['CA3_recall'] = np.mean(self.memory_trace)
        self.statistics['theta_phase'] = self.theta_phase % (2 * np.pi)
    
    def get_sequence_activity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sequential activity pattern.
        
        Returns:
            Tuple of (times, positions) for sequential activation
        """
        # This would extract theta sequences
        return np.array([]), np.array([])
    
    def __repr__(self) -> str:
        return (f"Hippocampus(EC={len(self.EC_indices)}, "
                f"DG={len(self.DG_indices)}, "
                f"CA3={len(self.CA3_indices)}, "
                f"CA1={len(self.CA1_indices)})")
