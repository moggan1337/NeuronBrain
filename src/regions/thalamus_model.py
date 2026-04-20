"""
Thalamus Model.

The thalamus serves as a relay station for sensory information.
This module implements a simplified thalamic model including:
- Thalamocortical relay cells (TC)
- Reticular nucleus (RE) cells
- Sensory-specific nuclei (LGN, MGN, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_region import BrainRegion, RegionConfig


@dataclass
class ThalamusConfig(RegionConfig):
    """Configuration for thalamus model."""
    # Nuclei sizes
    LGN_size: int = 50     # Lateral geniculate nucleus (vision)
    MGN_size: int = 30     # Medial geniculate nucleus (audition)
    VL_size: int = 40     # Ventral lateral nucleus (motor)
    RE_size: int = 50     # Reticular nucleus
    
    # Synaptic parameters
    TC_to_TC_weight: float = 0.2   # Thalamocortical connections
    RE_to_TC_weight: float = 1.0   # Reticular inhibition
    TC_to_RE_weight: float = 0.5  # TC -> RE excitation
    RE_to_RE_weight: float = 0.3  # Reticular lateral inhibition
    
    # Burst parameters
    burst_threshold: float = -65.0  # mV for burst initiation
    T_current_conductance: float = 1.0


class Thalamus(BrainRegion):
    """
    Thalamic relay model.
    
    The thalamus contains distinct nuclei that relay sensory
    information to cortex. This model includes:
    - Relay nuclei (LGN, MGN, VL)
    - Reticular nucleus (inhibitory)
    - Burst and tonic firing modes
    - Gating mechanisms
    
    Features:
    - Nucleus-specific processing
    - TC-RE feedback loops
    - Burst generation (T-type Ca2+ channels)
    - Attention/gating modulation
    
    Reference:
        McCormick, D.A. and Huguenard, J. (1992). A model of the
        electrophysiological properties of thalamocortical relay
        neurons. Journal of Neurophysiology, 68(4), 1384-1400.
    """
    
    def __init__(self, config: Optional[ThalamusConfig] = None):
        """Initialize thalamus."""
        super().__init__(config or ThalamusConfig())
        
        # Add input/output ports
        self.add_input_port("sensory", "LGN", weight=1.0)
        self.add_input_port("reticular", "RE", weight=1.0)
        self.add_output_port("cortical", "all")
        self.add_output_port("burst_mode", "TC")
        
        # State
        self.state['mode'] = 'tonic'  # or 'burst'
        self.state['attention_gate'] = 1.0
        self.state['LGN_activity'] = np.zeros(self.config.LGN_size)
        self.state['RE_activity'] = np.zeros(self.config.RE_size)
    
    def _initialize_components(self):
        """Initialize thalamic nuclei."""
        from ..models.izhikevich import IzhikevichNeuron, IzhikevichParameters, IzhikevichNeuronType
        
        config = self.config
        idx = 0
        
        # LGN (vision)
        self.LGN_indices = list(range(idx, idx + config.LGN_size))
        idx += config.LGN_size
        
        # MGN (audition)
        self.MGN_indices = list(range(idx, idx + config.MGN_size))
        idx += config.MGN_size
        
        # VL (motor)
        self.VL_indices = list(range(idx, idx + config.VL_size))
        idx += config.VL_size
        
        # RE (reticular)
        self.RE_indices = list(range(idx, idx + config.RE_size))
        idx += config.RE_size
        
        # Create neurons
        self.neurons = []
        
        # TC neurons (relay cells)
        for i in range(idx - config.RE_size):
            # Use regular spiking with adaptation
            params = IzhikevichParameters(a=0.02, b=0.2, c=-65.0, d=8.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # RE neurons (thalamic reticular)
        for i in range(config.RE_size):
            # Fast spiking for RE
            params = IzhikevichParameters(a=0.1, b=0.2, c=-65.0, d=2.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # Create connectivity
        self._create_connectivity()
    
    def _create_connectivity(self):
        """Create thalamic connectivity."""
        config = self.config
        
        # RE to TC (inhibitory)
        self.RE_to_TC = np.random.uniform(
            0.5, 1.5, (config.RE_size, idx := config.LGN_size + config.MGN_size + config.VL_size)
        ) * (-config.RE_to_TC_weight)
        
        # TC to RE (excitatory)
        self.TC_to_RE = np.random.uniform(
            0.5, 1.5, (idx, config.RE_size)
        ) * config.TC_to_RE_weight
        
        # RE to RE (lateral inhibition)
        self.RE_to_RE = np.random.binomial(
            1, 0.1, (config.RE_size, config.RE_size)
        ).astype(float) * (-config.RE_to_RE_weight)
        
        # Within-nucleus connections
        self._create_intranuclear_connections()
    
    def _create_intranuclear_connections(self):
        """Create within-nucleus connections."""
        # This would create sparse connections within each nucleus
        pass
    
    def _compute_dynamics(self, dt: float):
        """Compute thalamic dynamics."""
        config = self.config
        
        # Get sensory input
        sensory_input = np.zeros(config.LGN_size)
        if "sensory" in self.input_ports:
            activity = self.input_ports["sensory"].activity
            sensory_input += activity * np.random.uniform(0.5, 1.5, config.LGN_size)
        
        # Apply attention gating
        sensory_input *= self.state['attention_gate']
        
        # Update LGN
        lgn_input = sensory_input
        for i, idx in enumerate(self.LGN_indices):
            self.neurons[idx].update(dt, lgn_input[i])
        
        self.state['LGN_activity'] = self._get_population_activity(self.LGN_indices)
        
        # Update MGN and VL similarly
        for i, idx in enumerate(self.MGN_indices):
            self.neurons[idx].update(dt, 0.0)  # No direct input
        
        for i, idx in enumerate(self.VL_indices):
            self.neurons[idx].update(dt, 0.0)  # Motor input
        
        # Update RE neurons
        # RE receives input from TC and has lateral inhibition
        all_tc_activity = np.concatenate([
            self.state['LGN_activity'],
            self._get_population_activity(self.MGN_indices),
            self._get_population_activity(self.VL_indices)
        ])
        
        re_input = np.dot(all_tc_activity, self.TC_to_RE[:len(all_tc_activity)])
        re_lateral = np.dot(
            self._get_population_activity(self.RE_indices),
            self.RE_to_RE
        )
        
        for i, idx in enumerate(self.RE_indices):
            self.neurons[idx].update(dt, re_input[i] + re_lateral[i])
        
        self.state['RE_activity'] = self._get_population_activity(self.RE_indices)
        
        # Apply RE inhibition to TC neurons
        re_inhibition = np.dot(
            self.state['RE_activity'],
            self.RE_to_TC[:, :len(self.state['LGN_activity'])]
        )
        
        for i, idx in enumerate(self.LGN_indices):
            # Apply inhibition
            total_input = lgn_input[i] + re_inhibition[i]
            self.neurons[idx].update(dt, total_input)
    
    def _get_population_activity(self, indices: List[int]) -> np.ndarray:
        """Get activity for a population."""
        rates = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            rates[i] = self.neurons[idx].get_average_firing_rate(window_ms=100)
        return rates
    
    def _update_outputs(self):
        """Update output activities."""
        lgn_rate = np.mean(self.state['LGN_activity'])
        self.output_ports['cortical'].update(lgn_rate)
        self.output_ports['burst_mode'].update(
            1.0 if self.state['mode'] == 'burst' else 0.0
        )
    
    def set_attention(self, gate: float):
        """
        Set attention gate (0-1).
        
        Args:
            gate: Attention strength
        """
        self.state['attention_gate'] = np.clip(gate, 0.0, 1.0)
    
    def switch_mode(self, mode: str):
        """
        Switch between burst and tonic mode.
        
        Args:
            mode: 'burst' or 'tonic'
        """
        if mode in ['burst', 'tonic']:
            self.state['mode'] = mode
    
    def inject_sensory_input(self, nucleus: str, pattern: np.ndarray):
        """
        Inject sensory input to a nucleus.
        
        Args:
            nucleus: Nucleus name ('LGN', 'MGN', 'VL')
            pattern: Input pattern
        """
        if nucleus == 'LGN':
            self.inject_input('sensory', np.mean(pattern))
        elif nucleus == 'MGN':
            # Would inject to MGN neurons
            pass
    
    def get_receptive_fields(self) -> Dict[str, np.ndarray]:
        """
        Get simulated receptive fields.
        
        Returns:
            Dictionary of receptive field patterns
        """
        receptive_fields = {}
        
        # Simple Gabor-like receptive fields for LGN
        for i, idx in enumerate(self.LGN_indices[:10]):
            # 2D Gabor function (simplified)
            x = np.linspace(-1, 1, 20)
            y = np.linspace(-1, 1, 20)
            X, Y = np.meshgrid(x, y)
            
            # On-center or off-center
            if i % 2 == 0:
                rf = np.exp(-(X**2 + Y**2)) - 0.5 * np.exp(-((X-0.3)**2 + (Y-0.3)**2))
            else:
                rf = -np.exp(-(X**2 + Y**2)) + 0.5 * np.exp(-((X-0.3)**2 + (Y-0.3)**2))
            
            receptive_fields[idx] = rf
        
        return receptive_fields
    
    def _record_custom_statistics(self):
        """Record thalamus-specific statistics."""
        self.statistics['LGN_synchrony'] = self._compute_synchrony(self.state['LGN_activity'])
        self.statistics['RE_inhibition'] = np.mean(np.abs(self.RE_to_TC)) * np.mean(self.state['RE_activity'])
        self.statistics['attention'] = self.state['attention_gate']
    
    def _compute_synchrony(self, activity: np.ndarray) -> float:
        """Compute population synchrony."""
        if len(activity) < 2:
            return 0.0
        # Coefficient of variation of spike times
        return 1.0 - np.std(activity) / (np.mean(activity) + 1e-10)
    
    def __repr__(self) -> str:
        return (f"Thalamus(LGN={len(self.LGN_indices)}, "
                f"MGN={len(self.MGN_indices)}, "
                f"VL={len(self.VL_indices)}, "
                f"RE={len(self.RE_indices)})")
