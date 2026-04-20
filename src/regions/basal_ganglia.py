"""
Basal Ganglia Model.

The basal ganglia are involved in motor control, habit formation,
and reinforcement learning. This module implements a simplified
basal ganglia model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_region import BrainRegion, RegionConfig


@dataclass 
class BasalGangliaConfig(RegionConfig):
    """Configuration for basal ganglia model."""
    # Population sizes
    Striatum_size: int = 200    # D1 and D2 medium spiny neurons
    GPe_size: int = 50          # Globus pallidus externus
    GPi_size: int = 30          # Globus pallidus internus / SNr
    STN_size: int = 20          # Subthalamic nucleus
    SNc_size: int = 10          # Substantia nigra pars compacta
    
    # D1/D2 ratio
    D1_ratio: float = 0.5       # Proportion of D1 (direct) neurons
    
    # Synaptic weights
    cortex_striatum: float = 1.0
    striatum_GPe: float = 1.0
    striatum_GPi: float = 1.0   # Direct pathway
    STN_GPi: float = 1.0
    GPe_STN: float = 0.5
    GPe_striatum: float = 0.3
    SNc_dopamine: float = 1.0


class BasalGanglia(BrainRegion):
    """
    Basal ganglia model.
    
    This model implements the classic basal ganglia circuitry:
    - Cortex -> Striatum (excitatory)
    - Striatum -> GPe, GPi/SNr (inhibitory)
    - STN -> GPi/SNr (excitatory)
    - GPe -> STN (inhibitory)
    - SNc -> Striatum (dopaminergic modulation)
    
    Pathways:
    - Direct (D1): Cortex -> Striatum -> GPi -> Thalamus
    - Indirect (D2): Cortex -> Striatum -> GPe -> STN -> GPi
    - Hyperdirect: Cortex -> STN -> GPi
    
    Features:
    - D1/D2 pathway segregation
    - Dopamine modulation
    - Action selection
    - Reward prediction
    
    Reference:
        Frank, M.J. (2005). Bad decisions unfold. Science, 310(5747), 439-441.
    """
    
    def __init__(self, config: Optional[BasalGangliaConfig] = None):
        """Initialize basal ganglia."""
        super().__init__(config or BasalGangliaConfig())
        
        # Input/Output
        self.add_input_port("motor_cortex", "Striatum", weight=1.0)
        self.add_input_port("premotor", "STN", weight=0.5)
        self.add_output_port("thalamus", "GPi")
        self.add_output_port("action_selection", "all")
        
        # Dopamine system
        self.dopamine_level: float = 0.5
        self.reward_prediction_error: float = 0.0
        
        # State
        self.state['D1_activity'] = np.zeros(int(self.config.Striatum_size * self.config.D1_ratio))
        self.state['D2_activity'] = np.zeros(int(self.config.Striatum_size * (1 - self.config.D1_ratio)))
        self.state['GPi_activity'] = np.zeros(self.config.GPi_size)
        self.state['action_values'] = np.zeros(5)  # Simulated action values
    
    def _initialize_components(self):
        """Initialize basal ganglia nuclei."""
        from ..models.izhikevich import IzhikevichNeuron, IzhikevichParameters, IzhikevichNeuronType
        
        config = self.config
        idx = 0
        
        # D1 neurons (direct pathway - Go)
        n_D1 = int(config.Striatum_size * config.D1_ratio)
        self.D1_indices = list(range(idx, idx + n_D1))
        idx += n_D1
        
        # D2 neurons (indirect pathway - NoGo)
        n_D2 = config.Striatum_size - n_D1
        self.D2_indices = list(range(idx, idx + n_D2))
        idx += n_D2
        
        # GPe
        self.GPe_indices = list(range(idx, idx + config.GPe_size))
        idx += config.GPe_size
        
        # GPi/SNr
        self.GPi_indices = list(range(idx, idx + config.GPi_size))
        idx += config.GPi_size
        
        # STN
        self.STN_indices = list(range(idx, idx + config.STN_size))
        idx += config.STN_size
        
        # SNc (dopamine neurons - simplified)
        self.SNc_indices = list(range(idx, idx + config.SNc_size))
        idx += config.SNc_size
        
        # Create neurons
        self.neurons = []
        
        # MSNs (medium spiny neurons)
        for i in range(config.Striatum_size):
            # D1 neurons are more excitable (lower threshold)
            is_D1 = i < n_D1
            params = IzhikevichParameters(
                a=0.02, b=0.2, c=-65.0, d=8.0,
                threshold=-50.0 if is_D1 else -55.0
            )
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # GPe neurons (fast spiking)
        for i in range(config.GPe_size):
            params = IzhikevichParameters(a=0.1, b=0.2, c=-65.0, d=2.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # GPi/SNr neurons (output)
        for i in range(config.GPi_size):
            params = IzhikevichParameters(a=0.02, b=0.2, c=-50.0, d=2.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # STN neurons
        for i in range(config.STN_size):
            params = IzhikevichParameters(a=0.005, b=0.25, c=-60.0, d=4.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # SNc neurons (dopamine)
        for i in range(config.SNc_size):
            params = IzhikevichParameters(a=0.02, b=0.2, c=-55.0, d=4.0)
            neuron = IzhikevichNeuron(params, id=len(self.neurons))
            self.neurons.append(neuron)
        
        # Create connectivity
        self._create_connectivity()
    
    def _create_connectivity(self):
        """Create basal ganglia connectivity."""
        config = self.config
        
        # Striatum -> GPe (D2-mediated)
        self.Str_to_GPe = np.random.uniform(0.5, 1.5, (config.Striatum_size, config.GPe_size))
        
        # Striatum -> GPi (D1-mediated - direct)
        self.Str_to_GPi = np.random.uniform(0.5, 1.5, (config.Striatum_size, config.GPi_size))
        
        # STN -> GPi (hyperdirect)
        self.STN_to_GPi = np.random.uniform(0.5, 1.5, (config.STN_size, config.GPi_size))
        
        # GPe -> STN (indirect)
        self.GPe_to_STN = np.random.uniform(0.3, 0.7, (config.GPe_size, config.STN_size))
        
        # GPe -> Striatum (feedback)
        self.GPe_to_Str = np.random.uniform(0.2, 0.5, (config.GPe_size, config.Striatum_size))
    
    def _compute_dynamics(self, dt: float):
        """Compute basal ganglia dynamics."""
        config = self.config
        
        # Get cortical input
        cortex_input = np.zeros(config.Striatum_size)
        if "motor_cortex" in self.input_ports:
            activity = self.input_ports["motor_cortex"].activity
            cortex_input += activity * np.random.uniform(0.5, 1.5, config.Striatum_size)
        
        # Apply dopamine modulation
        # D1 neurons: increased excitability with dopamine
        # D2 neurons: decreased excitability with dopamine
        dopamine = self.dopamine_level
        
        n_D1 = len(self.D1_indices)
        d1_input = cortex_input[:n_D1] * (1 + dopamine * 0.5)
        d2_input = cortex_input[n_D1:] * (1 - dopamine * 0.3)
        
        # Update D1 neurons (direct pathway - facilitates movement)
        for i, idx in enumerate(self.D1_indices):
            self.neurons[idx].update(dt, d1_input[i])
        
        self.state['D1_activity'] = self._get_population_activity(self.D1_indices)
        
        # Update D2 neurons (indirect pathway - inhibits movement)
        for i, idx in enumerate(self.D2_indices):
            self.neurons[idx].update(dt, d2_input[i])
        
        self.state['D2_activity'] = self._get_population_activity(self.D2_indices)
        
        # GPe receives from D2
        striatum_total = np.concatenate([self.state['D1_activity'], self.state['D2_activity']])
        gpe_input = np.dot(striatum_total, self.Str_to_GPe) * config.striatum_GPe
        
        # GPe inhibits STN
        gpe_inhibition = np.dot(
            self._get_population_activity(self.GPe_indices),
            self.GPe_to_STN
        ) * config.GPe_STN
        
        # STN receives cortical input (hyperdirect) and GPe inhibition
        stn_input = np.zeros(config.STN_size)
        if "premotor" in self.input_ports:
            stn_input += self.input_ports["premotor"].activity * np.random.uniform(0.3, 0.7, config.STN_size)
        stn_input -= gpe_inhibition
        
        for i, idx in enumerate(self.STN_indices):
            self.neurons[idx].update(dt, stn_input[i])
        
        # GPi receives from D1 (direct) and STN (hyperdirect)
        gp_i_input = np.zeros(config.GPi_size)
        
        # Direct pathway (D1 -> GPi, disinhibits thalamus)
        direct_input = np.dot(self.state['D1_activity'], 
                              self.Str_to_GPi[:len(self.D1_indices)]) * config.striatum_GPi
        gp_i_input += direct_input
        
        # Hyperdirect pathway (STN -> GPi, rapid inhibition)
        stn_activity = self._get_population_activity(self.STN_indices)
        hyperdirect_input = np.dot(stn_activity, self.STN_to_GPi) * config.STN_GPi
        gp_i_input += hyperdirect_input
        
        # GPi output (inhibits thalamus)
        for i, idx in enumerate(self.GPi_indices):
            self.neurons[idx].update(dt, gp_i_input[i])
        
        self.state['GPi_activity'] = self._get_population_activity(self.GPi_indices)
        
        # Update GPe
        for i, idx in enumerate(self.GPe_indices):
            self.neurons[idx].update(dt, gpe_input[i])
        
        # Update SNc (reward prediction)
        self._update_dopamine(dt)
    
    def _get_population_activity(self, indices: List[int]) -> np.ndarray:
        """Get activity for a population."""
        rates = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            rates[i] = self.neurons[idx].get_average_firing_rate(window_ms=100)
        return rates
    
    def _update_dopamine(self, dt: float):
        """Update dopamine levels based on reward."""
        # This would integrate reward prediction error
        # For simplicity, using a decaying dopamine level
        self.dopamine_level *= 0.999
        self.dopamine_level = max(0.1, self.dopamine_level)
    
    def _update_outputs(self):
        """Update outputs."""
        # GPi activity inversely related to thalamic disinhibition
        gpi_rate = np.mean(self.state['GPi_activity'])
        thalamic_activity = max(0, 10 - gpi_rate)  # Inverse relationship
        
        self.output_ports['thalamus'].update(thalamic_activity)
        
        # Action selection based on pathway activities
        d1_mean = np.mean(self.state['D1_activity'])
        d2_mean = np.mean(self.state['D2_activity'])
        
        # Net action selection signal
        selection = d1_mean - d2_mean * 0.5
        self.output_ports['action_selection'].update(selection)
        
        # Update action values
        self.state['action_values'] += 0.1 * (d1_mean - self.state['action_values'])
    
    def inject_reward(self, reward: float):
        """
        Inject reward signal (for learning).
        
        Args:
            reward: Reward value
        """
        self.dopamine_level = max(1.0, reward)
        self.reward_prediction_error = reward - self.dopamine_level
    
    def inject_reward_prediction_error(self, rpe: float):
        """
        Inject reward prediction error directly.
        
        Args:
            rpe: Reward prediction error
        """
        self.dopamine_level = np.clip(0.5 + rpe, 0.0, 2.0)
    
    def select_action(self) -> int:
        """
        Select action based on basal ganglia activity.
        
        Returns:
            Selected action index
        """
        # Winner-take-all based on D1 activity
        d1_activity = self.state['D1_activity']
        
        if len(d1_activity) == 0:
            return 0
        
        # Map to action space
        n_actions = 5
        bins = np.array_split(d1_activity, n_actions)
        action_scores = [np.mean(bin) for bin in bins if len(bin) > 0]
        
        return int(np.argmax(action_scores))
    
    def get_pathway_activities(self) -> Dict[str, float]:
        """Get mean activity for each pathway."""
        return {
            'D1_direct': np.mean(self.state['D1_activity']),
            'D2_indirect': np.mean(self.state['D2_activity']),
            'GPi_output': np.mean(self.state['GPi_activity']),
            'dopamine': self.dopamine_level,
        }
    
    def _record_custom_statistics(self):
        """Record basal ganglia-specific statistics."""
        self.statistics['D1_D2_balance'] = (
            np.mean(self.state['D1_activity']) - np.mean(self.state['D2_activity'])
        )
        self.statistics['dopamine'] = self.dopamine_level
        self.statistics['thalamic_activity'] = self.output_ports['thalamus'].activity
    
    def __repr__(self) -> str:
        return (f"BasalGanglia(Striatum={self.config.Striatum_size}, "
                f"GPe={self.config.GPe_size}, GPi={self.config.GPi_size})")
