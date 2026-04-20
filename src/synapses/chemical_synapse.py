"""
Chemical Synapse Model with Dynamic Properties.

Chemical synapses are the predominant type of synapse in the nervous system.
They transmit signals via neurotransmitter release and receptor activation.
This module implements conductance-based synapse models with short-term
plasticity (facilitation and depression) and various receptor types.

The synapse model follows:
    I_syn = g_syn * (V - E_syn) * s
    
where s is the synaptic conductance that depends on presynaptic activity
and receptor dynamics.

Reference:
    Destexhe, A., Mainen, Z.F., and Sejnowski, T.J. (1994). Synthesis
    of models for excitable membranes, synaptic transmission and
    neuromodulation using a common kinetic formalism. Journal of
    Computational Neuroscience, 1(3), 195-230.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SynapticReceptor(Enum):
    """
    Types of synaptic receptors with different kinetics.
    
    Different receptor types have different time courses and
    reversal potentials.
    """
    AMPA = "ampa"      # Fast excitatory (α-amino-3-hydroxyl-5-methyl-4-isoxazolepropionic acid)
    NMDA = "nmda"      # Slow excitatory with Mg²⁺ block (N-methyl-D-aspartate)
    GABA_A = "gaba_a"  # Fast inhibitory (γ-aminobutyric acid type A)
    GABA_B = "gaba_b"  # Slow inhibitory (G-protein coupled)
    Glycine = "gly"    # Inhibitory (glycine)
    mGluR = "mglu"     # Metabotropic (modulatory)


# Receptor-specific parameters
RECEPTOR_PARAMS = {
    SynapticReceptor.AMPA: {
        'tau_rec': 0.1,      # Recovery from depression (s)
        'tau_fac': 0.01,     # Recovery from facilitation (s)
        'U': 0.6,            # Baseline release probability
        'E': 0.0,            # Reversal potential (mV)
        'g_max': 1.0,        # Maximum conductance (nS)
        'delay': 0.5,        # Synaptic delay (ms)
    },
    SynapticReceptor.NMDA: {
        'tau_rec': 0.2,
        'tau_fac': 0.01,
        'U': 0.5,
        'E': 0.0,
        'g_max': 1.0,
        'delay': 1.0,
        'Mg_conc': 1.0,      # Extracellular Mg²⁺ concentration (mM)
    },
    SynapticReceptor.GABA_A: {
        'tau_rec': 0.1,
        'tau_fac': 0.01,
        'U': 0.25,
        'E': -70.0,          # Cl⁻ reversal potential
        'g_max': 1.0,
        'delay': 0.5,
    },
    SynapticReceptor.GABA_B: {
        'tau_rec': 0.5,
        'tau_fac': 0.05,
        'U': 0.2,
        'E': -90.0,          # K⁺ reversal potential
        'g_max': 0.5,
        'delay': 3.0,
        'Gprotein': True,    # G-protein coupled receptor
    },
}


@dataclass
class SynapticParameters:
    """
    Parameters for synapse dynamics.
    
    These parameters control the temporal dynamics of synaptic
    transmission including rise time, decay, and plasticity.
    """
    # Time constants
    tau_rise: float = 0.5    # Rise time constant (ms)
    tau_decay: float = 5.0   # Decay time constant (ms)
    
    # Synaptic strength
    g_max: float = 1.0      # Maximum conductance (nS)
    E_syn: float = 0.0      # Reversal potential (mV)
    
    # Short-term plasticity
    tau_facilitation: float = 0.01  # Facilitation time constant (s)
    tau_depression: float = 0.1     # Depression time constant (s)
    U_base: float = 0.6             # Baseline release probability
    
    # Synaptic delay
    delay: float = 0.5       # Transmission delay (ms)
    
    # Connectivity
    pre_neuron_id: Optional[int] = None
    post_neuron_id: Optional[int] = None
    
    # Receptor type
    receptor: SynapticReceptor = SynapticReceptor.AMPA


class SynapticState:
    """
    State variables for synapse dynamics.
    
    Tracks the current state of the synapse including conductance,
    release probability, and available vesicles.
    """
    def __init__(self):
        self.conductance: float = 0.0     # Current conductance
        self.rise_factor: float = 0.0     # Intermediate for double-exponential
        self.release_prob: float = 0.0    # Current release probability
        self.vesicles: float = 1.0       # Available vesicles (0-1)
        self.last_spike_time: float = -1000.0  # Last spike time
        self.pending_spike: bool = False  # Spike waiting in delay queue
        
        # STDP state
        self.trace: float = 0.0          # Pre-synaptic trace for STDP
        self.last_activation: float = -1000.0
        
        # Receptor-specific states
        self.NMDA_Mg_block: float = 1.0  # Mg²⁺ block factor
        
    def reset(self):
        """Reset state to initial conditions."""
        self.conductance = 0.0
        self.rise_factor = 0.0
        self.release_prob = 0.0
        self.vesicles = 1.0
        self.last_spike_time = -1000.0
        self.pending_spike = False
        self.trace = 0.0
        self.last_activation = -1000.0
        self.NMDA_Mg_block = 1.0


class ChemicalSynapse:
    """
    Conductance-based chemical synapse model.
    
    This synapse model includes:
    - Double-exponential synaptic conductance
    - Short-term plasticity (facilitation and depression)
    - Receptor-specific kinetics
    - Synaptic delay
    - NMDA Mg²⁺ voltage-dependence
    
    The synaptic current is computed as:
        I_syn = g(t) * (V_post - E_syn) * s(t)
        
    where s(t) represents the fraction of open channels.
    
    Example:
        >>> synapse = ChemicalSynapse(
        ...     receptor=SynapticReceptor.AMPA,
        ...     g_max=2.0,
        ...     pre_id=1,
        ...     post_id=2
        ... )
        >>> 
        >>> # In simulation loop:
        >>> # 1. Notify synapse of presynaptic spike
        >>> synapse.presynaptic_spike(current_time)
        >>> # 2. Update synapse state
        >>> synapse.update(dt, post_membrane_voltage)
        >>> # 3. Get synaptic current
        >>> current = synapse.get_current()
    """
    
    def __init__(
        self,
        parameters: Optional[SynapticParameters] = None,
        receptor: SynapticReceptor = SynapticReceptor.AMPA,
        pre_id: Optional[int] = None,
        post_id: Optional[int] = None,
        weight: float = 1.0,
        plasticity_enabled: bool = True
    ):
        """
        Initialize chemical synapse.
        
        Args:
            parameters: Custom synapse parameters
            receptor: Type of synaptic receptor
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            weight: Synaptic weight multiplier
            plasticity_enabled: Enable short-term plasticity
        """
        if parameters is None:
            parameters = SynapticParameters(
                receptor=receptor,
                pre_neuron_id=pre_id,
                post_neuron_id=post_id
            )
            # Set receptor-specific defaults
            if receptor in RECEPTOR_PARAMS:
                rp = RECEPTOR_PARAMS[receptor]
                parameters.g_max = rp.get('g_max', parameters.g_max)
                parameters.E_syn = rp.get('E', parameters.E_syn)
                parameters.delay = rp.get('delay', parameters.delay)
                parameters.U_base = rp.get('U', parameters.U_base)
                parameters.tau_depression = rp.get('tau_rec', parameters.tau_depression)
                parameters.tau_facilitation = rp.get('tau_fac', parameters.tau_facilitation)
        
        self.parameters = parameters
        self.receptor = receptor
        self.pre_id = pre_id or parameters.pre_neuron_id
        self.post_id = post_id or parameters.post_neuron_id
        self.weight = weight
        self.plasticity_enabled = plasticity_enabled
        self.state = SynapticState()
        
        # Synaptic delay buffer
        self._delay_buffer: list = []
        self._delay_steps: int = max(1, int(parameters.delay / 0.1))  # Assume 0.1ms base dt
        
        # Statistics
        self.num_transmissions: int = 0
        self.total_releases: int = 0
        
    def presynaptic_spike(self, time: float):
        """
        Notify synapse of presynaptic spike.
        
        This schedules a synaptic event after the transmission delay.
        
        Args:
            time: Current simulation time (ms)
        """
        self.state.last_spike_time = time
        self._delay_buffer.append(time)
        
    def update(self, dt: float, post_voltage: float, current_time: float) -> float:
        """
        Update synapse state and compute synaptic current.
        
        Args:
            dt: Time step in ms
            post_voltage: Postsynaptic membrane potential (mV)
            current_time: Current simulation time (ms)
            
        Returns:
            Synaptic current in nA
        """
        params = self.parameters
        state = self.state
        
        # Process delayed spikes
        while self._delay_buffer and self._delay_buffer[0] + params.delay <= current_time:
            spike_time = self._delay_buffer.pop(0)
            self._trigger_release(spike_time)
        
        # Update short-term plasticity
        if self.plasticity_enabled:
            self._update_plasticity(dt)
        
        # Update conductance dynamics (double-exponential)
        # ds/dt = -s/tau_decay + rise_factor
        # drise/dt = -rise_factor/tau_rise + weight * delta(t - spike)
        if params.tau_rise > 0:
            decay_rate = 1.0 / params.tau_decay
            rise_rate = 1.0 / params.tau_rise
            
            # Update rise and decay factors
            state.rise_factor *= np.exp(-dt * rise_rate)
            state.conductance = state.conductance * np.exp(-dt * decay_rate) + state.rise_factor * dt
            
        else:
            # Single exponential
            state.conductance *= np.exp(-dt / params.tau_decay)
        
        # Update NMDA Mg²⁺ block
        if self.receptor == SynapticReceptor.NMDA:
            Mg = RECEPTOR_PARAMS[SynapticReceptor.NMDA].get('Mg_conc', 1.0)
            state.NMDA_Mg_block = 1.0 / (1.0 + Mg * np.exp(-0.062 * post_voltage) / 3.57)
        
        # Compute synaptic current
        g_syn = state.conductance * params.g_max * self.weight
        current = g_syn * (post_voltage - params.E_syn)
        
        # Apply NMDA Mg²⁺ block
        if self.receptor == SynapticReceptor.NMDA:
            current *= state.NMDA_Mg_block
        
        # Update traces for STDP
        state.trace *= np.exp(-dt / 20.0)  # Trace decay
        state.last_activation = current_time
        
        return current
    
    def _trigger_release(self, spike_time: float):
        """
        Trigger neurotransmitter release.
        
        Computes release probability based on facilitation/depression
        and updates vesicle pool.
        
        Args:
            spike_time: Time of presynaptic spike
        """
        params = self.parameters
        state = self.state
        
        # Compute release probability with facilitation
        if params.tau_facilitation > 0:
            # Facilitation: U increases with each spike
            u_eff = params.U_base + (1 - params.U_base) * state.release_prob
        else:
            u_eff = params.U_base
        
        # Compute release (vesicle depletion)
        release_amount = u_eff * state.vesicles
        release_amount = min(release_amount, 1.0)  # Clamp to 1
        
        # Trigger release
        if release_amount > 0.01:  # Only if significant
            # Add to rise factor (drives conductance)
            state.rise_factor += release_amount * params.g_max * self.weight
            self.num_transmissions += 1
        
        # Update vesicle pool (depression)
        state.vesicles -= release_amount
        state.vesicles = max(state.vesicles, 0.0)  # Can't go negative
        
        # Update release probability for facilitation
        state.release_prob = state.release_prob * np.exp(-(spike_time - state.last_spike_time) / params.tau_depression) + release_amount
        
        self.total_releases += 1
    
    def _update_plasticity(self, dt: float):
        """Update short-term plasticity variables."""
        params = self.parameters
        state = self.state
        
        # Vesicle recovery (depression recovery)
        if params.tau_depression > 0:
            state.vesicles += (1.0 - state.vesicles) * dt / params.tau_depression
        
        # Facilitation decay
        if params.tau_facilitation > 0 and state.release_prob > params.U_base:
            state.release_prob += (params.U_base - state.release_prob) * dt / params.tau_facilitation
    
    def get_conductance(self) -> float:
        """Get current synaptic conductance."""
        return self.state.conductance * self.parameters.g_max * self.weight
    
    def get_current(self, post_voltage: float) -> float:
        """
        Get synaptic current at current state.
        
        Args:
            post_voltage: Postsynaptic membrane potential (mV)
            
        Returns:
            Synaptic current (nA)
        """
        g = self.get_conductance()
        E = self.parameters.E_syn
        
        current = g * (post_voltage - E)
        
        # Apply NMDA Mg²⁺ block if applicable
        if self.receptor == SynapticReceptor.NMDA:
            Mg = RECEPTOR_PARAMS[SynapticReceptor.NMDA].get('Mg_conc', 1.0)
            block = 1.0 / (1.0 + Mg * np.exp(-0.062 * post_voltage) / 3.57)
            current *= block
        
        return current
    
    def apply_stdp_update(
        self, 
        delta_t: float, 
        learning_rate: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012
    ):
        """
        Apply STDP weight update.
        
        Args:
            delta_t: Time difference (t_post - t_pre) in ms
            learning_rate: Learning rate
            tau_plus: Potentiation time constant
            tau_minus: Depression time constant
            A_plus: Potentiation amplitude
            A_minus: Depression amplitude
        """
        if delta_t > 0:
            # Post before pre: depression (if this is the postsynaptic neuron)
            dw = -A_minus * np.exp(-delta_t / tau_minus)
        else:
            # Pre before post: potentiation
            dw = A_plus * np.exp(delta_t / tau_plus)
        
        self.weight += learning_rate * dw
        self.weight = np.clip(self.weight, 0.01, 10.0)  # Bound weights
    
    def reset(self):
        """Reset synapse to initial state."""
        self.state.reset()
        self._delay_buffer.clear()
        self.num_transmissions = 0
        self.total_releases = 0
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get synapse statistics."""
        return {
            'weight': self.weight,
            'num_transmissions': self.num_transmissions,
            'release_rate': self.num_transmissions / max(1, self.total_releases),
            'current_conductance': self.get_conductance(),
            'vesicles_available': self.state.vesicles,
            'release_probability': self.state.release_prob,
        }
    
    def __repr__(self) -> str:
        return (f"ChemicalSynapse(pre={self.pre_id}, post={self.post_id}, "
                f"receptor={self.receptor.value}, weight={self.weight:.2f})")
