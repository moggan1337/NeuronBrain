"""
Izhikevich Spiking Neuron Model.

The Izhikevich model is a mathematical model of spiking neurons that
combines the biological realism of the Hodgkin-Huxley model with the
computational efficiency of simpler models like the leaky integrate-and-fire.

The model is described by two coupled differential equations:
    dv/dt = 0.04*v² + 5*v + 140 - u + I
    du/dt = a*(b*v - u)

with reset conditions:
    if v >= 30 mV: v <- c, u <- u + d

where:
    v: membrane potential (mV)
    u: recovery variable (represents K+ current and Na+ inactivation)
    I: input current
    a, b, c, d: dimensionless parameters controlling spiking behavior

This simple model can reproduce many types of cortical neuron behavior
including regular spiking, intrinsic bursting, chattering, and fast
spiking, making it ideal for large-scale network simulations.

Reference:
    Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE
    Transactions on Neural Networks, 14(6), 1569-1572.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .base import Neuron, NeuronParameters, NeuronType, NeuronState


class IzhikevichNeuronType(Enum):
    """
    Preset neuron types from the Izhikevich model.
    
    These parameter sets reproduce the firing patterns observed in
    different cortical neuron types.
    """
    # Excitatory neurons
    RegularSpiking = "RS"      # Regular spiking (pyramidal cells)
    IntrinsicBursting = "IB"  # Intrinsic bursting
    Chattering = "CH"         # Chattering (fast burst)
    
    # Inhibitory neurons
    FastSpiking = "FS"         # Fast spiking (interneurons)
    LowThresholdSpiking = "LTS"  # Low-threshold spiking
    Resonator = "RZ"          # Resonator (thalamocortical)
    
    # Special types
    Thalamocortical = "TC"    # Thalamocortical relay
    Reticular = "RE"          # Reticular thalamic
    Mixed = "MX"              # Mixed mode


@dataclass
class IzhikevichParameters(NeuronParameters):
    """
    Parameters for the Izhikevich model.
    
    The a, b, c, d parameters control different aspects of
    neuronal dynamics:
        a: Recovery variable time constant (smaller = slower)
        b: Sensitivity of recovery variable to subthreshold oscillations
        c: After-spike reset value
        d: After-spike reset of recovery variable
    """
    # Core Izhikevich parameters
    a: float = 0.02      # Recovery time constant (ms⁻¹)
    b: float = 0.2       # Subthreshold coupling
    c: float = -65.0     # After-spike reset potential (mV)
    d: float = 8.0       # After-spike recovery increment
    
    # Threshold and reset (overrides base class)
    threshold: float = 30.0  # Spike threshold (mV)
    reset_potential: float = -65.0
    
    # Optional features
    adaptation_enabled: bool = False
    noise_std: float = 0.0  # Noise standard deviation
    
    # Coupling parameters for network simulations
    w: float = 1.0  # Synaptic weight scaling


# Preset parameter configurations for different neuron types
IZHIKEVICH_PRESETS = {
    IzhikevichNeuronType.RegularSpiking: IzhikevichParameters(
        a=0.02, b=0.2, c=-65.0, d=8.0,
        description="Regular spiking pyramidal cell"
    ),
    IzhikevichNeuronType.IntrinsicBursting: IzhikevichParameters(
        a=0.02, b=0.2, c=-55.0, d=4.0,
        description="Intrinsic bursting neuron"
    ),
    IzhikevichNeuronType.Chattering: IzhikevichParameters(
        a=0.02, b=0.2, c=-50.0, d=2.0,
        description="Chattering neuron (fast bursts)"
    ),
    IzhikevichNeuronType.FastSpiking: IzhikevichParameters(
        a=0.1, b=0.2, c=-65.0, d=2.0,
        description="Fast spiking interneuron"
    ),
    IzhikevichNeuronType.LowThresholdSpiking: IzhikevichParameters(
        a=0.02, b=0.25, c=-65.0, d=2.0,
        description="Low-threshold spiking interneuron"
    ),
    IzhikevichNeuronType.Resonator: IzhikevichParameters(
        a=0.1, b=0.25, c=-65.0, d=2.0,
        description="Resonator neuron"
    ),
    IzhikevichNeuronType.Thalamocortical: IzhikevichParameters(
        a=0.02, b=0.25, c=-65.0, d=0.05,
        description="Thalamocortical relay cell"
    ),
    IzhikevichNeuronType.Reticular: IzhikevichParameters(
        a=0.02, b=0.25, c=-65.0, d=2.0,
        description="Reticular thalamic neuron"
    ),
}


@dataclass
class IzhikevichState(NeuronState):
    """State variables for Izhikevich model."""
    u: float = -14.0  # Recovery variable


class IzhikevichNeuron(Neuron):
    """
    Izhikevich spiking neuron model.
    
    This model captures the essential dynamics of biological neurons
    while being computationally efficient enough for large-scale
    network simulations with millions of neurons.
    
    Key features:
    - Preset configurations for different cortical neuron types
    - Spike timing and rate coding
    - Bursting behavior
    - Network coupling support
    - Optional adaptation mechanisms
    
    Example:
        >>> # Create regular spiking neuron
        >>> params = IzhikevichParameters(a=0.02, b=0.2, c=-65.0, d=8.0)
        >>> neuron = IzhikevichNeuron(params)
        >>> 
        >>> # Or use preset
        >>> neuron = IzhikevichNeuron.from_preset(IzhikevichNeuronType.FastSpiking)
        >>> 
        >>> # Simulation loop
        >>> for t in range(1000):
        ...     V = neuron.update(0.1, 10.0)  # 0.1 ms dt, 10 nA input
    """
    
    def __init__(
        self,
        parameters: Optional[IzhikevichParameters] = None,
        id: Optional[int] = None,
        position: Optional[tuple] = None,
        izhikevich_type: IzhikevichNeuronType = IzhikevichNeuronType.RegularSpiking
    ):
        """
        Initialize Izhikevich neuron.
        
        Args:
            parameters: Custom Izhikevich parameters
            id: Neuron identifier
            position: 3D position in space
            izhikevich_type: Type of cortical neuron to model
        """
        if parameters is None:
            parameters = IzhikevichParameters()
            
        super().__init__(
            parameters=parameters,
            id=id,
            position=position,
            neuron_type=NeuronType.Izhikevich
        )
        self.state = IzhikevichState(
            membrane_potential=parameters.resting_potential
        )
        self.izhikevich_type = izhikevich_type
        self._initialize_recovery()
        
        # RNG for noise
        self._rng = np.random.RandomState(id)
        
    @classmethod
    def from_preset(
        cls, 
        preset: IzhikevichNeuronType,
        id: Optional[int] = None,
        position: Optional[tuple] = None
    ) -> 'IzhikevichNeuron':
        """
        Create neuron from preset configuration.
        
        Args:
            preset: Preset neuron type
            id: Neuron identifier
            position: 3D position
            
        Returns:
            Configured Izhikevich neuron
        """
        params = IZHIKEVICH_PRESETS.get(preset)
        if params is None:
            raise ValueError(f"Unknown preset: {preset}")
        return cls(params, id, position, preset)
    
    def _initialize_recovery(self):
        """Initialize recovery variable from parameters."""
        params = self.parameters
        if isinstance(params, IzhikevichParameters):
            # Initialize u at steady state: u = b*v
            self.state.u = params.b * self.state.membrane_potential
    
    def compute_dynamics(
        self, 
        dt: float, 
        state: Dict[str, float], 
        input_current: float
    ) -> Dict[str, float]:
        """
        Compute state derivatives for Izhikevich model.
        
        The dynamics are:
            dv/dt = 0.04*v² + 5*v + 140 - u + I
            du/dt = a*(b*v - u)
            
        Args:
            dt: Time step in ms
            state: Current state (v, u)
            input_current: Input current in nA
            
        Returns:
            Dictionary of derivatives
        """
        params = self.parameters
        if not isinstance(params, IzhikevichParameters):
            params = IzhikevichParameters()
            
        v = state['V']
        u = state.get('u', params.b * v)
        
        # Izhikevich dynamics with proper scaling
        # Note: coefficients are scaled for v in mV, dt in ms
        dv_dt = 0.04 * (v ** 2) + 5 * v + 140 - u + input_current
        du_dt = params.a * (params.b * v - u)
        
        return {
            'dV_dt': dv_dt,
            'd_u_dt': du_dt
        }
    
    def update(self, dt: float, input_current: float) -> float:
        """
        Update Izhikevich neuron using Euler integration.
        
        The update follows these steps:
        1. Add input current (with optional noise)
        2. Update membrane potential v
        3. Update recovery variable u
        4. Check for spike and apply reset if needed
        
        Args:
            dt: Time step in milliseconds
            input_current: Input current in nanoamperes (nA)
            
        Returns:
            Updated membrane potential in mV
        """
        params = self.parameters
        if not isinstance(params, IzhikevichParameters):
            params = IzhikevichParameters()
            
        state = self.state
        
        # Skip update during refractory period
        if state.refractory_time > 0:
            state.refractory_time -= dt
            return state.membrane_potential
        
        v = state.membrane_potential
        u = state.u
        
        # Add noise if enabled
        if params.noise_std > 0:
            input_current += self._rng.normal(0, params.noise_std)
        
        # Izhikevich dynamics
        dv = (0.04 * v * v + 5 * v + 140 - u + input_current) * dt
        du = params.a * (params.b * v - u) * dt
        
        v_new = v + dv
        u_new = u + du
        
        # Check for spike (using 30 mV threshold)
        if v_new >= params.threshold:
            self.spike(dt)
            
            # Reset dynamics
            v_new = params.c
            u_new = u_new + params.d
            
            # Apply adaptation if enabled
            if params.adaptation_enabled:
                # Additional adaptation based on spike history
                if len(state.spike_history) > 10:
                    recent_rate = len([s for s in state.spike_history[-10:] 
                                     if s >= state.spike_history[-1] - 100]) / 0.1
                    u_new += 0.01 * recent_rate
        
        state.membrane_potential = v_new
        state.u = u_new
        
        return v_new
    
    def update RK4(self, dt: float, input_current: float) -> float:
        """
        Update using 4th-order Runge-Kutta method.
        
        This provides more accurate integration at the cost of
        4x the computation compared to simple Euler.
        
        Args:
            dt: Time step in ms
            input_current: Input current in nA
            
        Returns:
            Updated membrane potential
        """
        params = self.parameters
        if not isinstance(params, IzhikevichParameters):
            params = IzhikevichParameters()
            
        state = self.state
        
        v = state.membrane_potential
        u = state.u
        
        def derivatives(v, u, I):
            dv = (0.04 * v * v + 5 * v + 140 - u + I)
            du = params.a * (params.b * v - u)
            return dv, du
        
        # RK4 integration
        k1_v, k1_u = derivatives(v, u, input_current)
        k2_v, k2_u = derivatives(v + 0.5*dt*k1_v, u + 0.5*dt*k1_u, input_current)
        k3_v, k3_u = derivatives(v + 0.5*dt*k2_v, u + 0.5*dt*k2_u, input_current)
        k4_v, k4_u = derivatives(v + dt*k3_v, u + dt*k3_u, input_current)
        
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        u_new = u + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        
        # Check for spike
        if v_new >= params.threshold:
            self.spike(dt)
            v_new = params.c
            u_new = u_new + params.d
        
        state.membrane_potential = v_new
        state.u = u_new
        
        return v_new
    
    def get_phase_plane_trajectory(
        self, 
        input_current: float,
        duration: float,
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute phase plane trajectory (v vs u).
        
        Useful for visualizing the nullclines and trajectory.
        
        Args:
            input_current: Constant input current
            duration: Simulation duration in ms
            dt: Time step
            
        Returns:
            Tuple of (time, v_history, u_history)
        """
        params = self.parameters
        if not isinstance(params, IzhikevichParameters):
            params = IzhikevichParameters()
            
        n_steps = int(duration / dt)
        time = np.zeros(n_steps)
        v_history = np.zeros(n_steps)
        u_history = np.zeros(n_steps)
        
        v = self.state.membrane_potential
        u = self.state.u
        
        for i in range(n_steps):
            time[i] = i * dt
            v_history[i] = v
            u_history[i] = u
            
            dv = (0.04 * v * v + 5 * v + 140 - u + input_current) * dt
            du = params.a * (params.b * v - u) * dt
            
            v = v + dv
            u = u + du
            
            if v >= params.threshold:
                v = params.c
                u = u + params.d
        
        return time, v_history, u_history
    
    def compute_nullclines(self, v_range: np.ndarray, input_current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute v-nullcline and u-nullcline.
        
        The nullclines show where dv/dt = 0 and du/dt = 0,
        revealing the fixed points and dynamics of the system.
        
        Args:
            v_range: Membrane potential range
            input_current: Input current
            
        Returns:
            Tuple of (v-nullcline, u-nullcline)
        """
        params = self.parameters
        if not isinstance(params, IzhikevichParameters):
            params = IzhikevichParameters()
            
        # v-nullcline: dv/dt = 0 => 0.04*v² + 5*v + 140 - u + I = 0
        # => u = 0.04*v² + 5*v + 140 + I
        u_from_v = 0.04 * v_range**2 + 5 * v_range + 140 + input_current
        
        # u-nullcline: du/dt = 0 => a*(b*v - u) = 0 => u = b*v
        u_nullcline = params.b * v_range
        
        return u_from_v, u_nullcline
    
    def get_burst_detection(self) -> Dict[str, any]:
        """
        Detect bursting behavior from spike train.
        
        Returns:
            Dictionary with burst statistics
        """
        spikes = self.state.spike_history
        if len(spikes) < 4:
            return {'is_bursting': False, 'burst_frequency': 0.0}
        
        # Compute interspike intervals
        isis = np.diff(spikes)
        
        # Burst detection: pairs of spikes with short ISI (< threshold)
        burst_threshold = 10.0  # ms
        bursts = []
        current_burst = []
        
        for i, isi in enumerate(isis):
            if isi < burst_threshold:
                current_burst.append(spikes[i])
                current_burst.append(spikes[i+1])
            else:
                if len(current_burst) >= 4:
                    bursts.append(current_burst)
                current_burst = []
        
        if len(current_burst) >= 4:
            bursts.append(current_burst)
        
        return {
            'is_bursting': len(bursts) > 0,
            'num_bursts': len(bursts),
            'burst_frequency': len(bursts) / (spikes[-1] - spikes[0]) * 1000 if len(spikes) > 1 else 0.0,
            'spikes_per_burst': np.mean([len(b)/2 for b in bursts]) if bursts else 0.0
        }
    
    def set_coupling_weight(self, w: float):
        """Set synaptic coupling weight for network simulations."""
        params = self.parameters
        if isinstance(params, IzhikevichParameters):
            params.w = w
    
    def __repr__(self) -> str:
        return (f"IzhikevichNeuron(id={self.id}, type={self.izhikevich_type.value}, "
                f"V={self.state.membrane_potential:.1f}mV, u={self.state.u:.1f})")
