"""
Leaky Integrate-and-Fire (LIF) Neuron Model.

The LIF model is a simplified neuron model that captures the essential
dynamics of neuronal integration and spiking while being computationally
efficient. It treats the neuron as a leaky capacitor that integrates
input current until reaching a threshold, at which point it fires a
spike and resets.

The LIF model is governed by:
    τ_m * dV/dt = -(V - V_rest) + R_m * I_ext

where:
    τ_m = R_m * C is the membrane time constant
    V_rest is the resting potential
    R_m is the membrane resistance
    C is the membrane capacitance
    I_ext is the external input current

This model is widely used in large-scale neural network simulations
due to its computational efficiency while still capturing key
neurodynamical properties.

Reference:
    Lapicque, L. (1907). Recherches quantitatives sur l'excitation
    electrique des nerfs traitee comme une polarization. Journal de
    Physiologie et de Pathologie Generale, 9, 620-635.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from .base import Neuron, NeuronParameters, NeuronType, NeuronState


@dataclass
class LIFParameters(NeuronParameters):
    """
    Parameters for the Leaky Integrate-and-Fire model.
    
    These parameters define the biophysical properties that control
    the neuron's integration dynamics and spiking behavior.
    """
    # Membrane properties
    tau_mem: float = 20.0  # Membrane time constant (ms)
    r_mem: float = 10.0    # Membrane resistance (MΩ)
    
    # Reset dynamics
    reset_potential: float = -70.0  # mV (typically same as resting)
    reset_time: float = 1.0  # ms (additional reset time constant)
    
    # Adaptation (optional)
    adaptation_enabled: bool = False
    tau_adapt: float = 100.0  # Adaptation time constant (ms)
    adapt_coefficient: float = 0.01  # Adaptation increment per spike
    
    # Subthreshold dynamics
    subthresh_oscillations: bool = False
    osc_frequency: float = 10.0  # Hz (for subthreshold oscillations)
    osc_amplitude: float = 1.0  # mV


@dataclass
class LIFState(NeuronState):
    """State variables for LIF model."""
    adaptation: float = 0.0  # Spike frequency adaptation current
    reset_timer: float = 0.0  # Timer for reset dynamics


class LeakyIntegrateAndFire(Neuron):
    """
    Leaky Integrate-and-Fire neuron model.
    
    The LIF model is one of the most widely used neuron models in
    computational neuroscience and neural network research. It provides
    a good balance between biological realism and computational
    efficiency.
    
    Key features:
    - Exponential integrate-and-fire variant available
    - Spike frequency adaptation support
    - Subthreshold oscillation capability
    - Configurable reset dynamics
    
    Example:
        >>> params = LIFParameters(tau_mem=20.0, threshold=-55.0)
        >>> neuron = LeakyIntegrateAndFire(params)
        >>> 
        >>> # Simulate with constant current injection
        >>> for t in range(1000):
        ...     V = neuron.update(0.1, 5.0)  # 5 nA input
        ...     if V >= neuron.parameters.threshold:
        ...         print(f"Spike at t={t}")
    """
    
    def __init__(
        self,
        parameters: Optional[LIFParameters] = None,
        id: Optional[int] = None,
        position: Optional[tuple] = None,
        exponential: bool = False
    ):
        """
        Initialize LIF neuron.
        
        Args:
            parameters: LIF-specific parameters
            id: Neuron identifier
            position: 3D position in space
            exponential: Use exponential integrate-and-fire variant
        """
        super().__init__(
            parameters=parameters or LIFParameters(),
            id=id,
            position=position,
            neuron_type=NeuronType.LeakyIntegrateAndFire
        )
        self.state = LIFState(
            membrane_potential=self.parameters.resting_potential
        )
        self.exponential = exponential
        self._delta_T = 2.0  # Spike slope factor for exponential LIF
        
    def compute_dynamics(
        self, 
        dt: float, 
        state: Dict[str, float], 
        input_current: float
    ) -> Dict[str, float]:
        """
        Compute state derivatives for LIF model.
        
        Args:
            dt: Time step in ms
            state: Current state (V, adaptation, reset_timer)
            input_current: Input current (can be in nA or µA depending on R)
            
        Returns:
            Dictionary of derivatives
        """
        params = self.parameters
        V = state['V']
        adapt = state.get('adaptation', 0.0)
        
        # Convert current to voltage units (assuming R_mem in MΩ, current in nA)
        # Or if current is already in mV (after multiplying by R)
        if isinstance(input_current, (int, float)):
            # Assume input_current is already in mV equivalent
            I_input = input_current
        else:
            I_input = float(input_current)
        
        # Adaptation current dynamics
        if params.adaptation_enabled:
            d_adapt_dt = -adapt / params.tau_adapt
        else:
            d_adapt_dt = 0.0
        
        # Subthreshold oscillations
        if params.subthresh_oscillations:
            osc_term = params.osc_amplitude * np.sin(
                2 * np.pi * params.osc_frequency * dt / 1000.0
            )
        else:
            osc_term = 0.0
        
        # Membrane potential dynamics: τ_m * dV/dt = -(V - V_rest) + R*I
        # Using V_rest = resting_potential
        dV_dt = (-(V - params.resting_potential) + I_input - adapt) / params.tau_mem
        
        return {
            'dV_dt': dV_dt,
            'd_adapt_dt': d_adapt_dt,
            'oscillation': osc_term
        }
    
    def update(self, dt: float, input_current: float) -> float:
        """
        Update LIF neuron state.
        
        Uses exact solution of the LIF differential equation for
        numerical stability.
        
        Args:
            dt: Time step in milliseconds
            input_current: Input current (treated as voltage equivalent: R*I)
            
        Returns:
            Updated membrane potential in mV
        """
        params = self.parameters
        state = self.state
        
        # Handle reset timer
        if state.reset_timer > 0:
            state.reset_timer -= dt
            return params.reset_potential
        
        # Skip update during refractory period
        if state.refractory_time > 0:
            state.refractory_time -= dt
            return state.membrane_potential
        
        V = state.membrane_potential
        adapt = state.adaptation
        
        # Exponential LIF nonlinearity
        if self.exponential:
            exponential_term = self._delta_T * np.exp((V - params.threshold) / self._delta_T)
        else:
            exponential_term = 0.0
        
        # Membrane time constant
        tau = params.tau_mem
        
        # Input current (treated as voltage drive)
        I_eff = input_current - adapt + exponential_term
        
        # Analytical solution for leaky integrator
        V_inf = params.resting_potential + I_eff
        
        # Exponential decay toward steady state
        V_new = V_inf + (V - V_inf) * np.exp(-dt / tau)
        
        # Add subthreshold oscillations
        if params.subthresh_oscillations:
            V_new += params.osc_amplitude * np.sin(
                2 * np.pi * params.osc_frequency * dt / 1000.0
            ) * (1 - np.exp(-dt / tau))
        
        # Check for spike
        if V_new >= params.threshold:
            self.spike(dt)
            V_new = params.reset_potential
            
            # Apply spike frequency adaptation
            if params.adaptation_enabled:
                state.adaptation += params.adapt_coefficient * params.threshold
                state.adaptation = min(state.adaptation, 20.0)  # Cap adaptation
            
            # Reset dynamics
            state.reset_timer = params.reset_time
        
        state.membrane_potential = V_new
        return V_new
    
    def update_exponential_euler(
        self, 
        dt: float, 
        input_current: float
    ) -> float:
        """
        Alternative update using exponential Euler method.
        
        This method is more accurate for stiff systems but
        slightly more computationally expensive.
        
        Args:
            dt: Time step in ms
            input_current: Input current
            
        Returns:
            Updated membrane potential
        """
        params = self.parameters
        state = self.state
        
        V = state.membrane_potential
        tau = params.tau_mem
        
        # Local time constant at current V
        dV_dI = 1.0  # dV/dI = R for linear system
        
        # Compute effective time constant
        effective_tau = tau  # Simplified
        
        # Steady state for current input
        V_ss = params.resting_potential + input_current - state.adaptation
        
        # Exponential update
        V_new = V_ss + (V - V_ss) * np.exp(-dt / effective_tau)
        
        # Check threshold
        if V_new >= params.threshold:
            self.spike(dt)
            V_new = params.reset_potential
            
            if params.adaptation_enabled:
                state.adaptation += params.adapt_coefficient
        
        state.membrane_potential = V_new
        return V_new
    
    def inject_current_pulse(
        self, 
        amplitude: float, 
        duration: float, 
        start_time: float,
        current_time: float
    ) -> float:
        """
        Compute current from a pulse input.
        
        Args:
            amplitude: Pulse amplitude
            duration: Pulse duration in ms
            start_time: Pulse start time in ms
            current_time: Current simulation time
            
        Returns:
            Current amplitude at current time
        """
        if start_time <= current_time < start_time + duration:
            return amplitude
        return 0.0
    
    def compute_firing_rate(
        self, 
        input_current: float, 
        duration_ms: float = 1000.0
    ) -> float:
        """
        Compute steady-state firing rate for constant input.
        
        For LIF neurons, the firing rate follows:
            r = 1 / (τ_m * ln(R*I / (R*I - θ)) + t_ref)
            
        where R is resistance, I is current, θ is threshold,
        and t_ref is refractory period.
        
        Args:
            input_current: Constant input current
            duration_ms: Simulation duration for numerical estimate
            
        Returns:
            Firing rate in Hz
        """
        params = self.parameters
        
        if input_current <= 0:
            return 0.0
        
        # Effective driving current above threshold
        V_th = params.threshold - params.resting_potential
        I_drive = input_current * params.r_mem  # Convert to voltage
        
        if I_drive <= V_th:
            return 0.0
        
        # Firing rate formula for LIF
        tau = params.tau_mem
        t_ref = params.refractory_period
        
        # Instantaneous rate: r = 1 / (τ * ln(I / (I - θ)) + t_ref)
        if I_drive > 0 and I_drive > V_th:
            rate = 1.0 / (tau * np.log(I_drive / (I_drive - V_th + 1e-9)) + t_ref)
            return min(rate * 1000.0, 1000.0 / t_ref)  # Convert to Hz, cap at refractory limit
        return 0.0
    
    def get_phase_response_curve(
        self, 
        stimulus_phase: float,
        stimulus_duration: float = 1.0,
        num_cycles: int = 10
    ) -> float:
        """
        Compute phase response curve (PRC).
        
        The PRC describes how a neuron's spike timing is affected by
        a brief input stimulus as a function of the stimulus phase.
        
        Args:
            stimulus_phase: Phase at which stimulus is applied (0 to 1)
            stimulus_duration: Duration of stimulus in ms
            num_cycles: Number of cycles to average over
            
        Returns:
            Phase shift in milliseconds
        """
        # Simplified PRC calculation
        # Real implementation would run multiple simulations
        params = self.parameters
        
        # Approximate PRC shape (Type I vs Type II)
        if isinstance(params, LIFParameters) and params.adaptation_enabled:
            # Type I PRC (similar to quadratic integrate-and-fire)
            prc = np.sin(2 * np.pi * stimulus_phase) * params.adapt_coefficient
        else:
            # Type II PRC
            prc = np.cos(2 * np.pi * stimulus_phase) * 0.5
        
        return prc * stimulus_duration
