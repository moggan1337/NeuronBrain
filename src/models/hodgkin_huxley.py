"""
Hodgkin-Huxley Neuron Model Implementation.

The Hodgkin-Huxley model is a mathematical model that describes how
action potentials in neurons are initiated and propagated. It was
developed by Alan Hodgkin and Andrew Huxley in 1952, based on the
giant axon of the squid.

The model uses a circuit analog where the cell membrane is represented
as a capacitor and resistor in parallel with three types of voltage-gated
ion channels (Na+, K+, and leak).

Reference:
    Hodgkin, A.L. and Huxley, A.F. (1952). A quantitative description of
    membrane current and its application to conduction and excitation
    in nerve. The Journal of Physiology, 117(4), 500-544.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from .base import Neuron, NeuronParameters, NeuronType, NeuronState


class HodgkinHuxleyParameters(NeuronParameters):
    """
    Parameters specific to the Hodgkin-Huxley model.
    
    The HH model includes detailed conductance parameters for
    sodium, potassium, and leak channels.
    """
    # Maximum conductances (mS/cm²)
    g_na: float = 120.0  # Sodium channel conductance
    g_k: float = 36.0   # Potassium channel conductance
    g_l: float = 0.3    # Leak channel conductance
    
    # Reversal potentials (mV)
    E_na: float = 50.0   # Sodium reversal potential
    E_k: float = -77.0   # Potassium reversal potential
    E_l: float = -54.387 # Leak reversal potential (calculated for resting -70mV)
    
    # Membrane properties
    membrane_capacitance: float = 1.0  # µF/cm²
    
    # Channel kinetics temperature adjustment
    Q10: float = 3.0  # Temperature coefficient
    temperature: float = 6.3  # Temperature at which parameters were measured (°C)
    

@dataclass
class HodgkinHuxleyState(NeuronState):
    """State variables specific to Hodgkin-Huxley model."""
    m: float = 0.05  # Sodium activation gate
    h: float = 0.6   # Sodium inactivation gate
    n: float = 0.32  # Potassium activation gate
    

class HodgkinHuxleyNeuron(Neuron):
    """
    Hodgkin-Huxley neuron model implementation.
    
    This model accurately reproduces the action potential dynamics
    of biological neurons, including the characteristic phases:
    - Resting state
    - Depolarization (rapid Na+ influx)
    - Repolarization (K+ efflux)
    - Hyperpolarization (brief after-hyperpolarization)
    
    The model solves the following system of differential equations:
    
        C * dV/dt = g_na * m³ * h * (E_na - V) + 
                   g_k * n⁴ * (E_k - V) + 
                   g_l * (E_l - V) + I_ext
    
    where m, h, and n are gating variables governed by:
        dx/dt = α_x * (1 - x) - β_x * x   for x ∈ {m, h, n}
    
    Example:
        >>> params = HodgkinHuxleyParameters()
        >>> neuron = HodgkinHuxleyNeuron(params)
        >>> for t in range(1000):
        ...     V = neuron.update(0.01, 10.0)  # 10 µA/cm² input
    """
    
    def __init__(
        self,
        parameters: Optional[HodgkinHuxleyParameters] = None,
        id: Optional[int] = None,
        position: Optional[tuple] = None
    ):
        """
        Initialize Hodgkin-Huxley neuron.
        
        Args:
            parameters: HH-specific parameters
            id: Neuron identifier
            position: 3D position
        """
        super().__init__(
            parameters=parameters or HodgkinHuxleyParameters(),
            id=id,
            position=position,
            neuron_type=NeuronType.HodgkinHuxley
        )
        self.state = HodgkinHuxleyState(
            membrane_potential=self.parameters.resting_potential
        )
        self._initialize_gates()
        
    def _initialize_gates(self):
        """Initialize gating variables at steady state for resting potential."""
        V = self.state.membrane_potential
        self.state.m = self._m_inf(V)
        self.state.h = self._h_inf(V)
        self.state.n = self._n_inf(V)
        
    def _temperature_factor(self) -> float:
        """
        Calculate temperature correction factor using Q10.
        
        The rate constants in HH equations are temperature-dependent.
        This follows the Q10 relationship.
        """
        params = self.parameters
        if not isinstance(params, HodgkinHuxleyParameters):
            params = HodgkinHuxleyParameters()
        return params.Q10 ** ((params.temperature - 6.3) / 10.0)
    
    def _alpha_m(self, V: float) -> float:
        """
        Rate constant for m gate activation.
        
        Args:
            V: Membrane potential in mV
            
        Returns:
            Rate constant in ms⁻¹
        """
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def _beta_m(self, V: float) -> float:
        """Rate constant for m gate deactivation."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def _alpha_h(self, V: float) -> float:
        """Rate constant for h gate activation."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def _beta_h(self, V: float) -> float:
        """Rate constant for h gate deactivation."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def _alpha_n(self, V: float) -> float:
        """Rate constant for n gate activation."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def _beta_n(self, V: float) -> float:
        """Rate constant for n gate deactivation."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def _m_inf(self, V: float) -> float:
        """Steady-state value for m gate."""
        return self._alpha_m(V) / (self._alpha_m(V) + self._beta_m(V))
    
    def _h_inf(self, V: float) -> float:
        """Steady-state value for h gate."""
        return self._alpha_h(V) / (self._alpha_h(V) + self._beta_h(V))
    
    def _n_inf(self, V: float) -> float:
        """Steady-state value for n gate."""
        return self._alpha_n(V) / (self._alpha_n(V) + self._beta_n(V))
    
    def _tau_m(self, V: float) -> float:
        """Time constant for m gate (ms)."""
        return 1.0 / (self._alpha_m(V) + self._beta_m(V))
    
    def _tau_h(self, V: float) -> float:
        """Time constant for h gate (ms)."""
        return 1.0 / (self._alpha_h(V) + self._beta_h(V))
    
    def _tau_n(self, V: float) -> float:
        """Time constant for n gate (ms)."""
        return 1.0 / (self._alpha_n(V) + self._beta_n(V))
    
    def compute_dynamics(
        self, 
        dt: float, 
        state: Dict[str, float], 
        input_current: float
    ) -> Dict[str, float]:
        """
        Compute state derivatives for Hodgkin-Huxley model.
        
        Args:
            dt: Time step in ms
            state: Current state variables (V, m, h, n)
            input_current: External input current in µA/cm²
            
        Returns:
            Dictionary of derivatives {dV/dt, dm/dt, dh/dt, dn/dt}
        """
        V = state['V']
        m = state['m']
        h = state['h']
        n = state['n']
        
        params = self.parameters
        
        # Temperature correction
        temp_factor = self._temperature_factor()
        
        # Gating variable kinetics (with temperature correction)
        dm_dt = temp_factor * (self._alpha_m(V) * (1 - m) - self._beta_m(V) * m)
        dh_dt = temp_factor * (self._alpha_h(V) * (1 - h) - self._beta_h(V) * h)
        dn_dt = temp_factor * (self._alpha_n(V) * (1 - n) - self._beta_n(V) * n)
        
        # Ion currents
        I_na = params.g_na * (m ** 3) * h * (V - params.E_na)
        I_k = params.g_k * (n ** 4) * (V - params.E_k)
        I_l = params.g_l * (V - params.E_l)
        
        # Membrane potential dynamics
        dV_dt = (-(I_na + I_k + I_l) + input_current) / params.membrane_capacitance
        
        return {
            'dV_dt': dV_dt,
            'dm_dt': dm_dt,
            'dh_dt': dh_dt,
            'dn_dt': dn_dt
        }
    
    def update(self, dt: float, input_current: float) -> float:
        """
        Update neuron state using exponential Euler integration.
        
        The exponential Euler method is more stable than standard Euler
        for stiff systems like the Hodgkin-Huxley equations.
        
        Args:
            dt: Time step in milliseconds
            input_current: Input current in µA/cm²
            
        Returns:
            Updated membrane potential in mV
        """
        params = self.parameters
        state = self.state
        
        # Skip update during refractory period
        if state.refractory_time > 0:
            state.refractory_time -= dt
            return state.membrane_potential
        
        # Current state
        V = state.membrane_potential
        m, h, n = state.m, state.h, state.n
        
        # Temperature correction
        temp_factor = self._temperature_factor()
        
        # Compute time constants and steady states for gates
        tau_m = self._tau_m(V)
        tau_h = self._tau_h(V)
        tau_n = self._tau_n(V)
        
        m_inf = self._m_inf(V)
        h_inf = self._h_inf(V)
        n_inf = self._n_inf(V)
        
        # Exponential Euler update for gating variables
        m_new = m_inf + (m - m_inf) * np.exp(-dt * temp_factor / tau_m)
        h_new = h_inf + (h - h_inf) * np.exp(-dt * temp_factor / tau_h)
        n_new = n_inf + (n - n_inf) * np.exp(-dt * temp_factor / tau_n)
        
        # Compute currents at old state
        I_na = params.g_na * (m ** 3) * h * (V - params.E_na)
        I_k = params.g_k * (n ** 4) * (V - params.E_k)
        I_l = params.g_l * (V - params.E_l)
        
        # Membrane time constant
        g_total = params.g_na * (m ** 3) * h + params.g_k * (n ** 4) + params.g_l
        tau_V = params.membrane_capacitance / g_total if g_total > 0 else 1.0
        
        # Exponential Euler for membrane potential
        V_inf = (params.g_na * (m ** 3) * h * params.E_na + 
                params.g_k * (n ** 4) * params.E_k + 
                params.g_l * params.E_l + input_current) / g_total if g_total > 0 else V
        
        V_new = V_inf + (V - V_inf) * np.exp(-dt / tau_V)
        
        # Check for spike
        if V_new >= params.threshold:
            self.spike(dt)  # Use dt as time approximation
            V_new = params.reset_potential
            # Reset gating variables to near-threshold state
            m_new = self._m_inf(params.reset_potential)
            h_new = 0.9  # Fast recovery from inactivation
            n_new = self._n_inf(params.reset_potential)
        
        # Update state
        state.membrane_potential = V_new
        state.m = m_new
        state.h = h_new
        state.n = n_new
        
        return V_new
    
    def get_ion_currents(self) -> Dict[str, float]:
        """
        Get individual ion channel currents.
        
        Returns:
            Dictionary with sodium, potassium, and leak currents
        """
        params = self.parameters
        V = self.state.membrane_potential
        m, h, n = self.state.m, self.state.h, self.state.n
        
        return {
            'I_Na': params.g_na * (m ** 3) * h * (V - params.E_na),
            'I_K': params.g_k * (n ** 4) * (V - params.E_k),
            'I_L': params.g_l * (V - params.E_l)
        }
    
    def get_conductances(self) -> Dict[str, float]:
        """
        Get effective channel conductances.
        
        Returns:
            Dictionary with sodium, potassium, and leak conductances
        """
        params = self.parameters
        m, h, n = self.state.m, self.state.h, self.state.n
        
        return {
            'g_Na': params.g_na * (m ** 3) * h,
            'g_K': params.g_k * (n ** 4),
            'g_L': params.g_l
        }
