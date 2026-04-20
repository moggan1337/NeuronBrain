"""
Electrical Synapse (Gap Junction) Models.

Electrical synapses provide direct electrical coupling between neurons
through gap junctions. Unlike chemical synapses, they allow bidirectional
current flow and have no synaptic delay. They are important in various
brain regions including the thalamus, hypothalamus, and neocortex.

Gap junctions are modeled as a conductance between two neurons:
    I_gap = g_gap * (V_pre - V_post)

where g_gap is the gap junction conductance that may be voltage-dependent.

Reference:
    Bennett, M.V. and Zukin, R.S. (2004). Electrical coupling and
    neuronal synchronization in the mammalian brain. Neuron, 41(4),
    495-511.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class GapJunctionParameters:
    """
    Parameters for gap junction dynamics.
    
    Electrical synapses can have complex voltage-dependence and
    modulation by neuromodulators.
    """
    g_max: float = 1.0       # Maximum conductance (nS)
    v_dependence: bool = False  # Voltage-dependent gating
    v_half: float = 0.0       # Half-activation voltage (mV)
    gating_slope: float = 10.0  # Slope of voltage dependence
    rectification: bool = False  # Rectifying gap junctions
    delay: float = 0.0       # No delay for electrical synapses
    
    # Pre and post neuron IDs
    pre_neuron_id: Optional[int] = None
    post_neuron_id: Optional[int] = None
    
    # Modulation
    dopamine_modulation: float = 1.0  # Dopamine effect
    calcium_modulation: float = 1.0   # Ca²⁺ dependent gating


class GapJunctionState:
    """State variables for gap junction."""
    def __init__(self):
        self.gap_conductance: float = 0.0  # Current conductance
        self.last_v_pre: float = -70.0     # Previous presynaptic V
        self.last_v_post: float = -70.0     # Previous postsynaptic V
        self.modulation_factor: float = 1.0
        
    def reset(self):
        """Reset state."""
        self.gap_conductance = 0.0
        self.last_v_pre = -70.0
        self.last_v_post = -70.0
        self.modulation_factor = 1.0


class ElectricalSynapse:
    """
    Electrical synapse model (gap junction).
    
    Electrical synapses provide:
    - Bidirectional current flow
    - Zero-latency signaling
    - Synchronization of neuronal activity
    - Possible voltage-dependence
    
    The gap junction current is:
        I_gap = g_gap * (V_pre - V_post)
        
    Where g_gap may be voltage-dependent:
        g_gap = g_max / (1 + exp((V - v_half) / slope))
        
    Example:
        >>> gap_junction = ElectricalSynapse(
        ...     pre_id=1,
        ...     post_id=2,
        ...     g_max=2.0
        ... )
        >>> 
        >>> # Bidirectional coupling
        >>> I_pre = gap_junction.get_current(v_pre, v_post)  # Current at pre neuron
        >>> I_post = -I_pre  # Equal and opposite at post neuron
    """
    
    def __init__(
        self,
        parameters: Optional[GapJunctionParameters] = None,
        pre_id: Optional[int] = None,
        post_id: Optional[int] = None,
        conductance: float = 1.0,
        voltage_dependent: bool = False
    ):
        """
        Initialize electrical synapse.
        
        Args:
            parameters: Custom gap junction parameters
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            conductance: Gap junction conductance (nS)
            voltage_dependent: Enable voltage-dependent gating
        """
        if parameters is None:
            parameters = GapJunctionParameters(
                pre_neuron_id=pre_id,
                post_neuron_id=post_id,
                g_max=conductance,
                v_dependence=voltage_dependent
            )
        
        self.parameters = parameters
        self.pre_id = pre_id or parameters.pre_neuron_id
        self.post_id = post_id or parameters.post_neuron_id
        self.state = GapJunctionState()
        
        # For network simulations - track which neurons this couples
        self._coupled = (self.pre_id, self.post_id)
        
    def compute_conductance(self, v_pre: float, v_post: float) -> float:
        """
        Compute effective gap junction conductance.
        
        Args:
            v_pre: Presynaptic membrane potential (mV)
            v_post: Postsynaptic membrane potential (mV)
            
        Returns:
            Effective conductance (nS)
        """
        params = self.parameters
        state = self.state
        
        g = params.g_max
        
        # Voltage-dependent gating
        if params.v_dependence:
            # Use transjunctional voltage
            v_junction = v_pre - v_post
            gate = 1.0 / (1.0 + np.exp((np.abs(v_junction) - params.v_half) / params.gating_slope))
            g *= gate
        
        # Rectification (asymmetric conductance)
        if params.rectification:
            if v_pre > v_post:
                g *= 1.2  # More conductance in one direction
            else:
                g *= 0.8  # Less in the other
        
        # Neuromodulator modulation
        g *= state.modulation_factor * params.dopamine_modulation * params.calcium_modulation
        
        return g
    
    def get_current(self, v_pre: float, v_post: float) -> Tuple[float, float]:
        """
        Compute bidirectional gap junction current.
        
        Args:
            v_pre: Presynaptic membrane potential (mV)
            v_post: Postsynaptic membrane potential (mV)
            
        Returns:
            Tuple of (current_at_pre, current_at_post) in nA
        """
        g = self.compute_conductance(v_pre, v_post)
        
        # Current flows from higher to lower potential
        # Positive current at pre neuron = current flowing INTO pre from post
        current = g * (v_pre - v_post)
        
        # Update state
        self.state.gap_conductance = g
        self.state.last_v_pre = v_pre
        self.state.last_v_post = v_post
        
        # Current at pre neuron is opposite to current at post neuron
        return current, -current
    
    def get_current_pre_to_post(self, v_pre: float, v_post: float) -> float:
        """
        Get current flowing from pre to post neuron.
        
        Args:
            v_pre: Presynaptic membrane potential (mV)
            v_post: Postsynaptic membrane potential (mV)
            
        Returns:
            Current in nA (positive = flowing pre to post)
        """
        g = self.compute_conductance(v_pre, v_post)
        return g * (v_pre - v_post)
    
    def update_modulation(
        self, 
        dopamine: float = 1.0,
        calcium: float = 1.0,
        other_modulators: Optional[Dict[str, float]] = None
    ):
        """
        Update gap junction modulation by neuromodulators.
        
        Gap junctions can be modulated by:
        - Dopamine (reduces coupling in some regions)
        - Calcium (can close gap junctions)
        - Nitric oxide
        - Neuromodulators
        
        Args:
            dopamine: Dopamine modulation factor
            calcium: Calcium-dependent modulation
            other_modulators: Additional modulator effects
        """
        self.parameters.dopamine_modulation = dopamine
        self.parameters.calcium_modulation = calcium
        
        if other_modulators:
            total = 1.0
            for mod in other_modulators.values():
                total *= mod
            self.state.modulation_factor = total
    
    def apply_plasticity(
        self, 
        delta_v: float,
        learning_rate: float = 0.001
    ):
        """
        Apply gap junction plasticity based on activity.
        
        Gap junction conductance can change based on:
        - Correlated activity (upregulate)
        - Uncorrelated activity (downregulate)
        
        Args:
            delta_v: Voltage difference (v_pre - v_post)
            learning_rate: Plasticity learning rate
        """
        # Simple Hebbian-like plasticity
        correlation = np.exp(-delta_v**2 / 100)  # Higher for correlated activity
        
        delta_g = learning_rate * (correlation - 0.5)  # Target ~50% correlation
        self.parameters.g_max += delta_g
        self.parameters.g_max = np.clip(self.parameters.g_max, 0.01, 10.0)
    
    def reset(self):
        """Reset gap junction state."""
        self.state.reset()
        
    def get_conductance(self) -> float:
        """Get current gap junction conductance."""
        return self.state.gap_conductance
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gap junction statistics."""
        return {
            'g_max': self.parameters.g_max,
            'current_conductance': self.state.gap_conductance,
            'modulation': self.state.modulation_factor,
            'dopamine_effect': self.parameters.dopamine_modulation,
            'calcium_effect': self.parameters.calcium_modulation,
        }
    
    def __repr__(self) -> str:
        return (f"ElectricalSynapse(pre={self.pre_id}, post={self.post_id}, "
                f"g={self.parameters.g_max:.2f}nS)")


# Alias for clarity
GapJunction = ElectricalSynapse


class GapJunctionNetwork:
    """
    Network of gap junctions for efficient simulation.
    
    When simulating networks with many gap junctions, this class
    provides efficient matrix-based computation.
    
    Example:
        >>> # Create gap junction matrix (n x n) where G[i,j] = conductance
        >>> G = np.array([[0, 1.0], [1.0, 0]])  # Symmetric for undirected
        >>> network = GapJunctionNetwork(G, neuron_ids=[1, 2])
        >>> 
        >>> # Compute all gap junction currents
        >>> currents = network.compute_currents(voltages)  # n neurons
    """
    
    def __init__(
        self, 
        conductance_matrix: np.ndarray,
        neuron_ids: Optional[list] = None
    ):
        """
        Initialize gap junction network.
        
        Args:
            conductance_matrix: N x N symmetric matrix of conductances
            neuron_ids: List of neuron IDs corresponding to rows/columns
        """
        self.G = conductance_matrix.astype(float)
        self.n = len(conductance_matrix)
        self.neuron_ids = neuron_ids or list(range(self.n))
        
        # Verify symmetry (gap junctions are bidirectional)
        if not np.allclose(self.G, self.G.T):
            raise ValueError("Gap junction matrix must be symmetric")
    
    def compute_currents(self, voltages: np.ndarray) -> np.ndarray:
        """
        Compute gap junction currents for all neurons.
        
        I_i = sum_j(G[i,j] * (V_i - V_j))
        
        Args:
            voltages: Array of membrane potentials (n neurons)
            
        Returns:
            Array of gap junction currents
        """
        # Compute current matrix
        I_matrix = self.G * (voltages[:, np.newaxis] - voltages[np.newaxis, :])
        
        # Sum along columns to get current at each neuron
        currents = np.sum(I_matrix, axis=1)
        
        return currents
    
    def compute_coupling_matrix(self) -> np.ndarray:
        """
        Compute effective coupling strength for each neuron pair.
        
        Returns:
            N x N matrix of coupling strengths
        """
        return self.G.copy()
    
    def get_synchronization_index(self, voltages: np.ndarray) -> float:
        """
        Compute synchronization index (Kuramoto order parameter).
        
        Args:
            voltages: Array of membrane potentials
            
        Returns:
            Order parameter r (0 = async, 1 = sync)
        """
        # Normalize voltages to phase-like representation
        v_centered = voltages - np.mean(voltages)
        r = np.abs(np.mean(np.exp(1j * v_centered / np.std(v_centered))))
        return r
    
    def add_gap_junction(self, i: int, j: int, g: float):
        """Add or update a gap junction between neurons i and j."""
        if i >= self.n or j >= self.n:
            raise ValueError(f"Neuron index out of range (n={self.n})")
        self.G[i, j] = g
        self.G[j, i] = g  # Ensure symmetry
    
    def remove_gap_junction(self, i: int, j: int):
        """Remove gap junction between neurons i and j."""
        self.add_gap_junction(i, j, 0.0)
    
    def get_total_conductance(self) -> float:
        """Get total gap junction conductance in the network."""
        return np.sum(self.G) / 2  # Divide by 2 since symmetric
    
    def get_connectivity_density(self) -> float:
        """Get fraction of possible connections that exist."""
        n_connections = np.sum(self.G > 0) / 2
        n_possible = self.n * (self.n - 1) / 2
        return n_connections / n_possible if n_possible > 0 else 0.0
