"""
Spike-Timing-Dependent Plasticity (STDP) Implementation.

STDP is a biological learning rule where the synaptic strength changes
based on the relative timing of presynaptic and postsynaptic spikes.
If the presynaptic spike occurs before the postsynaptic spike (LTP),
the synapse is strengthened. If the postsynaptic spike occurs before
the presynaptic spike (LTD), the synapse is weakened.

The classic STDP rule is:
    Δw = A_plus * exp(-Δt / tau_plus) if Δt > 0
    Δw = -A_minus * exp(Δt / tau_minus) if Δt < 0
    
where Δt = t_post - t_pre

This module implements several STDP variants including:
- Additive STDP
- Multiplicative STDP  
- Soft bounds vs hard bounds
- Triplet STDP
- Burst-dependent STDP

Reference:
    Bi, G. and Poo, M. (2001). Synaptic modification by correlated
    activity: Hebb's postulate revisited. Annual Review of Neuroscience,
    24(1), 139-166.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class STDPCurve(Enum):
    """STDP curve type."""
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    TRIANGLE = "triangle"
    GAUSSIAN = "gaussian"


@dataclass
class STDPParameters:
    """
    Parameters for STDP learning.
    
    These parameters control the shape of the STDP learning window
    and the learning rate.
    """
    # Learning rates
    A_plus: float = 0.01     # Potentiation amplitude
    A_minus: float = 0.012   # Depression amplitude
    
    # Time constants
    tau_plus: float = 20.0   # Potentiation time constant (ms)
    tau_minus: float = 20.0  # Depression time constant (ms)
    
    # Weight bounds
    w_min: float = 0.0       # Minimum weight
    w_max: float = 1.0       # Maximum weight
    
    # STDP type
    curve_type: STDPCurve = STDPCurve.EXPONENTIAL
    soft_bounds: bool = True  # Soft vs hard bounds
    
    # Optional parameters
    use_triplets: bool = False
    triplet_scale: float = 1.0
    
    # Timing parameters
    learning_window: float = 100.0  # Time window for spike pairs (ms)
    
    # Modulation
    neuromodulation: bool = False
    dopamine_factor: float = 1.0
    
    # Eligibility trace
    use_eligibility_trace: bool = False
    eligibility_tau: float = 1000.0  # ms


class STDPSynapse:
    """
    Synapse with STDP learning capabilities.
    
    This class wraps a synapse and adds STDP learning rules.
    """
    
    def __init__(
        self,
        synapse: Any,
        parameters: Optional[STDPParameters] = None
    ):
        """
        Initialize STDP synapse.
        
        Args:
            synapse: Base synapse object
            parameters: STDP parameters
        """
        self.synapse = synapse
        self.params = parameters or STDPParameters()
        
        # STDP state
        self.pre_traces: Dict[str, float] = {'A': 0.0, 'B': 0.0}
        self.post_traces: Dict[str, float] = {'A': 0.0, 'B': 0.0}
        
        # Eligibility trace for reward-modulated STDP
        self.eligibility_trace = 0.0
        
        # Learning history
        self.weight_history: List[float] = [synapse.weight]
        self.stdp_events: List[Dict] = []
        
    def update_traces(
        self, 
        dt: float, 
        pre_spike: bool = False, 
        post_spike: bool = False
    ):
        """
        Update eligibility traces.
        
        Args:
            dt: Time step (ms)
            pre_spike: Whether presynaptic spike occurred
            post_spike: Whether postsynaptic spike occurred
        """
        params = self.params
        p = self.pre_traces
        q = self.post_traces
        
        if params.use_triplets:
            # Triplet STDP model (Pfister & Gerstner)
            # Two pre traces and two post traces
            decay_A = np.exp(-dt / params.tau_plus)
            decay_B = np.exp(-dt / 100.0)  # Slow trace
            
            p['A'] *= decay_A
            p['B'] *= decay_B
            q['A'] *= decay_A
            q['B'] *= decay_B
            
            if pre_spike:
                p['A'] += 1.0
                p['B'] += 1.0
            if post_spike:
                q['A'] += 1.0
                q['B'] += 1.0
        else:
            # Standard STDP with single trace
            decay = np.exp(-dt / params.tau_plus)
            
            p['A'] *= decay
            q['A'] *= decay
            
            if pre_spike:
                p['A'] += 1.0
            if post_spike:
                q['A'] += 1.0
        
        # Update eligibility trace
        if params.use_eligibility_trace:
            self.eligibility_trace *= np.exp(-dt / params.eligibility_tau)
    
    def compute_weight_update(
        self, 
        pre_spike: bool, 
        post_spike: bool,
        delta_t: float
    ) -> float:
        """
        Compute weight change based on STDP rule.
        
        Args:
            pre_spike: Presynaptic spike occurred
            post_spike: Postsynaptic spike occurred
            delta_t: t_post - t_pre (ms)
            
        Returns:
            Weight change
        """
        params = self.params
        w = self.synapse.weight
        
        if params.curve_type == STDPCurve.EXPONENTIAL:
            if delta_t > 0:
                # LTP: pre before post
                dw = params.A_plus * np.exp(-delta_t / params.tau_plus)
            else:
                # LTD: post before pre
                dw = -params.A_minus * np.exp(delta_t / params.tau_minus)
                
        elif params.curve_type == STDPCurve.POWER_LAW:
            if delta_t > 0:
                dw = params.A_plus * (1 - w / params.w_max) * np.power(delta_t / params.tau_plus + 1, -1)
            else:
                dw = -params.A_minus * (w / params.w_min) * np.power(-delta_t / params.tau_minus + 1, -1)
                
        elif params.curve_type == STDPCurve.GAUSSIAN:
            if delta_t > 0:
                sigma = params.tau_plus / 2
                dw = params.A_plus * np.exp(-delta_t**2 / (2 * sigma**2))
            else:
                sigma = params.tau_minus / 2
                dw = -params.A_minus * np.exp(-delta_t**2 / (2 * sigma**2))
                
        else:  # Triangle
            if delta_t > 0:
                dw = params.A_plus * max(0, 1 - delta_t / params.tau_plus)
            else:
                dw = -params.A_minus * max(0, 1 + delta_t / params.tau_minus)
        
        # Apply neuromodulation
        if params.neuromodulation:
            dw *= params.dopamine_factor
        
        return dw
    
    def apply_weight_update(self, dw: float):
        """Apply weight change with bounds."""
        params = self.params
        w = self.synapse.weight
        
        if params.soft_bounds:
            # Soft bounds: weights converge to bounds
            if dw > 0:
                w += dw * (params.w_max - w)
            else:
                w += dw * (w - params.w_min)
        else:
            # Hard bounds: clip to limits
            w += dw
            w = np.clip(w, params.w_min, params.w_max)
        
        self.synapse.weight = w
        self.weight_history.append(w)
    
    def reset(self):
        """Reset STDP state."""
        self.pre_traces = {'A': 0.0, 'B': 0.0}
        self.post_traces = {'A': 0.0, 'B': 0.0}
        self.eligibility_trace = 0.0
        self.stdp_events.clear()


class STDP:
    """
    STDP Learning Rule Implementation.
    
    This class manages STDP learning across a network of synapses.
    It can apply different STDP variants and track learning.
    
    Example:
        >>> stdp = STDP(parameters)
        >>> 
        >>> # In simulation loop
        >>> for synapse in network.synapses:
        ...     if pre_spike:
        ...         stdp.record_presynaptic_spike(synapse, time)
        ...     if post_spike:
        ...         stdp.record_postsynaptic_spike(synapse, time)
        ... 
        ...     dw = stdp.compute_update(synapse)
        ...     synapse.weight += dw
    """
    
    def __init__(
        self, 
        parameters: Optional[STDPParameters] = None,
        learning_rate: float = 1.0
    ):
        """
        Initialize STDP learner.
        
        Args:
            parameters: STDP parameters
            learning_rate: Overall learning rate multiplier
        """
        self.params = parameters or STDPParameters()
        self.learning_rate = learning_rate
        
        # Spike timing records
        self._pre_spikes: Dict[str, List[float]] = {}
        self._post_spikes: Dict[str, List[float]] = {}
        
        # Statistics
        self.total_potentiation: float = 0.0
        self.total_depression: float = 0.0
        self.num_events: int = 0
    
    def record_presynaptic_spike(self, synapse_id: str, time: float):
        """
        Record a presynaptic spike.
        
        Args:
            synapse_id: Unique identifier for the synapse
            time: Spike time (ms)
        """
        if synapse_id not in self._pre_spikes:
            self._pre_spikes[synapse_id] = []
        self._pre_spikes[synapse_id].append(time)
        
        # Clean old spikes
        self._clean_old_spikes(synapse_id, time)
    
    def record_postsynaptic_spike(self, synapse_id: str, time: float):
        """
        Record a postsynaptic spike.
        
        Args:
            synapse_id: Unique identifier for the synapse
            time: Spike time (ms)
        """
        if synapse_id not in self._post_spikes:
            self._post_spikes[synapse_id] = []
        self._post_spikes[synapse_id].append(time)
        
        self._clean_old_spikes(synapse_id, time)
    
    def _clean_old_spikes(self, synapse_id: str, current_time: float):
        """Remove spikes outside the learning window."""
        window = self.params.learning_window
        
        if synapse_id in self._pre_spikes:
            self._pre_spikes[synapse_id] = [
                t for t in self._pre_spikes[synapse_id]
                if current_time - t < window
            ]
        
        if synapse_id in self._post_spikes:
            self._post_spikes[synapse_id] = [
                t for t in self._post_spikes[synapse_id]
                if current_time - t < window
            ]
    
    def compute_update(
        self, 
        synapse_id: str,
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute STDP weight update for a synapse.
        
        Args:
            synapse_id: Synapse identifier
            current_time: Current simulation time
            
        Returns:
            Weight change
        """
        if synapse_id not in self._pre_spikes or synapse_id not in self._post_spikes:
            return 0.0
        
        pre_times = self._pre_spikes[synapse_id]
        post_times = self._post_spikes[synapse_id]
        
        if len(pre_times) == 0 or len(post_times) == 0:
            return 0.0
        
        dw_total = 0.0
        
        # Compute weight changes based on all spike pairs
        if current_time is not None:
            # Use current time as reference
            t_ref = current_time
        else:
            t_ref = max(max(pre_times, default=0), max(post_times, default=0))
        
        # LTP: pre spikes followed by post spikes
        for t_pre in pre_times:
            for t_post in post_times:
                if t_post > t_pre:
                    delta_t = t_post - t_pre
                    if delta_t < self.params.learning_window:
                        dw = self._compute_ltp(delta_t)
                        dw_total += dw
                        self.total_potentiation += dw
        
        # LTD: post spikes followed by pre spikes
        for t_post in post_times:
            for t_pre in pre_times:
                if t_pre > t_post:
                    delta_t = t_post - t_pre  # Negative
                    if abs(delta_t) < self.params.learning_window:
                        dw = self._compute_ltd(delta_t)
                        dw_total += dw
                        self.total_depression += abs(dw)
        
        self.num_events += 1
        
        return dw_total * self.learning_rate
    
    def _compute_ltp(self, delta_t: float) -> float:
        """
        Compute LTP (potentiation) component.
        
        Args:
            delta_t: t_post - t_pre (positive)
            
        Returns:
            Weight change
        """
        params = self.params
        
        if params.curve_type == STDPCurve.EXPONENTIAL:
            return params.A_plus * np.exp(-delta_t / params.tau_plus)
        elif params.curve_type == STDPCurve.POWER_LAW:
            return params.A_plus * np.power(delta_t / params.tau_plus + 1, -1)
        elif params.curve_type == STDPCurve.GAUSSIAN:
            sigma = params.tau_plus / 2
            return params.A_plus * np.exp(-delta_t**2 / (2 * sigma**2))
        elif params.curve_type == STDPCurve.TRIANGLE:
            return params.A_plus * max(0, 1 - delta_t / params.tau_plus)
        
        return 0.0
    
    def _compute_ltd(self, delta_t: float) -> float:
        """
        Compute LTD (depression) component.
        
        Args:
            delta_t: t_post - t_pre (negative)
            
        Returns:
            Weight change (negative)
        """
        params = self.params
        
        if params.curve_type == STDPCurve.EXPONENTIAL:
            return -params.A_minus * np.exp(delta_t / params.tau_minus)
        elif params.curve_type == STDPCurve.POWER_LAW:
            return -params.A_minus * np.power(-delta_t / params.tau_minus + 1, -1)
        elif params.curve_type == STDPCurve.GAUSSIAN:
            sigma = params.tau_minus / 2
            return -params.A_minus * np.exp(-delta_t**2 / (2 * sigma**2))
        elif params.curve_type == STDPCurve.TRIANGLE:
            return -params.A_minus * max(0, 1 + delta_t / params.tau_minus)
        
        return 0.0
    
    def apply_update(
        self, 
        synapse: Any, 
        dw: float,
        soft_bounds: bool = True
    ):
        """
        Apply weight update to a synapse.
        
        Args:
            synapse: Synapse object with weight attribute
            dw: Weight change
            soft_bounds: Use soft bounds (converging)
        """
        params = self.params
        w = synapse.weight
        
        if soft_bounds or params.soft_bounds:
            if dw > 0:
                w += dw * (params.w_max - w)
            else:
                w += dw * (w - params.w_min)
        else:
            w += dw
            w = np.clip(w, params.w_min, params.w_max)
        
        synapse.weight = w
    
    def reset(self):
        """Reset STDP state and statistics."""
        self._pre_spikes.clear()
        self._post_spikes.clear()
        self.total_potentiation = 0.0
        self.total_depression = 0.0
        self.num_events = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get STDP learning statistics."""
        return {
            'total_potentiation': self.total_potentiation,
            'total_depression': self.total_depression,
            'num_events': self.num_events,
            'net_plasticity': self.total_potentiation - self.total_depression,
            'balance': self.total_potentiation / max(self.total_depression, 1e-10),
        }
    
    def get_learning_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the STDP learning window for visualization.
        
        Returns:
            Tuple of (delta_t values, weight changes)
        """
        dt_range = np.linspace(-100, 100, 1000)
        dw_range = np.zeros_like(dt_range)
        
        for i, dt in enumerate(dt_range):
            if dt > 0:
                dw_range[i] = self._compute_ltp(dt)
            else:
                dw_range[i] = self._compute_ltd(dt)
        
        return dt_range, dw_range


class RewardModulatedSTDP(STDP):
    """
    Reward-modulated STDP (R-STDP).
    
    This variant of STDP incorporates neuromodulatory signals
    (like dopamine) to gate learning. Synaptic changes only occur
    when a reward signal coincides with eligibility traces.
    
    Reference:
        Izhikevich, E.M. (2007). Solving the distal reward problem
        through linkage of STDP and dopamine signaling. Neural
        Computation, 19(10), 2571-2625.
    """
    
    def __init__(
        self, 
        parameters: Optional[STDPParameters] = None,
        learning_rate: float = 1.0,
        reward_factor: float = 1.0
    ):
        super().__init__(parameters, learning_rate)
        self.reward_factor = reward_factor
        self.reward_signal: float = 0.0
        self.eligibility_traces: Dict[str, float] = {}
        
    def update_eligibility_trace(
        self, 
        synapse_id: str, 
        dw: float, 
        dt: float
    ):
        """
        Update eligibility trace for a synapse.
        
        Args:
            synapse_id: Synapse identifier
            dw: Raw STDP weight change
            dt: Time step
        """
        tau = self.params.eligibility_tau
        
        if synapse_id not in self.eligibility_traces:
            self.eligibility_traces[synapse_id] = 0.0
        
        # Decay trace
        self.eligibility_traces[synapse_id] *= np.exp(-dt / tau)
        
        # Add new contribution
        self.eligibility_traces[synapse_id] += dw
    
    def apply_reward(
        self, 
        reward: float, 
        current_time: float,
        synapses: Dict[str, Any]
    ):
        """
        Apply reward signal to all synapses based on eligibility traces.
        
        Args:
            reward: Reward signal value
            current_time: Current time
            synapses: Dictionary of synapses
        """
        self.reward_signal = reward
        
        for synapse_id, trace in self.eligibility_traces.items():
            if synapse_id in synapses:
                dw_reward = self.reward_factor * reward * trace
                synapse = synapses[synapse_id]
                self.apply_update(synapse, dw_reward)
        
        # Decay eligibility traces
        for synapse_id in self.eligibility_traces:
            self.eligibility_traces[synapse_id] *= np.exp(-1.0 / 100)  # Decay
    
    def reset(self):
        """Reset R-STDP state."""
        super().reset()
        self.eligibility_traces.clear()
        self.reward_signal = 0.0
