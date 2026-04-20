"""
Synaptic Plasticity Mechanisms Beyond STDP.

This module implements additional plasticity mechanisms including:
- Synaptic scaling (homeostatic plasticity)
- Metaplasticity (plasticity of plasticity)
- Structural plasticity (synapse formation/elimination)
- Intrinsic plasticity (neuron excitability changes)

Reference:
    Turrigiano, G.G. and Nelson, S.B. (2004). Homeostatic plasticity
    in the developing nervous system. Nature Reviews Neuroscience,
    5(2), 97-107.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PlasticityParameters:
    """Base parameters for plasticity mechanisms."""
    learning_rate: float = 0.01
    target_activity: float = 10.0  # Hz
    adaptation_time_constant: float = 86400.0  # seconds (slow)


class SynapticScaling:
    """
    Synaptic scaling (homeostatic plasticity).
    
    Synaptic scaling adjusts synaptic strengths to maintain overall
    neuronal activity at a target level. Unlike STDP, scaling affects
    all synapses of a neuron proportionally.
    
    The scaling rule is:
        dw_i/dt = alpha * (target_activity - actual_activity) * w_i
        
    where alpha is the learning rate and w_i is the weight of synapse i.
    
    Example:
        >>> scaling = SynapticScaling(target_activity=10.0, tau=1000.0)
        >>> 
        >>> for neuron in network.neurons:
        ...     actual_rate = neuron.get_average_firing_rate(window_ms=1000)
        ...     scaling.update(neuron, dt, actual_rate)
    """
    
    def __init__(
        self,
        parameters: Optional[PlasticityParameters] = None,
        target_activity: float = 10.0,
        tau: float = 1000.0,
        multiplicative: bool = True
    ):
        """
        Initialize synaptic scaling.
        
        Args:
            parameters: Plasticity parameters
            target_activity: Target firing rate (Hz)
            tau: Time constant for scaling (ms)
            multiplicative: Use multiplicative (True) or additive (False) scaling
        """
        self.params = parameters or PlasticityParameters()
        self.target_activity = target_activity
        self.tau = tau
        self.multiplicative = multiplicative
        
        # State variables
        self.scaling_factors: Dict[int, float] = {}  # neuron_id -> scaling factor
        self.activity_history: Dict[int, List[float]] = {}
        
    def update(
        self, 
        neuron_id: int,
        synapses: List[Any],
        actual_activity: float,
        dt: float
    ) -> float:
        """
        Update synaptic scaling for a neuron.
        
        Args:
            neuron_id: Neuron identifier
            synapses: List of synapses from this neuron
            actual_activity: Current firing rate (Hz)
            dt: Time step (ms)
            
        Returns:
            Scaling factor change
        """
        # Compute error signal
        error = self.target_activity - actual_activity
        
        # Initialize scaling factor if needed
        if neuron_id not in self.scaling_factors:
            self.scaling_factors[neuron_id] = 1.0
        
        # Record activity history
        if neuron_id not in self.activity_history:
            self.activity_history[neuron_id] = []
        self.activity_history[neuron_id].append(actual_activity)
        
        # Keep history manageable
        if len(self.activity_history[neuron_id]) > 1000:
            self.activity_history[neuron_id] = self.activity_history[neuron_id][-1000:]
        
        # Compute scaling update
        if self.multiplicative:
            # Multiplicative: scale proportionally
            delta_s = self.params.learning_rate * error / self.tau
            self.scaling_factors[neuron_id] += delta_s * self.scaling_factors[neuron_id]
        else:
            # Additive: change by fixed amount
            delta_s = self.params.learning_rate * error / self.tau
            self.scaling_factors[neuron_id] += delta_s
        
        # Bound scaling factor
        self.scaling_factors[neuron_id] = np.clip(
            self.scaling_factors[neuron_id], 0.1, 10.0
        )
        
        # Apply scaling to all synapses
        scaling = self.scaling_factors[neuron_id]
        for synapse in synapses:
            synapse.weight *= scaling
            synapse.weight = np.clip(synapse.weight, 0.01, 10.0)
        
        return self.scaling_factors[neuron_id]
    
    def get_scaling_factor(self, neuron_id: int) -> float:
        """Get current scaling factor for a neuron."""
        return self.scaling_factors.get(neuron_id, 1.0)
    
    def reset(self, neuron_id: Optional[int] = None):
        """Reset scaling factors."""
        if neuron_id is None:
            self.scaling_factors.clear()
            self.activity_history.clear()
        else:
            self.scaling_factors.pop(neuron_id, None)
            self.activity_history.pop(neuron_id, None)


class IntrinsicPlasticity:
    """
    Intrinsic plasticity (homeostatic adjustment of excitability).
    
    This mechanism adjusts neuronal excitability (membrane properties)
    to maintain stable firing activity. It can modify:
    - Threshold
    - Membrane time constant
    - Resting potential
    - Adaptation parameters
    
    Reference:
        Zhang, W. and Linden, D.J. (2003). The other side of plasticity:
        sensory input-dependent modulation of synaptic strength. Nature
        Reviews Neuroscience, 4(12), 941-947.
    """
    
    def __init__(
        self,
        target_activity: float = 10.0,
        learning_rate: float = 0.001,
        target_property: str = 'threshold'
    ):
        """
        Initialize intrinsic plasticity.
        
        Args:
            target_activity: Target firing rate (Hz)
            learning_rate: Rate of parameter adjustment
            target_property: Which property to adjust ('threshold', 'tau', 'reset')
        """
        self.target_activity = target_activity
        self.learning_rate = learning_rate
        self.target_property = target_property
        
        # State
        self.property_values: Dict[int, float] = {}
        
    def update(
        self,
        neuron: Any,
        actual_activity: float,
        dt: float
    ):
        """
        Update intrinsic properties of a neuron.
        
        Args:
            neuron: Neuron object with adjustable properties
            actual_activity: Current firing rate (Hz)
            dt: Time step (ms)
        """
        error = self.target_activity - actual_activity
        
        # Get current property value
        if neuron.id not in self.property_values:
            self.property_values[neuron.id] = getattr(
                neuron.parameters, self.target_property
            )
        
        # Compute adjustment
        delta = self.learning_rate * error
        
        # Apply adjustment
        current_value = self.property_values[neuron.id]
        new_value = current_value + delta
        
        # Set bounds based on property type
        if self.target_property == 'threshold':
            new_value = np.clip(new_value, -80.0, -40.0)
        elif self.target_property == 'tau_mem':
            new_value = np.clip(new_value, 5.0, 50.0)
        elif self.target_property == 'reset_potential':
            new_value = np.clip(new_value, -80.0, -50.0)
        
        # Update neuron parameter
        self.property_values[neuron.id] = new_value
        
        # Apply to neuron
        if hasattr(neuron.parameters, self.target_property):
            setattr(neuron.parameters, self.target_property, new_value)


class StructuralPlasticity:
    """
    Structural plasticity (synapse formation and elimination).
    
    This mechanism allows synapses to be created and removed based on
    activity patterns, implementing a form of structural plasticity
    that can reshape network connectivity.
    """
    
    def __init__(
        self,
        formation_rate: float = 0.001,
        elimination_rate: float = 0.0005,
        activity_threshold: float = 1.0,
        min_weight: float = 0.1
    ):
        """
        Initialize structural plasticity.
        
        Args:
            formation_rate: Rate of new synapse formation
            elimination_rate: Rate of synapse elimination
            activity_threshold: Activity level for synapse survival
            min_weight: Weight below which synapse is eliminated
        """
        self.formation_rate = formation_rate
        self.elimination_rate = elimination_rate
        self.activity_threshold = activity_threshold
        self.min_weight = min_weight
        
        # Statistics
        self.formations: int = 0
        self.eliminations: int = 0
        
    def update(
        self,
        synapses: List[Any],
        pre_activity: float,
        post_activity: float,
        dt: float
    ) -> Tuple[List[Any], List[Any]]:
        """
        Update structural plasticity.
        
        Args:
            synapses: Current list of synapses
            pre_activity: Presynaptic activity (Hz)
            post_activity: Postsynaptic activity (Hz)
            dt: Time step (ms)
            
        Returns:
            Tuple of (new_synapses_to_add, synapses_to_remove)
        """
        to_form = []
        to_remove = []
        
        # Check for synapse elimination
        for synapse in synapses:
            avg_activity = (pre_activity + post_activity) / 2
            avg_weight = synapse.weight
            
            # Eliminate weak synapses with low activity
            prob_eliminate = self.elimination_rate * (1 - avg_activity / self.activity_threshold)
            
            if avg_weight < self.min_weight or np.random.random() < prob_eliminate * dt:
                to_remove.append(synapse)
                self.eliminations += 1
        
        # Formation of new synapses (simplified model)
        if pre_activity > self.activity_threshold and post_activity < 1.0:
            prob_form = self.formation_rate * pre_activity / self.activity_threshold
            if np.random.random() < prob_form * dt:
                # This would return parameters for a new synapse
                to_form.append({
                    'pre_id': synapses[0].pre_id if synapses else None,
                    'post_id': synapses[0].post_id if synapses else None,
                    'weight': np.random.uniform(0.1, 0.5)
                })
                self.formations += 1
        
        return to_form, to_remove
    
    def get_statistics(self) -> Dict[str, int]:
        """Get structural plasticity statistics."""
        return {
            'total_formations': self.formations,
            'total_eliminations': self.eliminations,
            'net_change': self.formations - self.eliminations
        }


class BCMPlasticity:
    """
    Bienenstock-Cooper-Munro (BCM) learning rule.
    
    The BCM rule implements a sliding threshold for plasticity:
        dw/dt = phi(pre) * post * (post - theta)
        
    where phi is a nonlinear function and theta is a sliding threshold.
    The threshold moves based on average postsynaptic activity.
    
    Reference:
        Bienenstock, E.L., Cooper, L.N., and Munro, P.W. (1982). Theory
        for the development of neuron selectivity: orientation specificity
        and binocular interaction in visual cortex. Journal of Neuroscience,
        2(1), 32-48.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        theta_decay: float = 0.001,
        threshold_sliding_rate: float = 0.01
    ):
        """
        Initialize BCM plasticity.
        
        Args:
            learning_rate: Learning rate
            theta_decay: Decay rate for theta (plasticity threshold)
            threshold_sliding_rate: Rate at which threshold slides
        """
        self.learning_rate = learning_rate
        self.theta_decay = theta_decay
        self.threshold_sliding_rate = threshold_sliding_rate
        
        # State
        self.thresholds: Dict[int, float] = {}
        
    def compute_update(
        self,
        synapse: Any,
        pre_activity: float,
        post_activity: float,
        dt: float
    ) -> float:
        """
        Compute BCM weight update.
        
        Args:
            synapse: Synapse object
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            dt: Time step
            
        Returns:
            Weight change
        """
        # Get or initialize threshold
        if synapse.post_id not in self.thresholds:
            self.thresholds[synapse.post_id] = 1.0
        
        theta = self.thresholds[synapse.post_id]
        
        # BCM rule: dw = pre * post * (post - theta)
        # Using activities as approximations for spike rates
        dw = self.learning_rate * pre_activity * post_activity * (post_activity - theta)
        
        # Update threshold (moving average of post activity)
        theta += self.threshold_sliding_rate * (post_activity**2 - theta)
        theta = max(theta, 0.1)  # Keep theta positive
        self.thresholds[synapse.post_id] = theta
        
        return dw
    
    def apply_update(self, synapse: Any, dw: float):
        """Apply weight update with bounds."""
        synapse.weight += dw
        synapse.weight = np.clip(synapse.weight, 0.0, 10.0)


class OjaLearningRule:
    """
    Oja's learning rule with synaptic normalization.
    
    Oja's rule ensures that the sum of squared weights remains constant:
        dw/dt = eta * pre * post - eta * alpha * post^2 * w
        
    where alpha controls the normalization strength.
    
    Reference:
        Oja, E. (1982). Simplified neuron model as a principal component
        analyzer. Journal of Mathematical Biology, 15(3), 267-273.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        normalization: float = 1.0
    ):
        """
        Initialize Oja learning.
        
        Args:
            learning_rate: Learning rate (eta)
            normalization: Normalization strength (alpha)
        """
        self.learning_rate = learning_rate
        self.normalization = normalization
        
    def compute_update(
        self,
        synapse: Any,
        pre_activity: float,
        post_activity: float
    ) -> float:
        """
        Compute Oja weight update.
        
        Args:
            synapse: Synapse object
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            
        Returns:
            Weight change
        """
        # Oja's rule
        dw = (self.learning_rate * pre_activity * post_activity - 
              self.learning_rate * self.normalization * post_activity**2 * synapse.weight)
        
        return dw
