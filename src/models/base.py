"""
Base neuron class and types for NeuronBrain.

This module provides the foundational abstractions for all neuron models
in the simulator. Neurons are the basic computational units that simulate
the electrical behavior of biological neurons.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np


class NeuronType(Enum):
    """Enumeration of supported neuron types."""
    HodgkinHuxley = "hodgkin_huxley"
    LeakyIntegrateAndFire = "lif"
    Izhikevich = "izhikevich"
    Custom = "custom"


@dataclass
class NeuronState:
    """
    Represents the current state of a neuron.
    
    Attributes:
        membrane_potential: Current membrane voltage in millivolts (mV)
        refractory_time: Time remaining in refractory period (ms)
        spike_history: List of spike times
        internal_state: Dictionary for model-specific state variables
    """
    membrane_potential: float = -70.0  # Resting potential (mV)
    refractory_time: float = 0.0
    spike_history: List[float] = field(default_factory=list)
    internal_state: Dict[str, float] = field(default_factory=dict)
    
    def reset(self):
        """Reset state to initial conditions."""
        self.membrane_potential = -70.0
        self.refractory_time = 0.0
        self.spike_history.clear()
        self.internal_state.clear()


@dataclass
class NeuronParameters:
    """
    Base parameters for neuron models.
    
    These parameters define the biophysical properties shared across
    different neuron models.
    """
    # Membrane properties
    resting_potential: float = -70.0  # mV
    membrane_capacitance: float = 1.0  # µF/cm²
    membrane_conductance: float = 0.1  # mS/cm²
    
    # Threshold and reset
    threshold: float = -55.0  # mV
    reset_potential: float = -70.0  # mV
    
    # Refractory period
    refractory_period: float = 2.0  # ms
    
    # Extracellular environment
    temperature: float = 36.5  # °C (body temperature)
    extracellular_na: float = 145.0  # mM
    extracellular_k: float = 3.5  # mM
    extracellular_ca: float = 1.5  # mM


class Neuron(ABC):
    """
    Abstract base class for all neuron models.
    
    This class defines the interface that all neuron implementations
    must follow. It provides common functionality for state management,
    spike detection, and parameter handling.
    
    Example:
        >>> class MyNeuron(Neuron):
        ...     def update(self, dt, current):
        ...         # Implement neuron dynamics
        ...         return self.state.membrane_potential
    """
    
    def __init__(
        self,
        parameters: Optional[NeuronParameters] = None,
        id: Optional[int] = None,
        position: Optional[Tuple[float, float, float]] = None,
        neuron_type: NeuronType = NeuronType.Custom
    ):
        """
        Initialize a neuron.
        
        Args:
            parameters: Neuron biophysical parameters
            id: Unique identifier for this neuron
            position: 3D position in space (x, y, z) in micrometers
            neuron_type: Type of neuron model
        """
        self.parameters = parameters or NeuronParameters()
        self.id = id if id is not None else id(self)
        self.position = position or (0.0, 0.0, 0.0)
        self.neuron_type = neuron_type
        self.state = NeuronState(membrane_potential=self.parameters.resting_potential)
        self._connected_synapses: List[str] = []
        
    @abstractmethod
    def update(self, dt: float, input_current: float) -> float:
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step in milliseconds
            input_current: Total input current in microamperes/cm²
            
        Returns:
            Current membrane potential in millivolts
        """
        pass
    
    @abstractmethod
    def compute_dynamics(self, dt: float, state: Dict[str, float], 
                        input_current: float) -> Dict[str, float]:
        """
        Compute state derivatives for the neuron model.
        
        Args:
            dt: Time step in milliseconds
            state: Current state variables
            input_current: Input current
            
        Returns:
            Dictionary of state derivatives
        """
        pass
    
    def spike(self, time: float):
        """
        Handle spike event.
        
        Args:
            time: Current simulation time in milliseconds
        """
        self.state.spike_history.append(time)
        self.state.refractory_time = self.parameters.refractory_period
        
    def is_refractory(self) -> bool:
        """Check if neuron is in refractory period."""
        return self.state.refractory_time > 0
    
    def check_threshold(self, potential: float) -> bool:
        """Check if membrane potential exceeds threshold."""
        return potential >= self.parameters.threshold
    
    def reset_state(self):
        """Reset neuron to initial conditions."""
        self.state.reset()
        self.state.membrane_potential = self.parameters.resting_potential
        
    def inject_current(self, current: float, duration: float):
        """
        Inject current for a specified duration.
        
        Args:
            current: Current amplitude in µA/cm²
            duration: Duration in milliseconds
        """
        # This should be overridden by subclasses if needed
        pass
    
    @property
    def firing_rate(self) -> float:
        """
        Compute instantaneous firing rate.
        
        Returns:
            Firing rate in Hz, or 0 if no spikes
        """
        if len(self.state.spike_history) < 2:
            return 0.0
        intervals = np.diff(self.state.spike_history[-100:])
        if len(intervals) == 0 or np.mean(intervals) == 0:
            return 0.0
        return 1000.0 / np.mean(intervals)
    
    def get_average_firing_rate(self, window_ms: float = 100.0) -> float:
        """
        Compute average firing rate over a time window.
        
        Args:
            window_ms: Time window in milliseconds
            
        Returns:
            Average firing rate in Hz
        """
        if len(self.state.spike_history) == 0:
            return 0.0
        spikes = [s for s in self.state.spike_history if s >= self.state.spike_history[-1] - window_ms]
        if len(spikes) < 2:
            return 0.0
        return len(spikes) / (window_ms / 1000.0)
    
    def add_synapse(self, synapse_id: str):
        """Register a synapse connection."""
        if synapse_id not in self._connected_synapses:
            self._connected_synapses.append(synapse_id)
            
    def get_connectivity_info(self) -> Dict[str, Any]:
        """Get information about neuron's synaptic connections."""
        return {
            "id": self.id,
            "type": self.neuron_type.value,
            "num_synapses": len(self._connected_synapses),
            "synapse_ids": self._connected_synapses.copy()
        }
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.id}, "
                f"type={self.neuron_type.value}, "
                f"V={self.state.membrane_potential:.2f}mV)")
