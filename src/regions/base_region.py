"""
Base Brain Region class.

This module provides the base class for brain region models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RegionConfig:
    """Configuration for a brain region."""
    name: str = "region"
    num_neurons: int = 100
    neuron_type: str = "izhikevich"
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class InputPort:
    """Represents an input connection to a brain region."""
    
    def __init__(self, name: str, target_layer: str, weight: float = 1.0):
        self.name = name
        self.target_layer = target_layer
        self.weight = weight
        self.activity: float = 0.0
        self.history: List[float] = []
    
    def inject(self, activity: float):
        """Inject activity into this port."""
        self.activity = activity * self.weight
        self.history.append(self.activity)
    
    def reset(self):
        """Reset port state."""
        self.activity = 0.0


class OutputPort:
    """Represents an output connection from a brain region."""
    
    def __init__(self, name: str, source_layer: str):
        self.name = name
        self.source_layer = source_layer
        self.activity: float = 0.0
        self.history: List[float] = []
    
    def update(self, activity: float):
        """Update output activity."""
        self.activity = activity
        self.history.append(activity)
    
    def reset(self):
        """Reset port state."""
        self.activity = 0.0


class BrainRegion(ABC):
    """
    Abstract base class for brain region models.
    
    Brain regions are functional units that can receive inputs,
    process information, and produce outputs. Each region has:
    - Input ports for receiving signals
    - Output ports for sending signals
    - Internal neural population(s)
    - Region-specific dynamics
    
    Subclasses should implement:
    - _initialize_components(): Create neurons and connections
    - _compute_dynamics(): Compute region dynamics
    - _update_outputs(): Update output activities
    
    Example:
        >>> class MyRegion(BrainRegion):
        ...     def _initialize_components(self):
        ...         self.neurons = create_neurons(100)
        ...     def _compute_dynamics(self, dt):
        ...         pass
        ...     def _update_outputs(self):
        ...         self.output_ports['main'].update(mean_rate)
    """
    
    def __init__(self, config: RegionConfig):
        """
        Initialize brain region.
        
        Args:
            config: Region configuration
        """
        self.config = config
        self.name = config.name
        
        # Components
        self.neurons: List[Any] = []
        self.synapses: List[Any] = []
        
        # Input and output ports
        self.input_ports: Dict[str, InputPort] = {}
        self.output_ports: Dict[str, OutputPort] = {}
        
        # Internal state
        self.time: float = 0.0
        self.state: Dict[str, Any] = {}
        
        # Statistics
        self.statistics: Dict[str, List[float]] = {}
        
        # Initialize region-specific components
        self._initialize_components()
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize neurons, synapses, and ports. Override in subclass."""
        pass
    
    @abstractmethod
    def _compute_dynamics(self, dt: float):
        """Compute internal dynamics. Override in subclass."""
        pass
    
    def _update_outputs(self):
        """Update output activities. Override in subclass if needed."""
        pass
    
    def add_input_port(self, name: str, target_layer: str, weight: float = 1.0) -> InputPort:
        """
        Add an input port.
        
        Args:
            name: Port name
            target_layer: Target layer/region
            weight: Connection weight
            
        Returns:
            Created InputPort
        """
        port = InputPort(name, target_layer, weight)
        self.input_ports[name] = port
        return port
    
    def add_output_port(self, name: str, source_layer: str) -> OutputPort:
        """
        Add an output port.
        
        Args:
            name: Port name
            source_layer: Source layer/region
            
        Returns:
            Created OutputPort
        """
        port = OutputPort(name, source_layer)
        self.output_ports[name] = port
        return port
    
    def inject_input(self, port_name: str, activity: float):
        """
        Inject activity into an input port.
        
        Args:
            port_name: Name of input port
            activity: Activity to inject
        """
        if port_name in self.input_ports:
            self.input_ports[port_name].inject(activity)
        else:
            raise ValueError(f"Unknown input port: {port_name}")
    
    def get_output(self, port_name: str) -> float:
        """
        Get output from a port.
        
        Args:
            port_name: Name of output port
            
        Returns:
            Current output activity
        """
        if port_name in self.output_ports:
            return self.output_ports[port_name].activity
        else:
            raise ValueError(f"Unknown output port: {port_name}")
    
    def step(self, dt: float) -> Dict[str, Any]:
        """
        Simulate one time step.
        
        Args:
            dt: Time step in milliseconds
            
        Returns:
            Dictionary with step information
        """
        self.time += dt
        
        # Compute dynamics
        self._compute_dynamics(dt)
        
        # Update outputs
        self._update_outputs()
        
        # Record statistics
        self._record_statistics()
        
        return {
            'time': self.time,
            'num_neurons': len(self.neurons),
            'mean_activity': self._get_mean_activity(),
        }
    
    def _get_mean_activity(self) -> float:
        """Get mean neural activity."""
        if not self.neurons:
            return 0.0
        return np.mean([n.state.membrane_potential for n in self.neurons])
    
    def _record_statistics(self):
        """Record statistics for later analysis."""
        rates = self.get_firing_rates()
        
        self.statistics.setdefault('time', []).append(self.time)
        self.statistics.setdefault('mean_rate', []).append(np.mean(rates))
        self.statistics.setdefault('active_fraction', []).append(np.mean(rates > 0))
        
        if hasattr(self, '_record_custom_statistics'):
            self._record_custom_statistics()
    
    def get_firing_rates(self) -> np.ndarray:
        """Get firing rates for all neurons."""
        rates = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            rates[i] = neuron.get_average_firing_rate(window_ms=100)
        return rates
    
    def get_activity(self) -> Dict[str, float]:
        """
        Get current activity summary.
        
        Returns:
            Dictionary of activity metrics
        """
        rates = self.get_firing_rates()
        
        return {
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'max_rate': np.max(rates),
            'active_fraction': np.mean(rates > 0),
            'mean_membrane_potential': self._get_mean_activity(),
        }
    
    def reset(self):
        """Reset region state."""
        self.time = 0.0
        
        for neuron in self.neurons:
            neuron.reset_state()
        
        for port in self.input_ports.values():
            port.reset()
        
        for port in self.output_ports.values():
            port.reset()
        
        self.state.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', neurons={len(self.neurons)})"
