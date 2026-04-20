"""
Main Simulator Module for NeuronBrain.

This module provides the main simulation engine that coordinates
all components of the neural simulation framework.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time as time_module

from .circuits.network import NeuralNetwork, NetworkConfig, NetworkTopology
from .models import LeakyIntegrateAndFire, IzhikevichNeuron, HodgkinHuxleyNeuron
from .synapses import ChemicalSynapse, ElectricalSynapse, SynapseFactory
from .learning import STDP, STDPFactory
from .regions import Hippocampus, Thalamus, BasalGanglia
from .coding import RateCoder, PopulationCoder


@dataclass
class SimulatorConfig:
    """Configuration for the simulator."""
    dt: float = 0.1              # Time step (ms)
    duration: float = 1000.0     # Simulation duration (ms)
    
    # Network configuration
    num_neurons: int = 100
    neuron_type: str = "izhikevich"
    topology: str = "sparse_random"
    connection_probability: float = 0.1
    
    # Input
    input_type: str = "constant"  # constant, poisson, signal
    input_current: float = 10.0
    
    # Recording
    record_voltage: bool = False
    record_spikes: bool = True
    record_weights: bool = False
    
    # Learning
    stdp_enabled: bool = False
    plasticity_enabled: bool = True
    
    # Performance
    parallel: bool = False
    batch_size: int = 100


class SimulationResult:
    """Container for simulation results."""
    
    def __init__(self):
        self.spikes: List[tuple] = []  # (time, neuron_id)
        self.voltages: List[List[float]] = []
        self.weights: List[Dict] = []
        self.times: List[float] = []
        self.rates: Optional[np.ndarray] = None
        self.statistics: Dict[str, Any] = {}
        self.simulation_time: float = 0.0
        self.num_steps: int = 0
    
    def add_spike(self, time: float, neuron_id: int):
        """Add a spike to the record."""
        self.spikes.append((time, neuron_id))
    
    def add_voltage_snapshot(self, voltages: List[float], time: float):
        """Add voltage snapshot."""
        self.voltages.append(voltages)
        self.times.append(time)
    
    def finalize(self, network: NeuralNetwork):
        """Finalize results after simulation."""
        if network.spike_buffer:
            self.spikes = network.spike_buffer.copy()
        self.rates = network.get_firing_rates()
        self.statistics = network.get_statistics()
        self.num_steps = len(self.times) if self.times else 0


class Simulator:
    """
    Main neural network simulator.
    
    This class provides a unified interface for running neural
    simulations with various configurations.
    
    Features:
    - Multiple neuron models
    - Flexible connectivity
    - STDP learning
    - Multiple brain regions
    - Recording and analysis
    - Performance optimization
    
    Example:
        >>> config = SimulatorConfig(num_neurons=100, duration=1000)
        >>> sim = Simulator(config)
        >>> 
        >>> # Add components
        >>> sim.create_network()
        >>> sim.add_input(current=10.0)
        >>> 
        >>> # Run
        >>> result = sim.run()
        >>> 
        >>> # Analyze
        >>> print(f"Mean firing rate: {np.mean(result.rates):.2f} Hz")
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        """
        Initialize simulator.
        
        Args:
            config: Simulator configuration
        """
        self.config = config or SimulatorConfig()
        
        # Components
        self.network: Optional[NeuralNetwork] = None
        self.regions: Dict[str, Any] = {}
        self.coders: Dict[str, Any] = {}
        
        # Results
        self.result = SimulationResult()
        
        # State
        self.time: float = 0.0
        self.running: bool = False
        
        # Performance tracking
        self._step_times: List[float] = []
    
    def create_network(
        self,
        num_neurons: Optional[int] = None,
        neuron_type: Optional[str] = None,
        topology: Optional[str] = None,
        **kwargs
    ):
        """
        Create a neural network.
        
        Args:
            num_neurons: Number of neurons
            neuron_type: Type of neuron model
            topology: Network topology
            **kwargs: Additional network parameters
        """
        if num_neurons is None:
            num_neurons = self.config.num_neurons
        if neuron_type is None:
            neuron_type = self.config.neuron_type
        if topology is None:
            topology = self.config.topology
            
        topology_map = {
            "fully_connected": NetworkTopology.FULLY_CONNECTED,
            "sparse_random": NetworkTopology.SPARSE_RANDOM,
            "small_world": NetworkTopology.SMALL_WORLD,
            "spatial": NetworkTopology.SPATIAL,
            "layered": NetworkTopology.LAYERED,
        }
        
        config = NetworkConfig(
            num_neurons=num_neurons,
            topology=topology_map.get(topology, NetworkTopology.SPARSE_RANDOM),
            connection_probability=self.config.connection_probability,
            neuron_type=neuron_type,
            **kwargs
        )
        
        self.network = NeuralNetwork(config)
        
        # Initialize STDP if enabled
        if self.config.stdp_enabled:
            self.network._initialize_stdp()
    
    def add_region(self, name: str, region: Any):
        """
        Add a brain region to the simulation.
        
        Args:
            name: Region name
            region: BrainRegion object
        """
        self.regions[name] = region
        if hasattr(region, 'network'):
            region.network = self.network
    
    def create_hippocampus(self) -> Hippocampus:
        """Create and add hippocampus."""
        from .regions.hippocampus import Hippocampus, HippocampusConfig
        config = HippocampusConfig(name="hippocampus")
        region = Hippocampus(config)
        self.add_region("hippocampus", region)
        return region
    
    def create_thalamus(self):
        """Create and add thalamus."""
        from .regions.thalamus_model import Thalamus, ThalamusConfig
        config = ThalamusConfig(name="thalamus")
        region = Thalamus(config)
        self.add_region("thalamus", region)
        return region
    
    def create_basal_ganglia(self):
        """Create and add basal ganglia."""
        from .regions.basal_ganglia import BasalGanglia, BasalGangliaConfig
        config = BasalGangliaConfig(name="basal_ganglia")
        region = BasalGanglia(config)
        self.add_region("basal_ganglia", region)
        return region
    
    def add_input(self, current: Optional[float] = None, pattern: Optional[np.ndarray] = None):
        """
        Add input to the network.
        
        Args:
            current: Constant current value
            pattern: Time-varying current pattern
        """
        if self.network is None:
            return
            
        if current is not None:
            self.network.inject_current(np.full(self.config.num_neurons, current))
        elif pattern is not None:
            # Will be applied in step function
            self._input_pattern = pattern
            self._pattern_index = 0
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Step information
        """
        start_time = time_module.time()
        
        # Get input
        if hasattr(self, '_input_pattern') and self._input_pattern is not None:
            if self._pattern_index < len(self._input_pattern):
                self.network.inject_current(self._input_pattern[self._pattern_index])
                self._pattern_index += 1
        
        # Step network
        step_info = self.network.step(self.config.dt)
        
        # Step regions
        for region in self.regions.values():
            if hasattr(region, 'step'):
                region.step(self.config.dt)
        
        # Record
        if self.config.record_spikes and step_info['spikes']:
            for neuron_id in step_info['spikes']:
                self.result.add_spike(self.time, neuron_id)
        
        if self.config.record_voltage:
            voltages = [n.state.membrane_potential for n in self.network.neurons]
            self.result.add_voltage_snapshot(voltages, self.time)
        
        self.time += self.config.dt
        self._step_times.append(time_module.time() - start_time)
        
        return step_info
    
    def run(
        self,
        duration: Optional[float] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> SimulationResult:
        """
        Run the simulation.
        
        Args:
            duration: Simulation duration (ms)
            verbose: Print progress
            progress_callback: Optional callback function
            
        Returns:
            SimulationResult object
        """
        if self.network is None:
            self.create_network()
        
        if duration is None:
            duration = self.config.duration
        
        start_time = time_module.time()
        num_steps = int(duration / self.config.dt)
        
        self.running = True
        self.time = 0.0
        
        if verbose:
            print(f"Running simulation: {duration:.1f} ms, {num_steps} steps")
        
        for step in range(num_steps):
            self.step()
            
            if verbose and step % (num_steps // 10 + 1) == 0:
                progress = step / num_steps * 100
                print(f"  Progress: {progress:.1f}% (t={self.time:.1f} ms)")
            
            if progress_callback:
                progress_callback(step / num_steps, self.time)
            
            if not self.running:
                break
        
        # Finalize results
        self.result.finalize(self.network)
        self.result.simulation_time = time_module.time() - start_time
        
        if verbose:
            self._print_summary()
        
        return self.result
    
    def _print_summary(self):
        """Print simulation summary."""
        stats = self.result.statistics
        
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"Duration: {self.time:.1f} ms")
        print(f"Steps: {self.result.num_steps}")
        print(f"Simulation time: {self.result.simulation_time:.2f} s")
        print(f"Real-time factor: {self.time / 1000 / self.result.simulation_time:.1f}x")
        print()
        print("Network Statistics:")
        print(f"  Neurons: {stats.get('num_neurons', 0)}")
        print(f"  Synapses: {stats.get('num_synapses', 0)}")
        print(f"  Total spikes: {stats.get('total_spikes', 0)}")
        print(f"  Mean firing rate: {stats.get('mean_firing_rate', 0):.2f} Hz")
        print(f"  Active neurons: {stats.get('active_neurons', 0)}")
        print("=" * 50)
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
    
    def reset(self):
        """Reset the simulation."""
        if self.network:
            self.network.reset()
        self.time = 0.0
        self.result = SimulationResult()
        self._step_times.clear()
        self.running = False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._step_times:
            return {}
        
        return {
            'mean_step_time': np.mean(self._step_times) * 1000,  # ms
            'std_step_time': np.std(self._step_times) * 1000,
            'min_step_time': np.min(self._step_times) * 1000,
            'max_step_time': np.max(self._step_times) * 1000,
            'total_time': np.sum(self._step_times),
        }
    
    def __repr__(self) -> str:
        return (f"Simulator(neurons={self.config.num_neurons}, "
                f"duration={self.config.duration}ms, "
                f"dt={self.config.dt}ms)")


# Convenience function
def run_simulation(
    num_neurons: int = 100,
    duration: float = 1000.0,
    dt: float = 0.1,
    neuron_type: str = "izhikevich",
    input_current: float = 10.0,
    **kwargs
) -> SimulationResult:
    """
    Run a quick simulation with default settings.
    
    Args:
        num_neurons: Number of neurons
        duration: Duration in ms
        dt: Time step in ms
        neuron_type: Neuron model type
        input_current: Input current
        **kwargs: Additional simulator parameters
        
    Returns:
        SimulationResult
    """
    config = SimulatorConfig(
        num_neurons=num_neurons,
        duration=duration,
        dt=dt,
        neuron_type=neuron_type,
        input_current=input_current,
        **kwargs
    )
    
    sim = Simulator(config)
    sim.create_network()
    sim.add_input(current=input_current)
    
    return sim.run(verbose=False)
