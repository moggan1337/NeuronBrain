"""
Neural Network Implementation for NeuronBrain.

This module provides a comprehensive neural network simulation framework
that supports various neuron types, synapse configurations, and network
topologies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random


class NetworkTopology(Enum):
    """Network topology types."""
    FULLY_CONNECTED = "fully_connected"
    SPARSE_RANDOM = "sparse_random"
    SMALL_WORLD = "small_world"
    SPATIAL = "spatial"
    LAYERED = "layered"
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    HOPFIELD = "hopfield"


@dataclass
class NetworkConfig:
    """Configuration for a neural network."""
    num_neurons: int = 100
    topology: NetworkTopology = NetworkTopology.SPARSE_RANDOM
    connection_probability: float = 0.1
    
    # Neuron parameters
    neuron_type: str = "izhikevich"
    neuron_params: Dict[str, Any] = field(default_factory=dict)
    
    # Synapse parameters
    synapse_type: str = "ampa"
    synapse_params: Dict[str, Any] = field(default_factory=dict)
    
    # STDP
    stdp_enabled: bool = False
    stdp_params: Dict[str, Any] = field(default_factory=dict)
    
    # Connectivity
    excitatory_ratio: float = 0.8
    self_connections: bool = False
    
    # Spatial parameters (for spatial networks)
    spatial_layout: str = "random"  # random, grid, gaussian
    spatial_scale: float = 100.0   # µm


class NeuralNetwork:
    """
    Neural network simulation framework.
    
    This class provides a complete framework for simulating networks
    of spiking neurons with configurable connectivity and plasticity.
    
    Features:
    - Multiple neuron models (LIF, Izhikevich, Hodgkin-Huxley)
    - Multiple synapse types (AMPA, NMDA, GABA_A, GABA_B, gap junctions)
    - STDP learning
    - Various network topologies
    - Efficient vectorized simulations
    - Spike recording and analysis
    
    Example:
        >>> config = NetworkConfig(num_neurons=100, topology=NetworkTopology.SPARSE_RANDOM)
        >>> network = NeuralNetwork(config)
        >>> 
        >>> # Add external input
        >>> network.inject_current(bg_current)
        >>> 
        >>> # Run simulation
        >>> for t in range(10000):
        ...     network.step(0.1)  # 0.1 ms time step
        ... 
        >>> # Get results
        >>> spikes = network.get_spikes()
        >>> rates = network.get_firing_rates()
    """
    
    def __init__(self, config: NetworkConfig):
        """
        Initialize neural network.
        
        Args:
            config: Network configuration
        """
        self.config = config
        self.num_neurons = config.num_neurons
        
        # Initialize components
        self.neurons: List[Any] = []
        self.synapses: List[Any] = []
        self.neuron_to_synapses: Dict[int, List[int]] = {}
        
        # Create neurons
        self._create_neurons()
        
        # Create connectivity
        self._create_connectivity()
        
        # Initialize STDP if enabled
        self._initialize_stdp()
        
        # State variables
        self.time: float = 0.0
        self.spike_buffer: List[Tuple[float, int]] = []  # (time, neuron_id)
        self.current_input: np.ndarray = np.zeros(self.num_neurons)
        
        # Activity tracking
        self.spike_history: List[List[float]] = [[] for _ in range(self.num_neurons)]
        self.membrane_history: List[List[float]] = [[] for _ in range(self.num_neurons)]
        
    def _create_neurons(self):
        """Create neurons based on configuration."""
        from ..models import LeakyIntegrateAndFire, IzhikevichNeuron, HodgkinHuxleyNeuron
        from ..models.izhikevich import IzhikevichParameters, IzhikevichNeuronType
        from ..models.lif import LIFParameters
        
        neuron_type = self.config.neuron_type.lower()
        
        for i in range(self.num_neurons):
            # Determine if excitatory or inhibitory
            is_excitatory = i < int(self.num_neurons * self.config.excitatory_ratio)
            
            if neuron_type == "lif":
                params = LIFParameters(**self.config.neuron_params)
                if not is_excitatory:
                    params.E_syn = -70.0  # Inhibitory
                neuron = LeakyIntegrateAndFire(params, id=i)
                
            elif neuron_type == "izhikevich":
                if is_excitatory:
                    params = IzhikevichParameters(**self.config.neuron_params)
                    neuron = IzhikevichNeuron(params, id=i)
                else:
                    # Fast spiking for inhibitory
                    neuron = IzhikevichNeuron.from_preset(
                        IzhikevichNeuronType.FastSpiking, 
                        id=i
                    )
                    
            elif neuron_type == "hodgkin_huxley":
                from ..models.hodgkin_huxley import HodgkinHuxleyParameters
                params = HodgkinHuxleyParameters(**self.config.neuron_params)
                neuron = HodgkinHuxleyNeuron(params, id=i)
            else:
                # Default to Izhikevich
                params = IzhikevichParameters(**self.config.neuron_params)
                neuron = IzhikevichNeuron(params, id=i)
            
            self.neurons.append(neuron)
            self.neuron_to_synapses[i] = []
    
    def _create_connectivity(self):
        """Create synaptic connections based on topology."""
        from ..synapses import SynapseFactory, SynapticReceptor
        
        synapse_type = self.config.synapse_type.lower()
        
        if self.config.topology == NetworkTopology.FULLY_CONNECTED:
            self._create_fully_connected(synapse_type)
            
        elif self.config.topology == NetworkTopology.SPARSE_RANDOM:
            self._create_sparse_random(synapse_type)
            
        elif self.config.topology == NetworkTopology.SPATIAL:
            self._create_spatial(synapse_type)
            
        elif self.config.topology == NetworkTopology.LAYERED:
            self._create_layered(synapse_type)
            
        elif self.config.topology == NetworkTopology.SMALL_WORLD:
            self._create_small_world(synapse_type)
            
        else:
            self._create_sparse_random(synapse_type)
    
    def _create_fully_connected(self, synapse_type: str):
        """Create fully connected network."""
        from ..synapses import SynapseFactory, SynapticReceptor
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j and not self.config.self_connections:
                    continue
                
                is_excitatory = j < int(self.num_neurons * self.config.excitatory_ratio)
                
                if synapse_type == "ampa" or (synapse_type == "exc"):
                    synapse = SynapseFactory.create(
                        "EXCITATORY_AMPA" if is_excitatory else "INHIBITORY_GABA_A",
                        pre_id=j,
                        post_id=i,
                        weight=1.0 if is_excitatory else -1.0
                    )
                else:
                    synapse = SynapseFactory.create(synapse_type, pre_id=j, post_id=i)
                
                self.synapses.append(synapse)
                self.neuron_to_synapses[i].append(len(self.synapses) - 1)
    
    def _create_sparse_random(self, synapse_type: str):
        """Create randomly connected sparse network."""
        from ..synapses import SynapseFactory
        
        prob = self.config.connection_probability
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j and not self.config.self_connections:
                    continue
                
                if np.random.random() > prob:
                    continue
                
                is_excitatory = j < int(self.num_neurons * self.config.excitatory_ratio)
                
                # Weight based on neuron type
                weight = np.random.uniform(0.5, 1.5) if is_excitatory else np.random.uniform(-1.5, -0.5)
                
                synapse = SynapseFactory.create(
                    "EXCITATORY_AMPA" if is_excitatory else "INHIBITORY_GABA_A",
                    pre_id=j,
                    post_id=i,
                    weight=weight
                )
                
                self.synapses.append(synapse)
                self.neuron_to_synapses[i].append(len(self.synapses) - 1)
    
    def _create_spatial(self, synapse_type: str):
        """Create spatially structured network."""
        from ..synapses import SynapseFactory
        
        # Random positions
        positions = np.random.uniform(0, self.config.spatial_scale, (self.num_neurons, 3))
        
        prob = self.config.connection_probability
        spatial_scale = self.config.spatial_scale / 2
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j and not self.config.self_connections:
                    continue
                
                # Distance-dependent probability
                dist = np.linalg.norm(positions[i] - positions[j])
                conn_prob = prob * np.exp(-dist**2 / spatial_scale**2)
                
                if np.random.random() > conn_prob:
                    continue
                
                is_excitatory = j < int(self.num_neurons * self.config.excitatory_ratio)
                
                synapse = SynapseFactory.create(
                    "EXCITATORY_AMPA" if is_excitatory else "INHIBITORY_GABA_A",
                    pre_id=j,
                    post_id=i,
                    weight=1.0 if is_excitatory else -1.0
                )
                
                self.synapses.append(synapse)
                self.neuron_to_synapses[i].append(len(self.synapses) - 1)
    
    def _create_layered(self, synapse_type: str):
        """Create layered network."""
        from ..synapses import SynapseFactory
        
        num_layers = 4
        neurons_per_layer = self.num_neurons // num_layers
        
        for i in range(self.num_neurons):
            layer_i = i // neurons_per_layer
            
            # Within-layer connections
            for j in range(layer_i * neurons_per_layer, (layer_i + 1) * neurons_per_layer):
                if i == j and not self.config.self_connections:
                    continue
                if np.random.random() > 0.1:
                    continue
                
                is_excitatory = True
                synapse = SynapseFactory.create(
                    "EXCITATORY_AMPA",
                    pre_id=j,
                    post_id=i,
                    weight=0.8
                )
                self.synapses.append(synapse)
                self.neuron_to_synapses[i].append(len(self.synapses) - 1)
            
            # Between-layer connections (feedforward)
            for layer_j in range(layer_i + 1, num_layers):
                for j in range(layer_j * neurons_per_layer, (layer_j + 1) * neurons_per_layer):
                    if np.random.random() > 0.05:
                        continue
                    
                    synapse = SynapseFactory.create(
                        "EXCITATORY_AMPA",
                        pre_id=j,
                        post_id=i,
                        weight=1.0
                    )
                    self.synapses.append(synapse)
                    self.neuron_to_synapses[i].append(len(self.synapses) - 1)
    
    def _create_small_world(self, synapse_type: str):
        """Create small-world network (Watts-Strogatz model)."""
        from ..synapses import SynapseFactory
        
        n = self.num_neurons
        k = max(2, int(n * self.config.connection_probability * 2))  # Average degree
        p_rewire = 0.1  # Rewiring probability
        
        # Start with ring lattice
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                j_plus = (i + j) % n
                edges.append((i, j_plus))
        
        # Rewire edges with probability p
        for idx, (i, j) in enumerate(edges):
            if np.random.random() < p_rewire:
                new_j = np.random.randint(0, n)
                while new_j == i or (i, new_j) in edges or (new_j, i) in edges:
                    new_j = np.random.randint(0, n)
                edges[idx] = (i, new_j)
        
        # Create synapses
        for i, j in edges:
            is_excitatory = j < int(self.num_neurons * self.config.excitatory_ratio)
            
            synapse = SynapseFactory.create(
                "EXCITATORY_AMPA" if is_excitatory else "INHIBITORY_GABA_A",
                pre_id=j,
                post_id=i,
                weight=1.0 if is_excitatory else -1.0
            )
            self.synapses.append(synapse)
            self.neuron_to_synapses[i].append(len(self.synapses) - 1)
    
    def _initialize_stdp(self):
        """Initialize STDP learning if enabled."""
        if self.config.stdp_enabled:
            from ..learning import STDPFactory
            self.stdp = STDPFactory.create(**self.config.stdp_params)
        else:
            self.stdp = None
    
    def step(self, dt: float) -> Dict[str, Any]:
        """
        Simulate one time step.
        
        Args:
            dt: Time step in milliseconds
            
        Returns:
            Dictionary with step statistics
        """
        self.time += dt
        
        # Collect current inputs
        total_current = self.current_input.copy()
        
        # Update synapses and collect synaptic currents
        for i, synapse in enumerate(self.synapses):
            pre_neuron = self.neurons[synapse.pre_id]
            post_neuron = self.neurons[synapse.post_id]
            
            # Notify synapse of presynaptic spikes
            if pre_neuron.state.spike_history:
                last_spike = pre_neuron.state.spike_history[-1]
                if self.time - last_spike < dt:
                    synapse.presynaptic_spike(last_spike)
            
            # Update synapse
            current = synapse.update(dt, post_neuron.state.membrane_potential, self.time)
            
            # Add to postsynaptic current
            total_current[synapse.post_id] += current
        
        # Update neurons
        spikes_this_step = []
        for i, neuron in enumerate(self.neurons):
            old_spike_count = len(neuron.state.spike_history)
            V = neuron.update(dt, total_current[i])
            
            # Check for new spike
            if len(neuron.state.spike_history) > old_spike_count:
                spikes_this_step.append(i)
                self.spike_buffer.append((self.time, i))
            
            # Record state
            self.spike_history[i].append(self.time)
            self.membrane_history[i].append(V)
        
        # Apply STDP if enabled
        if self.stdp and spikes_this_step:
            for i in spikes_this_step:
                self.stdp.record_postsynaptic_spike(i, self.time)
                
                # Record presynaptic spikes for synapses to this neuron
                for syn_idx in self.neuron_to_synapses[i]:
                    synapse = self.synapses[syn_idx]
                    self.stdp.record_presynaptic_spike(synapse.pre_id, self.time)
        
        return {
            'time': self.time,
            'num_spikes': len(spikes_this_step),
            'spikes': spikes_this_step,
            'mean_v': np.mean([n.state.membrane_potential for n in self.neurons])
        }
    
    def inject_current(self, current: np.ndarray):
        """
        Inject current into neurons.
        
        Args:
            current: Array of currents (one per neuron)
        """
        self.current_input = np.array(current)
    
    def inject_current_pulse(
        self, 
        neuron_ids: List[int], 
        amplitude: float,
        duration: float,
        start_time: Optional[float] = None
    ):
        """
        Inject current pulse into specified neurons.
        
        Args:
            neuron_ids: List of neuron IDs
            amplitude: Current amplitude
            duration: Pulse duration (ms)
            start_time: Start time (uses current time if None)
        """
        start = start_time if start_time is not None else self.time
        
        for i in neuron_ids:
            self.current_input[i] += amplitude
    
    def get_spikes(self) -> List[Tuple[float, int]]:
        """Get all recorded spikes."""
        return self.spike_buffer.copy()
    
    def get_firing_rates(self) -> np.ndarray:
        """Get average firing rates for all neurons."""
        rates = np.zeros(self.num_neurons)
        
        for i, history in enumerate(self.spike_history):
            if len(history) < 2:
                rates[i] = 0.0
            else:
                # Use last 1000ms of activity
                recent = [t for t in history if t > self.time - 1000]
                if len(recent) >= 2:
                    rate = len(recent) / (recent[-1] - recent[0]) * 1000
                    rates[i] = rate
        
        return rates
    
    def get_raster_plot(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spike raster data.
        
        Returns:
            Tuple of (times, neuron_ids)
        """
        times = np.array([s[0] for s in self.spike_buffer])
        ids = np.array([s[1] for s in self.spike_buffer])
        return times, ids
    
    def get_connectivity_matrix(self) -> np.ndarray:
        """
        Get connectivity matrix.
        
        Returns:
            N x N adjacency matrix
        """
        conn = np.zeros((self.num_neurons, self.num_neurons))
        
        for synapse in self.synapses:
            conn[synapse.post_id, synapse.pre_id] = synapse.weight
        
        return conn
    
    def reset(self):
        """Reset network state."""
        self.time = 0.0
        self.spike_buffer.clear()
        
        for neuron in self.neurons:
            neuron.reset_state()
        
        for synapse in self.synapses:
            synapse.reset()
        
        for i in range(self.num_neurons):
            self.spike_history[i].clear()
            self.membrane_history[i].clear()
        
        self.current_input = np.zeros(self.num_neurons)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        rates = self.get_firing_rates()
        
        return {
            'num_neurons': self.num_neurons,
            'num_synapses': len(self.synapses),
            'mean_firing_rate': np.mean(rates),
            'std_firing_rate': np.std(rates),
            'total_spikes': len(self.spike_buffer),
            'active_neurons': np.sum(rates > 0),
            'mean_membrane_potential': np.mean([n.state.membrane_potential for n in self.neurons]),
        }
