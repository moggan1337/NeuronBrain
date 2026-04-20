"""
Simulation utility functions for NeuronBrain.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_network(
    num_neurons: int,
    neuron_type: str = "izhikevich",
    topology: str = "sparse_random",
    connection_prob: float = 0.1,
    excitatory_ratio: float = 0.8
) -> Any:
    """
    Create a neural network with specified configuration.
    
    Args:
        num_neurons: Number of neurons
        neuron_type: Type of neuron model
        topology: Network topology
        connection_prob: Connection probability
        excitatory_ratio: Ratio of excitatory neurons
        
    Returns:
        Configured NeuralNetwork
    """
    from src.circuits.network import NeuralNetwork, NetworkTopology, NetworkConfig
    
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
        connection_probability=connection_prob,
        neuron_type=neuron_type,
        excitatory_ratio=excitatory_ratio,
    )
    
    return NeuralNetwork(config)


def run_simulation(
    network: Any,
    duration: float,
    dt: float = 0.1,
    input_current: float = 0.0,
    record_voltage: bool = False
) -> Dict[str, Any]:
    """
    Run a network simulation.
    
    Args:
        network: Neural network to simulate
        duration: Simulation duration (ms)
        dt: Time step (ms)
        input_current: Constant input current
        record_voltage: Whether to record membrane potentials
        
    Returns:
        Dictionary with simulation results
    """
    num_steps = int(duration / dt)
    
    results = {
        'spikes': [],
        'times': [],
        'neuron_ids': [],
        'mean_voltage': [],
    }
    
    if record_voltage:
        results['voltages'] = []
    
    for step in range(num_steps):
        # Inject current
        if input_current > 0:
            current = np.full(network.num_neurons, input_current)
            network.inject_current(current)
        
        # Step simulation
        network.step(dt)
        
        # Record spikes
        if network.spike_buffer:
            for time, neuron_id in network.spike_buffer[-100:]:  # Last 100 spikes
                if abs(time - network.time) < dt:
                    results['spikes'].append((time, neuron_id))
                    results['times'].append(time)
                    results['neuron_ids'].append(neuron_id)
        
        # Record mean voltage
        mean_v = np.mean([n.state.membrane_potential for n in network.neurons])
        results['mean_voltage'].append(mean_v)
        
        if record_voltage:
            voltages = [n.state.membrane_potential for n in network.neurons]
            results['voltages'].append(voltages)
    
    return results


def simulate_poisson(
    rate: float,
    duration: float,
    dt: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate Poisson spike train.
    
    Args:
        rate: Firing rate (Hz)
        duration: Duration (ms)
        dt: Time step (ms)
        seed: Random seed
        
    Returns:
        Binary spike train
    """
    rng = np.random.RandomState(seed)
    
    num_steps = int(duration / dt)
    spike_train = np.zeros(num_steps)
    
    # Probability of spike per timestep
    prob = 1 - np.exp(-rate * dt / 1000)
    
    spikes = rng.random(num_steps) < prob
    return spikes.astype(float)


def compute_connectivity_matrix(
    network: Any,
    weight_threshold: float = 0.01
) -> np.ndarray:
    """
    Compute connectivity matrix from network.
    
    Args:
        network: Neural network
        weight_threshold: Threshold for connection
        
    Returns:
        Connectivity matrix
    """
    n = network.num_neurons
    conn = np.zeros((n, n))
    
    for synapse in network.synapses:
        if abs(synapse.weight) >= weight_threshold:
            conn[synapse.post_id, synapse.pre_id] = synapse.weight
    
    return conn


def inject_poisson_input(
    network: Any,
    rates: np.ndarray,
    dt: float
):
    """
    Inject Poisson input into network.
    
    Args:
        network: Neural network
        rates: Firing rates for each neuron
        dt: Time step
    """
    for i, rate in enumerate(rates):
        if rate > 0:
            prob = 1 - np.exp(-rate * dt / 1000)
            if np.random.random() < prob:
                network.current_input[i] += 1.0


def compute_network_statistics(network: Any) -> Dict[str, float]:
    """
    Compute statistics for a network.
    
    Args:
        network: Neural network
        
    Returns:
        Dictionary of statistics
    """
    rates = network.get_firing_rates()
    
    return {
        'num_neurons': network.num_neurons,
        'num_synapses': len(network.synapses),
        'mean_rate': np.mean(rates),
        'std_rate': np.std(rates),
        'max_rate': np.max(rates),
        'min_rate': np.min(rates),
        'active_fraction': np.mean(rates > 0),
        'total_spikes': len(network.spike_buffer),
        'connection_density': len(network.synapses) / (network.num_neurons ** 2),
    }


def run_batch_simulation(
    num_networks: int,
    params: Dict[str, Any],
    duration: float,
    dt: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Run multiple simulations with different parameters.
    
    Args:
        num_networks: Number of networks to simulate
        params: Parameter variations
        duration: Simulation duration
        dt: Time step
        
    Returns:
        List of results for each network
    """
    results = []
    
    for i in range(num_networks):
        # Create network with parameters
        network = create_network(**params)
        
        # Run simulation
        result = run_simulation(network, duration, dt)
        result['network_id'] = i
        
        results.append(result)
    
    return results
