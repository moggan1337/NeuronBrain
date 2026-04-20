"""
Performance benchmarks for NeuronBrain.

Run with: python benchmarks/benchmark_simulation.py
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import Simulator, SimulatorConfig
from src.circuits.network import NeuralNetwork, NetworkConfig, NetworkTopology
from src.models.izhikevich import IzhikevichNeuron
from src.synapses import ChemicalSynapse


def benchmark_neuron_models(num_neurons=1000, duration=100.0):
    """Benchmark different neuron models."""
    print("\n" + "=" * 60)
    print("NEURON MODEL BENCHMARKS")
    print("=" * 60)
    
    models = {
        'LIF': 'lif',
        'Izhikevich': 'izhikevich',
        'Hodgkin-Huxley': 'hodgkin_huxley'
    }
    
    results = {}
    
    for name, model_type in models.items():
        config = SimulatorConfig(
            num_neurons=num_neurons,
            duration=duration,
            dt=0.1,
            neuron_type=model_type,
            input_current=10.0,
            record_spikes=False
        )
        
        sim = Simulator(config)
        sim.create_network()
        
        start = time.time()
        sim.run(verbose=False)
        elapsed = time.time() - start
        
        results[name] = {
            'time': elapsed,
            'steps_per_sec': num_neurons * duration / 0.1 / elapsed
        }
        
        print(f"{name}: {elapsed:.3f}s ({results[name]['steps_per_sec']:.0f} steps/s)")
    
    return results


def benchmark_network_sizes():
    """Benchmark different network sizes."""
    print("\n" + "=" * 60)
    print("NETWORK SIZE BENCHMARKS")
    print("=" * 60)
    
    sizes = [50, 100, 200, 500, 1000]
    duration = 100.0
    
    results = {}
    
    for n in sizes:
        config = SimulatorConfig(
            num_neurons=n,
            duration=duration,
            dt=0.1,
            input_current=5.0
        )
        
        sim = Simulator(config)
        sim.create_network()
        
        start = time.time()
        sim.run(verbose=False)
        elapsed = time.time() - start
        
        results[n] = {
            'time': elapsed,
            'neurons_per_sec': n * duration / 0.1 / elapsed
        }
        
        print(f"N={n:4d}: {elapsed:.3f}s ({results[n]['neurons_per_sec']:.0f} neurons/s)")
    
    return results


def benchmark_synapse_types(num_neurons=200, duration=100.0):
    """Benchmark different synapse types."""
    print("\n" + "=" * 60)
    print("SYNAPSE TYPE BENCHMARKS")
    print("=" * 60)
    
    synapse_types = ['AMPA', 'NMDA', 'GABA_A']
    
    results = {}
    
    for syn_type in synapse_types:
        config = SimulatorConfig(
            num_neurons=num_neurons,
            duration=duration,
            dt=0.1,
            input_current=5.0,
            record_spikes=False
        )
        
        sim = Simulator(config)
        sim.create_network()
        
        # Override synapse type
        sim.network.config.synapse_type = syn_type.lower()
        
        start = time.time()
        sim.run(verbose=False)
        elapsed = time.time() - start
        
        results[syn_type] = elapsed
        print(f"{syn_type}: {elapsed:.3f}s")
    
    return results


def benchmark_topology():
    """Benchmark different network topologies."""
    print("\n" + "=" * 60)
    print("NETWORK TOPOLOGY BENCHMARKS")
    print("=" * 60)
    
    topologies = {
        'Fully Connected': 'fully_connected',
        'Sparse Random': 'sparse_random',
        'Small World': 'small_world',
        'Layered': 'layered'
    }
    
    num_neurons = 200
    duration = 100.0
    
    results = {}
    
    for name, topo in topologies.items():
        config = SimulatorConfig(
            num_neurons=num_neurons,
            duration=duration,
            dt=0.1,
            input_current=5.0
        )
        
        sim = Simulator(config)
        sim.create_network(topology=topo)
        
        start = time.time()
        sim.run(verbose=False)
        elapsed = time.time() - start
        
        results[name] = {
            'time': elapsed,
            'synapses': len(sim.network.synapses)
        }
        
        print(f"{name}: {elapsed:.3f}s (synapses: {results[name]['synapses']})")
    
    return results


def benchmark_parallel_vs_serial(num_neurons=500):
    """Compare parallel and serial execution."""
    print("\n" + "=" * 60)
    print("PARALLEL VS SERIAL BENCHMARK")
    print("=" * 60)
    
    # Serial
    config = SimulatorConfig(
        num_neurons=num_neurons,
        duration=100.0,
        dt=0.1,
        input_current=5.0,
        parallel=False
    )
    
    sim = Simulator(config)
    sim.create_network()
    
    start = time.time()
    sim.run(verbose=False)
    serial_time = time.time() - start
    
    print(f"Serial: {serial_time:.3f}s")
    
    # Note: Parallel implementation would go here
    # For now, just report serial results
    print(f"\nNote: Parallel execution not yet implemented")
    
    return {'serial': serial_time}


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("NEURONBRAIN PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    results = {}
    
    results['neuron_models'] = benchmark_neuron_models()
    results['network_sizes'] = benchmark_network_sizes()
    results['synapse_types'] = benchmark_synapse_types()
    results['topology'] = benchmark_topology()
    results['parallel'] = benchmark_parallel_vs_serial()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nFastest neuron model:")
    fastest = min(results['neuron_models'].items(), key=lambda x: x[1]['time'])
    print(f"  {fastest[0]}: {fastest[1]['time']:.3f}s")
    
    print("\nScalability:")
    for n, data in list(results['network_sizes'].items())[:3]:
        print(f"  N={n}: {data['time']:.3f}s")
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
