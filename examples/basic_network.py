"""
Example 1: Basic Network Simulation

This example demonstrates how to create and run a basic
neural network simulation with Izhikevich neurons.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import Simulator, SimulatorConfig
from src.utils.analysis import compute_cv, detect_bursts


def run_basic_simulation():
    """Run a basic network simulation."""
    print("=" * 60)
    print("Example 1: Basic Network Simulation")
    print("=" * 60)
    
    # Create simulator
    config = SimulatorConfig(
        num_neurons=100,
        duration=1000.0,
        dt=0.1,
        neuron_type="izhikevich",
        input_current=10.0,
        record_spikes=True
    )
    
    sim = Simulator(config)
    sim.create_network(topology="sparse_random")
    
    # Run simulation
    result = sim.run(verbose=True)
    
    # Analyze results
    print("\nAnalysis:")
    print(f"  Total spikes: {len(result.spikes)}")
    print(f"  Mean firing rate: {np.mean(result.rates):.2f} Hz")
    print(f"  Max firing rate: {np.max(result.rates):.2f} Hz")
    print(f"  Active neurons: {np.sum(result.rates > 0)}")
    
    # Compute CV for first active neuron
    active_neurons = np.where(result.rates > 0)[0]
    if len(active_neurons) > 0:
        first_neuron = active_neurons[0]
        spike_times = [t for t, n in result.spikes if n == first_neuron]
        if len(spike_times) > 2:
            cv = compute_cv(spike_times)
            print(f"  CV of neuron {first_neuron}: {cv:.3f}")
    
    return result


if __name__ == "__main__":
    run_basic_simulation()
