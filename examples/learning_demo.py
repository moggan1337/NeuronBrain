"""
Example 3: STDP Learning

This example demonstrates spike-timing-dependent plasticity (STDP)
in a small network.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator import Simulator, SimulatorConfig
from src.learning import STDPFactory, STDPParameters


def demonstrate_stdp():
    """Demonstrate STDP learning."""
    print("=" * 60)
    print("Example 3: STDP Learning")
    print("=" * 60)
    
    # Configuration with STDP
    config = SimulatorConfig(
        num_neurons=50,
        duration=2000.0,
        dt=0.1,
        neuron_type="izhikevich",
        input_current=8.0,
        stdp_enabled=True
    )
    
    sim = Simulator(config)
    sim.create_network(topology="sparse_random")
    
    # Get initial weights
    initial_weights = [s.weight for s in sim.network.synapses[:10]]
    
    print(f"\nInitial weights (first 10): {initial_weights[:5]}...")
    
    # Run simulation
    result = sim.run(verbose=True)
    
    # Get final weights
    final_weights = [s.weight for s in sim.network.synapses[:10]]
    
    print(f"\nFinal weights (first 10): {final_weights[:5]}...")
    
    # Analyze changes
    weight_changes = [f - i for i, f in zip(initial_weights, final_weights)]
    print(f"\nWeight changes: min={min(weight_changes):.4f}, "
          f"max={max(weight_changes):.4f}, "
          f"mean={np.mean(weight_changes):.4f}")
    
    # Analyze spike patterns
    print(f"\nSpike statistics:")
    print(f"  Total spikes: {len(result.spikes)}")
    print(f"  Mean rate: {np.mean(result.rates):.2f} Hz")
    
    return result


if __name__ == "__main__":
    demonstrate_stdp()
