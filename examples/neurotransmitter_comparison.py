"""
Example 2: Comparing Different Neuron Models

This example compares the behavior of different neuron models
(Hodgkin-Huxley, LIF, and Izhikevich) under the same input.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.models.lif import LeakyIntegrateAndFire, LIFParameters
from src.models.izhikevich import IzhikevichNeuron, IzhikevichParameters


def compare_neuron_models():
    """Compare different neuron models."""
    print("=" * 60)
    print("Example 2: Comparing Neuron Models")
    print("=" * 60)
    
    # Parameters
    duration = 500.0  # ms
    dt = 0.01  # ms
    input_current = 10.0  # nA
    
    # Create neurons
    hh = HodgkinHuxleyNeuron(HodgkinHuxleyParameters())
    lif = LeakyIntegrateAndFire(LIFParameters(tau_mem=20.0))
    izh = IzhikevichNeuron(IzhikevichParameters())
    
    neurons = {
        'Hodgkin-Huxley': hh,
        'Leaky Integrate-and-Fire': lif,
        'Izhikevich': izh
    }
    
    results = {name: {'voltage': [], 'spikes': []} for name in neurons}
    
    print(f"\nSimulating {duration} ms with input current {input_current} nA...")
    
    for name, neuron in neurons.items():
        t = 0.0
        while t < duration:
            V = neuron.update(dt, input_current)
            results[name]['voltage'].append(V)
            
            if len(neuron.state.spike_history) > len(results[name]['spikes']):
                results[name]['spikes'].append(t)
            
            t += dt
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    print(f"{'Model':<30} {'Spikes':<10} {'Mean V':<12} {'Spike Rate':<10}")
    print("-" * 60)
    
    for name, data in results.items():
        mean_v = np.mean(data['voltage'])
        rate = len(data['spikes']) / (duration / 1000) if duration > 0 else 0
        print(f"{name:<30} {len(data['spikes']):<10} {mean_v:<12.2f} {rate:<10.2f} Hz")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    time = np.arange(0, duration, dt)
    
    for i, (name, data) in enumerate(results.items()):
        axes[i].plot(time, data['voltage'], 'b-', linewidth=0.5)
        axes[i].axhline(y=-55, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[i].set_ylabel('V (mV)')
        axes[i].set_title(name)
        axes[i].set_ylim(-90, 50)
        
        # Mark spikes
        for spike in data['spikes']:
            idx = int(spike / dt)
            if idx < len(data['voltage']):
                axes[i].plot(spike, data['voltage'][idx], 'ro', markersize=3)
    
    axes[-1].set_xlabel('Time (ms)')
    plt.suptitle('Comparison of Neuron Models')
    plt.tight_layout()
    plt.savefig('neuron_comparison.png', dpi=150)
    print("\nSaved plot to neuron_comparison.png")
    
    return results


if __name__ == "__main__":
    compare_neuron_models()
