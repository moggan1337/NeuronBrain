# NeuronBrain: Biological Neural Network Simulator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Version-1.0.0-orange.svg" alt="Version">
</p>

NeuronBrain is a comprehensive Python library for simulating biological neural networks. It provides accurate implementations of well-established neural dynamics models, synaptic mechanisms, and plasticity rules, making it suitable for both research and educational purposes.

## 🎬 Demo
![NeuronBrain Demo](demo.gif)

*Biological neural network simulation*

## Screenshots
| Component | Preview |
|-----------|---------|
| Network Builder | ![builder](screenshots/network-builder.png) |
| Simulation View | ![sim](screenshots/simulation.png) |
| Spike Raster | ![raster](screenshots/spike-raster.png) |

## Visual Description
Network builder shows neurons and synapses being connected. Simulation view displays membrane potentials updating in real-time. Spike raster plots neuron firing patterns over time.

---


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Neuron Models](#neuron-models)
- [Synapse Models](#synapse-models)
- [Learning and Plasticity](#learning-and-plasticity)
- [Neural Circuits](#neural-circuits)
- [Brain Regions](#brain-regions)
- [Neural Coding](#neural-coding)
- [Simulation API](#simulation-api)
- [Examples](#examples)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features

NeuronBrain provides a comprehensive set of features for neural simulation:

### Neuron Models

- **Hodgkin-Huxley Model**: Detailed biophysical model with sodium, potassium, and leak channels
- **Leaky Integrate-and-Fire (LIF)**: Simplified model for efficient large-scale simulation
- **Izhikevich Spiking Neuron**: Computational efficient model with rich dynamics

### Synaptic Mechanisms

- **Chemical Synapses**: Conductance-based synapses with various receptor types (AMPA, NMDA, GABA_A, GABA_B)
- **Electrical Synapses (Gap Junctions)**: Bidirectional coupling with optional voltage dependence
- **Synaptic Plasticity**: Short-term facilitation and depression

### Learning Rules

- **STDP (Spike-Timing-Dependent Plasticity)**: Multiple variants including additive, multiplicative, and triplet STDP
- **Reward-Modulated STDP**: For reinforcement learning applications
- **Synaptic Scaling**: Homeostatic plasticity mechanisms
- **BCM Rule**: Bienenstock-Cooper-Munro learning rule
- **Oja's Rule**: With synaptic normalization

### Network Topologies

- **Fully Connected**: All-to-all connectivity
- **Sparse Random**: Random connections with configurable probability
- **Small-World**: Watts-Strogatz small-world networks
- **Spatial**: Distance-dependent connectivity
- **Layered**: Cortical-like layered structure

### Brain Regions

- **Hippocampus**: CA1, CA3, DG regions with trisynaptic circuit
- **Thalamus**: Relay nuclei and reticular formation
- **Basal Ganglia**: Direct and indirect pathways

### Neural Coding

- **Rate Coding**: Average firing rate representation
- **Temporal Coding**: Time-to-first-spike, latency coding
- **Population Coding**: Vector and population vector coding
- **Phase Coding**: Theta-gamma coupling

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0 (for visualization)
- pytest >= 6.2.0 (for testing)

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/moggan1337/NeuronBrain.git
cd NeuronBrain

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Installation

```bash
pip install neuronbrain
```

## Quick Start

### Basic Network Simulation

```python
from neuronbrain import Simulator, SimulatorConfig

# Create simulator
config = SimulatorConfig(
    num_neurons=100,
    duration=1000.0,
    dt=0.1,
    neuron_type="izhikevich",
    input_current=10.0
)

sim = Simulator(config)
sim.create_network()
sim.add_input(current=10.0)

# Run simulation
result = sim.run(verbose=True)

# Analyze results
print(f"Mean firing rate: {result.rates.mean():.2f} Hz")
print(f"Total spikes: {len(result.spikes)}")
```

### Comparing Neuron Models

```python
from neuronbrain.models import HodgkinHuxleyNeuron, LeakyIntegrateAndFire, IzhikevichNeuron

# Create neurons
hh = HodgkinHuxleyNeuron()
lif = LeakyIntegrateAndFire()
izh = IzhikevichNeuron()

# Simulate with same input
for _ in range(1000):
    hh.update(0.1, 10.0)
    lif.update(0.1, 10.0)
    izh.update(0.1, 10.0)

print(f"HH spikes: {len(hh.state.spike_history)}")
print(f"LIF spikes: {len(lif.state.spike_history)}")
print(f"Izh spikes: {len(izh.state.spike_history)}")
```

### STDP Learning

```python
from neuronbrain.simulator import Simulator, SimulatorConfig

config = SimulatorConfig(
    num_neurons=50,
    duration=2000.0,
    stdp_enabled=True
)

sim = Simulator(config)
sim.create_network()

result = sim.run(verbose=True)

# Check weight changes
initial_weights = [s.weight for s in sim.network.synapses[:10]]
```

## Architecture Overview

NeuronBrain follows a modular architecture with clear separation of concerns:

```
neuronbrain/
├── src/
│   ├── models/          # Neuron models
│   ├── synapses/        # Synaptic mechanisms
│   ├── learning/        # Plasticity rules
│   ├── circuits/        # Network implementations
│   ├── regions/         # Brain region models
│   ├── coding/          # Neural coding schemes
│   └── utils/           # Utilities
├── examples/            # Example scripts
├── tests/               # Test suite
└── benchmarks/          # Performance benchmarks
```

### Core Components

1. **Models**: Neuron implementations (HH, LIF, Izhikevich)
2. **Synapses**: Chemical and electrical synapses
3. **Learning**: STDP and other plasticity rules
4. **Circuits**: Network construction and simulation
5. **Regions**: Brain region models
6. **Coding**: Neural encoding schemes
7. **Utils**: Visualization and analysis tools

## Neuron Models

### Hodgkin-Huxley Model

The Hodgkin-Huxley model provides detailed biophysical realism:

```python
from neuronbrain.models import HodgkinHuxleyNeuron, HodgkinHuxleyParameters

params = HodgkinHuxleyParameters(
    g_na=120.0,    # Sodium conductance (mS/cm²)
    g_k=36.0,      # Potassium conductance (mS/cm²)
    g_l=0.3,       # Leak conductance (mS/cm²)
    E_na=50.0,     # Sodium reversal potential (mV)
    E_k=-77.0,     # Potassium reversal potential (mV)
    E_l=-54.387    # Leak reversal potential (mV)
)

neuron = HodgkinHuxleyNeuron(params)

# Simulation
for _ in range(10000):
    V = neuron.update(dt=0.01, input_current=10.0)
```

**Dynamics:**
- Membrane equation: C * dV/dt = I_na + I_k + I_l + I_ext
- Gating variables: m (Na+ activation), h (Na+ inactivation), n (K+ activation)
- Accurate action potential generation and propagation

### Leaky Integrate-and-Fire Model

The LIF model is computationally efficient:

```python
from neuronbrain.models import LeakyIntegrateAndFire, LIFParameters

params = LIFParameters(
    tau_mem=20.0,      # Membrane time constant (ms)
    r_mem=10.0,        # Membrane resistance (MΩ)
    threshold=-55.0,   # Spike threshold (mV)
    reset_potential=-70.0
)

neuron = LeakyIntegrateAndFire(params)
```

**Features:**
- Exponential integrate-and-fire variant
- Spike frequency adaptation
- Subthreshold oscillations
- Configurable reset dynamics

### Izhikevich Model

The Izhikevich model balances realism and efficiency:

```python
from neuronbrain.models import IzhikevichNeuron, IzhikevichParameters
from neuronbrain.models.izhikevich import IzhikevichNeuronType

# Using presets
neuron = IzhikevichNeuron.from_preset(IzhikevichNeuronType.FastSpiking)

# Custom parameters
params = IzhikevichParameters(
    a=0.02,    # Recovery time constant
    b=0.2,     # Subthreshold coupling
    c=-65.0,   # After-spike reset (mV)
    d=8.0      # Recovery increment
)
neuron = IzhikevichNeuron(params)
```

**Presets:**
- Regular Spiking (RS): Pyramidal cells
- Fast Spiking (FS): Interneurons
- Intrinsic Bursting (IB): Bursting neurons
- Chattering (CH): High-frequency bursts
- Low Threshold Spiking (LTS): Interneurons

## Synapse Models

### Chemical Synapses

```python
from neuronbrain.synapses import ChemicalSynapse, SynapticReceptor, SynapseFactory

# Direct creation
synapse = ChemicalSynapse(
    receptor=SynapticReceptor.AMPA,
    pre_id=0,
    post_id=1,
    weight=1.0
)

# Using factory
synapse = SynapseFactory.create(
    "EXCITATORY_AMPA",
    pre_id=0,
    post_id=1,
    weight=0.5
)
```

**Receptor Types:**
- AMPA: Fast excitatory (τ_decay ≈ 5-10 ms)
- NMDA: Slow excitatory with Mg²⁺ block
- GABA_A: Fast inhibitory (τ_decay ≈ 8-10 ms)
- GABA_B: Slow inhibitory, G-protein coupled

**Short-Term Plasticity:**
- Facilitation: Increased release probability
- Depression: Vesicle depletion

### Electrical Synapses (Gap Junctions)

```python
from neuronbrain.synapses import ElectricalSynapse

gap = ElectricalSynapse(
    pre_id=0,
    post_id=1,
    conductance=2.0,  # nS
    voltage_dependent=True
)

# Bidirectional current flow
i_pre, i_post = gap.get_current(v_pre=-50.0, v_post=-70.0)
```

## Learning and Plasticity

### STDP (Spike-Timing-Dependent Plasticity)

```python
from neuronbrain.learning import STDP, STDPFactory, STDPParameters

# Create with custom parameters
params = STDPParameters(
    A_plus=0.01,      # Potentiation amplitude
    A_minus=0.012,    # Depression amplitude
    tau_plus=20.0,    # Potentiation time constant (ms)
    tau_minus=20.0    # Depression time constant (ms)
)

stdp = STDP(params)

# Record spikes
stdp.record_presynaptic_spike('synapse_0', time=10.0)
stdp.record_postsynaptic_spike('synapse_0', time=15.0)

# Compute weight update
dw = stdp.compute_update('synapse_0')
```

**Variants:**
- Exponential
- Power-law
- Gaussian
- Triangular
- Triplet STDP

### Reward-Modulated STDP

```python
from neuronbrain.learning import RewardModulatedSTDP

stdp = RewardModulatedSTDP(
    reward_factor=1.0,
    eligibility_tau=1000.0
)

# Apply reward signal
stdp.apply_reward(reward=1.0, current_time=100.0, synapses={})
```

### Synaptic Scaling (Homeostatic Plasticity)

```python
from neuronbrain.learning import SynapticScaling

scaling = SynapticScaling(
    target_activity=10.0,  # Hz
    learning_rate=0.01
)

# Update scaling
scaling.update(neuron_id=0, synapses=[], actual_activity=5.0, dt=0.1)
```

## Neural Circuits

### Creating Networks

```python
from neuronbrain.circuits import NeuralNetwork, NetworkConfig, NetworkTopology

config = NetworkConfig(
    num_neurons=100,
    topology=NetworkTopology.SPARSE_RANDOM,
    connection_probability=0.1,
    neuron_type="izhikevich",
    excitatory_ratio=0.8,
    stdp_enabled=True
)

network = NeuralNetwork(config)
```

### Network Topologies

- **Fully Connected**: All neurons connected to all others
- **Sparse Random**: Random connections with probability p
- **Small World**: Watts-Strogatz model with rewiring
- **Spatial**: Distance-dependent connectivity
- **Layered**: Cortical-like layers

### Running Simulations

```python
# Inject current
network.inject_current(np.ones(100) * 10.0)

# Run simulation
for step in range(10000):
    result = network.step(dt=0.1)
    if result['num_spikes'] > 0:
        print(f"Time {network.time}: {result['num_spikes']} spikes")
```

### Recording Results

```python
# Get spike data
spikes = network.get_spikes()

# Get firing rates
rates = network.get_firing_rates()

# Get raster plot data
times, ids = network.get_raster_plot()

# Get connectivity matrix
conn = network.get_connectivity_matrix()
```

## Brain Regions

### Hippocampus

```python
from neuronbrain.regions import Hippocampus, HippocampusConfig

config = HippocampusConfig(name="hippocampus")
hippo = Hippocampus(config)

# Present spatial pattern
hippo.present_spatial_pattern(pattern)

# Run theta oscillation
for _ in range(10000):
    hippo.step(0.1)
    hippo.apply_theta_drive()
```

**Features:**
- Dentate gyrus (pattern separation)
- CA3 (autoassociative memory)
- CA1 (output)
- Theta-gamma coupling

### Thalamus

```python
from neuronbrain.regions import Thalamus, ThalamusConfig

thalamus = Thalamus(ThalamusConfig())

# Set attention gate
thalamus.set_attention(gate=0.8)

# Inject sensory input
thalamus.inject_sensory_input('LGN', pattern)
```

### Basal Ganglia

```python
from neuronbrain.regions import BasalGanglia, BasalGangliaConfig

bg = BasalGanglia(BasalGangliaConfig())

# Inject reward
bg.inject_reward(reward=1.0)

# Select action
action = bg.select_action()
```

## Neural Coding

### Rate Coding

```python
from neuronbrain.coding import RateCoder, PoissonCoder

# Encode signal as rates
coder = RateCoder(time_window=100.0)
rates = coder.encode(signal, num_neurons=100)

# Decode from spikes
signal_estimate = coder.decode(spike_train)

# Generate Poisson spikes
poisson = PoissonCoder()
spikes = poisson.generate_spikes(rate=10.0, duration=1000.0)
```

### Temporal Coding

```python
from neuronbrain.coding import TemporalCoder

coder = TemporalCoder()

# Time-to-first-spike encoding
first_spikes = coder.time_to_first_spike(signal)

# Latency coding
latencies = coder.latency_code(signal)

# Phase coding
spike_times, phases = coder.phase_code(signal, oscillation_frequency=8.0)
```

### Population Coding

```python
from neuronbrain.coding import PopulationCoder, VectorCoder

# Direction coding
pop_coder = PopulationCoder(num_neurons=100)
rates = pop_coder.encode(stimulus_value=45.0)
estimate = pop_coder.decode(rates)

# Vector coding
vec_coder = VectorCoder(dimensions=2, num_neurons=100)
rates = vec_coder.encode_direction([1.0, 0.0])
direction = vec_coder.decode_direction(rates)
```

## Simulation API

### Simulator Class

```python
from neuronbrain.simulator import Simulator, SimulatorConfig, run_simulation

# Full simulator
config = SimulatorConfig(
    num_neurons=100,
    duration=1000.0,
    dt=0.1,
    record_spikes=True,
    record_voltage=False
)

sim = Simulator(config)
sim.create_network()
sim.add_input(current=10.0)
result = sim.run()

# Quick simulation
result = run_simulation(
    num_neurons=50,
    duration=500.0,
    input_current=5.0
)
```

### Configuration Options

```python
config = SimulatorConfig(
    dt=0.1,                      # Time step (ms)
    duration=1000.0,             # Simulation duration (ms)
    num_neurons=100,             # Number of neurons
    neuron_type="izhikevich",   # or "lif", "hodgkin_huxley"
    topology="sparse_random",   # or "fully_connected", "small_world", etc.
    connection_probability=0.1,  # For sparse networks
    input_current=10.0,         # Input current (nA)
    stdp_enabled=False,         # Enable STDP
    record_spikes=True,         # Record spikes
    record_voltage=False        # Record membrane potentials
)
```

## Examples

### Example 1: Basic Network

```bash
python examples/basic_network.py
```

### Example 2: Neuron Model Comparison

```bash
python examples/neurotransmitter_comparison.py
```

### Example 3: STDP Learning

```bash
python examples/learning_demo.py
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- `TestHodgkinHuxley`: Tests for HH neuron model
- `TestLeakyIntegrateAndFire`: Tests for LIF neuron model
- `TestIzhikevich`: Tests for Izhikevich neuron model
- `TestChemicalSynapse`: Tests for chemical synapses
- `TestElectricalSynapse`: Tests for gap junctions
- `TestNeuralNetwork`: Tests for network simulation
- `TestSimulator`: Tests for simulator class
- `TestSTDP`: Tests for STDP learning

## Benchmarks

Run performance benchmarks:

```bash
python benchmarks/benchmark_simulation.py
```

### Benchmark Results

Typical performance on modern hardware:

| Component | Performance |
|-----------|-------------|
| LIF Neuron | ~500K updates/sec |
| Izhikevich | ~200K updates/sec |
| Hodgkin-Huxley | ~50K updates/sec |
| Chemical Synapse | ~100K updates/sec |
| Network (100 neurons) | ~10K steps/sec |

## API Reference

### Models

```python
# Base class
from neuronbrain.models import Neuron, NeuronType

# Implementations
from neuronbrain.models import HodgkinHuxleyNeuron, LeakyIntegrateAndFire, IzhikevichNeuron
```

### Synapses

```python
from neuronbrain.synapses import ChemicalSynapse, ElectricalSynapse, SynapseFactory
from neuronbrain.synapses.chemical_synapse import SynapticReceptor
```

### Learning

```python
from neuronbrain.learning import STDP, SynapticPlasticity, STDPFactory
from neuronbrain.learning.std import STDPParameters
```

### Circuits

```python
from neuronbrain.circuits import NeuralNetwork, NetworkConfig, NetworkTopology
from neuronbrain.circuits.cortical_column import CorticalColumn
```

### Regions

```python
from neuronbrain.regions import BrainRegion, Hippocampus, Thalamus, BasalGanglia
```

### Coding

```python
from neuronbrain.coding import RateCoder, TemporalCoder, PopulationCoder
```

### Simulator

```python
from neuronbrain.simulator import Simulator, SimulatorConfig, run_simulation
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

### Development Setup

```bash
# Clone and install
git clone https://github.com/moggan1337/NeuronBrain.git
cd NeuronBrain
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuronBrain in your research, please cite:

```bibtex
@software{neuronbrain,
  title={NeuronBrain: Biological Neural Network Simulator},
  author={NeuronBrain Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/moggan1337/NeuronBrain}
}
```

## References

1. Hodgkin, A.L. and Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *The Journal of Physiology*, 117(4), 500-544.

2. Izhikevich, E.M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.

3. Dayan, P. and Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.

4. Bi, G. and Poo, M. (2001). Synaptic modification by correlated activity: Hebb's postulate revisited. *Annual Review of Neuroscience*, 24(1), 139-166.

5. Destexhe, A., Mainen, Z.F., and Sejnowski, T.J. (1994). Synthesis of models for excitable membranes, synaptic transmission and neuromodulation using a common kinetic formalism. *Journal of Computational Neuroscience*, 1(3), 195-230.

## Acknowledgments

- The Hodgkin-Huxley model implementation is based on the original 1952 paper
- The Izhikevich model follows the 2003 IEEE publication
- STDP implementation follows the Bi & Poo (2001) framework
- Population coding follows Georgopoulos et al. (1986)

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/moggan1337/NeuronBrain/issues
- Email: contact@neuronbrain.org

---

<p align="center">
  Made with ❤️ for computational neuroscience
</p>
