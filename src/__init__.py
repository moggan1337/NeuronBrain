"""
NeuronBrain - Biological Neural Network Simulator

A comprehensive Python library for simulating biological neural networks
with accurate implementations of Hodgkin-Huxley, LIF, and Izhikevich
neuron models, along with synaptic plasticity and learning rules.

Example:
    >>> from neuronbrain import Simulator, SimulatorConfig
    >>> config = SimulatorConfig(num_neurons=100)
    >>> sim = Simulator(config)
    >>> sim.create_network()
    >>> result = sim.run()
"""

__version__ = "1.0.0"
__author__ = "NeuronBrain Team"

# Core imports
from .simulator import Simulator, SimulatorConfig, run_simulation, SimulationResult

# Models
from .models import (
    Neuron,
    NeuronType,
    HodgkinHuxleyNeuron,
    LeakyIntegrateAndFire,
    IzhikevichNeuron,
)

# Circuits
from .circuits import NeuralNetwork, NetworkConfig, NetworkTopology, CorticalColumn

# Regions
from .regions import BrainRegion, Hippocampus, Thalamus, BasalGanglia

# Synapses
from .synapses import ChemicalSynapse, ElectricalSynapse, SynapseFactory, SynapticReceptor

# Learning
from .learning import STDP, SynapticScaling, STDPFactory

# Coding
from .coding import RateCoder, TemporalCoder, PopulationCoder, PoissonCoder

__all__ = [
    # Version
    "__version__",
    # Simulator
    "Simulator",
    "SimulatorConfig",
    "run_simulation",
    "SimulationResult",
    # Models
    "Neuron",
    "NeuronType",
    "HodgkinHuxleyNeuron",
    "LeakyIntegrateAndFire",
    "IzhikevichNeuron",
    # Circuits
    "NeuralNetwork",
    "NetworkConfig",
    "NetworkTopology",
    "CorticalColumn",
    # Regions
    "BrainRegion",
    "Hippocampus",
    "Thalamus",
    "BasalGanglia",
    # Synapses
    "ChemicalSynapse",
    "ElectricalSynapse",
    "SynapseFactory",
    "SynapticReceptor",
    # Learning
    "STDP",
    "SynapticScaling",
    "STDPFactory",
    # Coding
    "RateCoder",
    "TemporalCoder",
    "PopulationCoder",
    "PoissonCoder",
]
