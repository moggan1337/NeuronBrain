"""
Synapses module for NeuronBrain.

This module implements various synapse models including chemical synapses
with dynamic properties and electrical gap junctions for direct coupling.
"""

from .chemical_synapse import ChemicalSynapse, SynapticReceptor
from .electrical_synapse import ElectricalSynapse, GapJunction
from .synapse_factory import SynapseFactory

__all__ = [
    "ChemicalSynapse",
    "SynapticReceptor",
    "ElectricalSynapse",
    "GapJunction",
    "SynapseFactory",
]
