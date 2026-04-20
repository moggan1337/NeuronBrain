"""
Neural circuits module for NeuronBrain.

This module implements neural circuit simulations including networks
of interconnected neurons with various topologies.
"""

from .network import NeuralNetwork, NetworkTopology
from .cortical_column import CorticalColumn

__all__ = [
    "NeuralNetwork",
    "NetworkTopology",
    "CorticalColumn",
]
