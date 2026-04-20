"""
Learning and plasticity module for NeuronBrain.

Implements spike-timing-dependent plasticity (STDP), synaptic scaling,
and other plasticity mechanisms.
"""

from .stdp import STDP, STDPParameters
from .plasticity import SynapticScaling, IntrinsicPlasticity, BCMPlasticity, OjaLearningRule
from .stdp_factory import STDPFactory

__all__ = [
    "STDP",
    "STDPParameters",
    "SynapticPlasticity",
    "HomeostaticPlasticity",
    "STDPFactory",
]
