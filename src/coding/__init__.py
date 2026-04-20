"""
Neural coding module for NeuronBrain.

This module implements various neural coding schemes including
rate coding, temporal coding, and population coding.
"""

from .rate_coding import RateCoder, PoissonCoder
from .temporal_coding import TemporalCoder, PhaseCoder
from .population_coding import PopulationCoder, VectorCoder

__all__ = [
    "RateCoder",
    "PoissonCoder",
    "TemporalCoder",
    "PhaseCoder",
    "PopulationCoder",
    "VectorCoder",
]
