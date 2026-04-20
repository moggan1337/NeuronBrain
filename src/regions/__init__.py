"""
Brain regions module for NeuronBrain.

This module implements models of various brain regions with their
specific connectivity patterns and functional properties.
"""

from .base_region import BrainRegion, RegionConfig
from .hippocampus import Hippocampus
from .thalamus_model import Thalamus
from .basal_ganglia import BasalGanglia

__all__ = [
    "BrainRegion",
    "RegionConfig",
    "Hippocampus",
    "Thalamus",
    "BasalGanglia",
]
