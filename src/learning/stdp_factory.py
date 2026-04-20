"""
STDP Factory for creating preconfigured STDP rules.

This module provides a factory for creating different STDP variants
commonly used in computational neuroscience research.
"""

from typing import Dict, Optional
from enum import Enum

from .stdp import STDP, STDPParameters, STDPCurve, RewardModulatedSTDP
from .plasticity import (
    SynapticScaling, 
    IntrinsicPlasticity, 
    BCMPlasticity,
    OjaLearningRule,
    StructuralPlasticity
)


class STDPType(Enum):
    """Preset STDP types."""
    STANDARD = "standard"
    SOFT_BOUNDS = "soft_bounds"
    HARD_BOUNDS = "hard_bounds"
    POWER_LAW = "power_law"
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"
    TRIPLET = "triplet"
    REWARD_MODULATED = "reward_modulated"


class STDPFactory:
    """
    Factory for creating STDP learning rules.
    
    This factory provides convenient methods for creating STDP
    implementations with common parameter sets.
    
    Example:
        >>> factory = STDPFactory()
        >>> 
        >>> # Create standard STDP
        >>> stdp = factory.create("STANDARD")
        >>> 
        >>> # Create triplet STDP
        >>> stdp = factory.create("TRIPLET")
        >>> 
        >>> # Create with custom parameters
        >>> params = factory.get_parameters("STANDARD")
        >>> params.A_plus = 0.02
        >>> stdp = factory.create_from_parameters(params)
    """
    
    # Preset configurations
    PRESETS: Dict[STDPType, Dict] = {
        STDPType.STANDARD: {
            'A_plus': 0.01,
            'A_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'soft_bounds': True,
        },
        STDPType.SOFT_BOUNDS: {
            'A_plus': 0.01,
            'A_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'soft_bounds': True,
        },
        STDPType.HARD_BOUNDS: {
            'A_plus': 0.01,
            'A_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'soft_bounds': False,
        },
        STDPType.POWER_LAW: {
            'A_plus': 0.005,
            'A_minus': 0.005,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'curve_type': STDPCurve.POWER_LAW,
        },
        STDPType.TRIANGULAR: {
            'A_plus': 0.01,
            'A_minus': 0.01,
            'tau_plus': 16.0,
            'tau_minus': 16.0,
            'curve_type': STDPCurve.TRIANGLE,
        },
        STDPType.GAUSSIAN: {
            'A_plus': 0.01,
            'A_minus': 0.01,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'curve_type': STDPCurve.GAUSSIAN,
        },
        STDPType.TRIPLET: {
            'A_plus': 0.01,
            'A_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'use_triplets': True,
        },
        STDPType.REWARD_MODULATED: {
            'A_plus': 0.01,
            'A_minus': 0.012,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'use_eligibility_trace': True,
            'eligibility_tau': 1000.0,
        },
    }
    
    # Brain region-specific presets
    REGION_PRESETS: Dict[str, STDPType] = {
        'hippocampus_ca1': STDPType.STANDARD,
        'visual_cortex': STDPType.SOFT_BOUNDS,
        'prefrontal_cortex': STDPType.TRIPLET,
        'cerebellum': STDPType.HARD_BOUNDS,
        'basal_ganglia': STDPType.REWARD_MODULATED,
    }
    
    @classmethod
    def create(
        cls,
        stdp_type: str = "STANDARD",
        learning_rate: float = 1.0
    ) -> STDP:
        """
        Create STDP learning rule.
        
        Args:
            stdp_type: Type name or STDPType enum
            learning_rate: Learning rate multiplier
            
        Returns:
            Configured STDP instance
        """
        # Parse type
        if isinstance(stdp_type, str):
            try:
                stdp_type = STDPType[stdp_type.upper()]
            except KeyError:
                stdp_type = STDPType.STANDARD
        
        # Get preset
        preset = cls.PRESETS.get(stdp_type, cls.PRESETS[STDPType.STANDARD])
        
        # Create parameters
        params = STDPParameters(
            A_plus=preset.get('A_plus', 0.01),
            A_minus=preset.get('A_minus', 0.012),
            tau_plus=preset.get('tau_plus', 20.0),
            tau_minus=preset.get('tau_minus', 20.0),
            soft_bounds=preset.get('soft_bounds', True),
            curve_type=preset.get('curve_type', STDPCurve.EXPONENTIAL),
            use_triplets=preset.get('use_triplets', False),
            use_eligibility_trace=preset.get('use_eligibility_trace', False),
            eligibility_tau=preset.get('eligibility_tau', 1000.0),
        )
        
        return STDP(params, learning_rate)
    
    @classmethod
    def create_reward_modulated(
        cls,
        reward_factor: float = 1.0,
        eligibility_tau: float = 1000.0
    ) -> RewardModulatedSTDP:
        """
        Create reward-modulated STDP.
        
        Args:
            reward_factor: Scaling factor for reward signal
            eligibility_tau: Time constant for eligibility traces (ms)
            
        Returns:
            R-STDP instance
        """
        params = STDPParameters(
            use_eligibility_trace=True,
            eligibility_tau=eligibility_tau,
        )
        
        return RewardModulatedSTDP(
            parameters=params,
            reward_factor=reward_factor
        )
    
    @classmethod
    def get_parameters(cls, stdp_type: str) -> STDPParameters:
        """
        Get parameters for a STDP type.
        
        Args:
            stdp_type: Type name
            
        Returns:
            STDPParameters instance
        """
        if isinstance(stdp_type, str):
            try:
                stdp_type = STDPType[stdp_type.upper()]
            except KeyError:
                stdp_type = STDPType.STANDARD
        
        preset = cls.PRESETS.get(stdp_type, cls.PRESETS[STDPType.STANDARD])
        
        return STDPParameters(
            A_plus=preset.get('A_plus', 0.01),
            A_minus=preset.get('A_minus', 0.012),
            tau_plus=preset.get('tau_plus', 20.0),
            tau_minus=preset.get('tau_minus', 20.0),
            soft_bounds=preset.get('soft_bounds', True),
            curve_type=preset.get('curve_type', STDPCurve.EXPONENTIAL),
            use_triplets=preset.get('use_triplets', False),
        )
    
    @classmethod
    def create_for_region(cls, region: str) -> STDP:
        """
        Create STDP appropriate for a brain region.
        
        Args:
            region: Brain region name
            
        Returns:
            Configured STDP instance
        """
        stdp_type = cls.REGION_PRESETS.get(region, STDPType.STANDARD)
        return cls.create(stdp_type)
    
    @classmethod
    def create_synaptic_scaling(
        cls,
        target_activity: float = 10.0,
        learning_rate: float = 0.01,
        multiplicative: bool = True
    ) -> SynapticScaling:
        """
        Create synaptic scaling plasticity.
        
        Args:
            target_activity: Target firing rate (Hz)
            learning_rate: Learning rate
            multiplicative: Use multiplicative scaling
            
        Returns:
            SynapticScaling instance
        """
        params = PlasticityParameters(
            learning_rate=learning_rate,
            target_activity=target_activity,
        )
        
        return SynapticScaling(
            parameters=params,
            target_activity=target_activity,
            multiplicative=multiplicative
        )
    
    @classmethod
    def create_intrinsic_plasticity(
        cls,
        target_activity: float = 10.0,
        target_property: str = 'threshold'
    ) -> IntrinsicPlasticity:
        """
        Create intrinsic plasticity.
        
        Args:
            target_activity: Target firing rate
            target_property: Which property to adjust
            
        Returns:
            IntrinsicPlasticity instance
        """
        return IntrinsicPlasticity(
            target_activity=target_activity,
            target_property=target_property
        )
    
    @classmethod
    def create_bcm_plasticity(
        cls,
        learning_rate: float = 0.01
    ) -> BCMPlasticity:
        """
        Create BCM plasticity.
        
        Args:
            learning_rate: Learning rate
            
        Returns:
            BCMPlasticity instance
        """
        return BCMPlasticity(learning_rate=learning_rate)
    
    @classmethod
    def create_oja_learning(
        cls,
        learning_rate: float = 0.01,
        normalization: float = 1.0
    ) -> OjaLearningRule:
        """
        Create Oja's learning rule.
        
        Args:
            learning_rate: Learning rate
            normalization: Normalization strength
            
        Returns:
            OjaLearningRule instance
        """
        return OjaLearningRule(
            learning_rate=learning_rate,
            normalization=normalization
        )
