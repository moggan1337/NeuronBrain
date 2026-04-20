"""
Synapse Factory for creating configured synapses.

This module provides a factory pattern for creating synapses with
presets for common configurations.
"""

from typing import Optional, Dict, Any, Type
from enum import Enum

from .chemical_synapse import ChemicalSynapse, SynapticReceptor, SynapticParameters
from .electrical_synapse import ElectricalSynapse, GapJunctionParameters


class SynapseType(Enum):
    """Types of synapses that can be created."""
    EXCITATORY_AMPA = "excitatory_ampa"
    EXCITATORY_NMDA = "excitatory_nmda"
    INHIBITORY_GABA_A = "inhibitory_gaba_a"
    INHIBITORY_GABA_B = "inhibitory_gaba_b"
    ELECTRICAL = "electrical"
    CUSTOM = "custom"


class SynapseFactory:
    """
    Factory for creating synapses with preset configurations.
    
    This factory provides convenient methods for creating synapses
    with common parameter sets for different brain regions and
    neuron types.
    
    Example:
        >>> factory = SynapseFactory()
        >>> 
        >>> # Create common synapse types
        >>> exc_syn = factory.create("EXCITATORY_AMPA", pre_id=1, post_id=2)
        >>> inh_syn = factory.create("INHIBITORY_GABA_A", pre_id=3, post_id=2)
        >>> 
        >>> # Create with custom parameters
        >>> syn = factory.create("CUSTOM", receptor="AMPA", g_max=2.0)
    """
    
    # Preset configurations
    PRESETS = {
        SynapseType.EXCITATORY_AMPA: {
            'class': ChemicalSynapse,
            'receptor': SynapticReceptor.AMPA,
            'g_max': 1.0,
            'E_syn': 0.0,
            'tau_rise': 0.3,
            'tau_decay': 5.0,
            'U_base': 0.6,
            'tau_facilitation': 0.01,
            'tau_depression': 0.1,
        },
        SynapseType.EXCITATORY_NMDA: {
            'class': ChemicalSynapse,
            'receptor': SynapticReceptor.NMDA,
            'g_max': 0.8,
            'E_syn': 0.0,
            'tau_rise': 2.0,
            'tau_decay': 50.0,
            'U_base': 0.5,
            'tau_facilitation': 0.05,
            'tau_depression': 0.2,
        },
        SynapseType.INHIBITORY_GABA_A: {
            'class': ChemicalSynapse,
            'receptor': SynapticReceptor.GABA_A,
            'g_max': 1.0,
            'E_syn': -70.0,
            'tau_rise': 0.3,
            'tau_decay': 8.0,
            'U_base': 0.25,
            'tau_facilitation': 0.01,
            'tau_depression': 0.1,
        },
        SynapseType.INHIBITORY_GABA_B: {
            'class': ChemicalSynapse,
            'receptor': SynapticReceptor.GABA_B,
            'g_max': 0.5,
            'E_syn': -90.0,
            'tau_rise': 5.0,
            'tau_decay': 100.0,
            'U_base': 0.2,
            'tau_facilitation': 0.05,
            'tau_depression': 0.5,
        },
        SynapseType.ELECTRICAL: {
            'class': ElectricalSynapse,
            'g_max': 1.0,
        },
    }
    
    # Brain region-specific presets
    REGION_PRESETS = {
        'neocortex_exc': {
            'AMPA': SynapseType.EXCITATORY_AMPA,
            'NMDA': SynapseType.EXCITATORY_NMDA,
            'GABA_A': SynapseType.INHIBITORY_GABA_A,
            'gap_junction': 0.1,  # Conductance for exc neurons
        },
        'neocortex_inh': {
            'AMPA': SynapseType.EXCITATORY_AMPA,
            'GABA_A': SynapseType.INHIBITORY_GABA_A,
            'gap_junction': 0.3,  # Higher for interneurons
        },
        'hippocampus_ca1': {
            'AMPA': SynapseType.EXCITATORY_AMPA,
            'NMDA': SynapseType.EXCITATORY_NMDA,
            'GABA_A': SynapseType.INHIBITORY_GABA_A,
            'GABA_B': SynapseType.INHIBITORY_GABA_B,
        },
        'thalamus': {
            'AMPA': SynapseType.EXCITATORY_AMPA,
            'GABA_A': SynapseType.INHIBITORY_GABA_A,
            'gap_junction': 0.2,
        },
        'cerebellum': {
            'AMPA': SynapseType.EXCITATORY_AMPA,
            'GABA_A': SynapseType.INHIBITORY_GABA_A,
        },
    }
    
    @classmethod
    def create(
        cls,
        synapse_type: str,
        pre_id: Optional[int] = None,
        post_id: Optional[int] = None,
        weight: float = 1.0,
        plasticity_enabled: bool = True,
        **kwargs
    ) -> Any:
        """
        Create a synapse with specified configuration.
        
        Args:
            synapse_type: Type name (string) or SynapseType enum
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            weight: Synaptic weight multiplier
            plasticity_enabled: Enable short-term plasticity
            **kwargs: Additional parameters to override presets
            
        Returns:
            Configured synapse instance
        """
        # Parse synapse type
        if isinstance(synapse_type, str):
            try:
                synapse_type = SynapseType[synapse_type.upper().replace('-', '_')]
            except KeyError:
                synapse_type = SynapseType.CUSTOM
        
        if synapse_type == SynapseType.CUSTOM:
            # Custom synapse
            if kwargs.get('electrical', False):
                return cls._create_electrical(pre_id, post_id, **kwargs)
            else:
                return cls._create_chemical_custom(pre_id, post_id, weight, plasticity_enabled, **kwargs)
        
        # Get preset
        preset = cls.PRESETS.get(synapse_type)
        if preset is None:
            raise ValueError(f"Unknown synapse type: {synapse_type}")
        
        # Create based on class type
        synapse_class = preset.get('class')
        
        if synapse_class == ChemicalSynapse:
            return cls._create_chemical(
                synapse_type, pre_id, post_id, weight, plasticity_enabled, **kwargs
            )
        elif synapse_class == ElectricalSynapse:
            return cls._create_electrical(pre_id, post_id, **kwargs)
        else:
            raise ValueError(f"Unknown synapse class: {synapse_class}")
    
    @classmethod
    def _create_chemical(
        cls,
        synapse_type: SynapseType,
        pre_id: Optional[int],
        post_id: Optional[int],
        weight: float,
        plasticity_enabled: bool,
        **kwargs
    ) -> ChemicalSynapse:
        """Create a chemical synapse from preset."""
        preset = cls.PRESETS[synapse_type]
        
        params = SynapticParameters(
            receptor=preset['receptor'],
            g_max=kwargs.get('g_max', preset.get('g_max', 1.0)),
            E_syn=kwargs.get('E_syn', preset.get('E_syn', 0.0)),
            tau_rise=kwargs.get('tau_rise', preset.get('tau_rise', 0.5)),
            tau_decay=kwargs.get('tau_decay', preset.get('tau_decay', 5.0)),
            U_base=kwargs.get('U_base', preset.get('U_base', 0.5)),
            tau_facilitation=kwargs.get('tau_facilitation', preset.get('tau_facilitation', 0.01)),
            tau_depression=kwargs.get('tau_depression', preset.get('tau_depression', 0.1)),
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
        )
        
        return ChemicalSynapse(
            parameters=params,
            receptor=preset['receptor'],
            pre_id=pre_id,
            post_id=post_id,
            weight=weight,
            plasticity_enabled=plasticity_enabled
        )
    
    @classmethod
    def _create_electrical(
        cls,
        pre_id: Optional[int],
        post_id: Optional[int],
        **kwargs
    ) -> ElectricalSynapse:
        """Create an electrical synapse (gap junction)."""
        params = GapJunctionParameters(
            g_max=kwargs.get('g_max', 1.0),
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
        )
        
        return ElectricalSynapse(
            parameters=params,
            pre_id=pre_id,
            post_id=post_id,
            conductance=kwargs.get('g_max', 1.0),
            voltage_dependent=kwargs.get('voltage_dependent', False)
        )
    
    @classmethod
    def _create_chemical_custom(
        cls,
        pre_id: Optional[int],
        post_id: Optional[int],
        weight: float,
        plasticity_enabled: bool,
        **kwargs
    ) -> ChemicalSynapse:
        """Create a custom chemical synapse."""
        receptor_name = kwargs.get('receptor', 'AMPA')
        try:
            receptor = SynapticReceptor[receptor_name.upper()]
        except KeyError:
            receptor = SynapticReceptor.AMPA
        
        params = SynapticParameters(
            receptor=receptor,
            g_max=kwargs.get('g_max', 1.0),
            E_syn=kwargs.get('E_syn', 0.0 if receptor == SynapticReceptor.AMPA else -70.0),
            tau_rise=kwargs.get('tau_rise', 0.5),
            tau_decay=kwargs.get('tau_decay', 5.0),
            U_base=kwargs.get('U_base', 0.5),
            tau_facilitation=kwargs.get('tau_facilitation', 0.01),
            tau_depression=kwargs.get('tau_depression', 0.1),
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
        )
        
        return ChemicalSynapse(
            parameters=params,
            receptor=receptor,
            pre_id=pre_id,
            post_id=post_id,
            weight=weight,
            plasticity_enabled=plasticity_enabled
        )
    
    @classmethod
    def create_for_region(
        cls,
        region: str,
        synapse_type: str,
        pre_id: Optional[int] = None,
        post_id: Optional[int] = None
    ) -> Any:
        """
        Create synapse appropriate for a brain region.
        
        Args:
            region: Brain region name
            synapse_type: Type of synapse ('AMPA', 'NMDA', 'GABA_A', 'GABA_B', 'gap_junction')
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            
        Returns:
            Configured synapse
        """
        if region not in cls.REGION_PRESETS:
            raise ValueError(f"Unknown region: {region}")
        
        region_preset = cls.REGION_PRESETS[region]
        
        if synapse_type == 'gap_junction':
            g = region_preset.get('gap_junction', 0.1)
            return cls.create('ELECTRICAL', pre_id, post_id, g_max=g)
        elif synapse_type in region_preset:
            return cls.create(region_preset[synapse_type], pre_id, post_id)
        else:
            raise ValueError(f"Unknown synapse type '{synapse_type}' for region '{region}'")
    
    @classmethod
    def create_layer_connectivity(
        cls,
        source_layer: str,
        target_layer: str,
        connectivity_pattern: str = 'full'
    ) -> Dict[str, Any]:
        """
        Get synapse parameters for layer-to-layer connectivity.
        
        Args:
            source_layer: Source cortical layer
            target_layer: Target cortical layer
            connectivity_pattern: 'full', 'sparse', or 'specific'
            
        Returns:
            Dictionary with synapse configurations
        """
        # Layer connectivity patterns
        LAYER_CONNECTIVITY = {
            ('L2/3', 'L2/3'): {'type': 'EXCITATORY_AMPA', 'density': 0.1, 'weight': 1.0},
            ('L2/3', 'L4'): {'type': 'EXCITATORY_AMPA', 'density': 0.05, 'weight': 0.8},
            ('L4', 'L2/3'): {'type': 'EXCITATORY_AMPA', 'density': 0.15, 'weight': 1.2},
            ('L4', 'L5'): {'type': 'EXCITATORY_AMPA', 'density': 0.1, 'weight': 1.0},
            ('L5', 'L5'): {'type': 'EXCITATORY_AMPA', 'density': 0.1, 'weight': 1.0},
            ('L5', 'L6'): {'type': 'EXCITATORY_AMPA', 'density': 0.05, 'weight': 0.8},
            ('L6', 'L4'): {'type': 'EXCITATORY_AMPA', 'density': 0.03, 'weight': 0.5},
        }
        
        key = (source_layer, target_layer)
        if key in LAYER_CONNECTIVITY:
            return LAYER_CONNECTIVITY[key]
        else:
            return {'type': 'EXCITATORY_AMPA', 'density': 0.05, 'weight': 0.8}
