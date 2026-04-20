"""
Test suite for NeuronBrain.

Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.models.lif import LeakyIntegrateAndFire, LIFParameters
from src.models.izhikevich import IzhikevichNeuron, IzhikevichParameters
from src.synapses import ChemicalSynapse, ElectricalSynapse
from src.circuits.network import NeuralNetwork, NetworkConfig, NetworkTopology
from src.simulator import Simulator, SimulatorConfig


class TestHodgkinHuxley:
    """Tests for Hodgkin-Huxley neuron."""
    
    def test_initialization(self):
        """Test HH neuron initialization."""
        neuron = HodgkinHuxleyNeuron()
        assert neuron.state.membrane_potential == pytest.approx(-70.0, rel=1.0)
        assert neuron.neuron_type.value == "hodgkin_huxley"
    
    def test_spike_generation(self):
        """Test that HH neuron can generate spikes."""
        neuron = HodgkinHuxleyNeuron()
        
        # Inject strong current
        for _ in range(1000):
            neuron.update(0.01, 50.0)
        
        assert len(neuron.state.spike_history) > 0
    
    def test_reset(self):
        """Test neuron reset."""
        neuron = HodgkinHuxleyNeuron()
        neuron.spike(100.0)
        neuron.reset_state()
        
        assert len(neuron.state.spike_history) == 0
        assert neuron.state.membrane_potential == pytest.approx(-70.0, abs=1.0)


class TestLeakyIntegrateAndFire:
    """Tests for LIF neuron."""
    
    def test_initialization(self):
        """Test LIF neuron initialization."""
        params = LIFParameters(tau_mem=20.0)
        neuron = LeakyIntegrateAndFire(params)
        
        assert neuron.parameters.tau_mem == 20.0
        assert neuron.state.membrane_potential == pytest.approx(-70.0, abs=1.0)
    
    def test_subthreshold_dynamics(self):
        """Test subthreshold membrane dynamics."""
        neuron = LeakyIntegrateAndFire()
        initial_v = neuron.state.membrane_potential
        
        # Small current (subthreshold)
        for _ in range(100):
            neuron.update(0.1, 1.0)
        
        # Should not spike but voltage should change
        assert len(neuron.state.spike_history) == 0
    
    def test_spike_with_strong_input(self):
        """Test spike generation with strong input."""
        neuron = LeakyIntegrateAndFire(LIFParameters(threshold=-55.0))
        
        for _ in range(1000):
            neuron.update(0.1, 20.0)
        
        assert len(neuron.state.spike_history) > 0


class TestIzhikevich:
    """Tests for Izhikevich neuron."""
    
    def test_initialization(self):
        """Test Izhikevich neuron initialization."""
        neuron = IzhikevichNeuron()
        assert neuron.neuron_type.value == "izhikevich"
    
    def test_regular_spiking(self):
        """Test regular spiking behavior."""
        neuron = IzhikevichNeuron(IzhikevichParameters(
            a=0.02, b=0.2, c=-65.0, d=8.0
        ))
        
        spikes = 0
        for _ in range(1000):
            V = neuron.update(0.1, 10.0)
            if len(neuron.state.spike_history) > spikes:
                spikes = len(neuron.state.spike_history)
        
        assert spikes > 0
    
    def test_preset_types(self):
        """Test Izhikevich preset neuron types."""
        from src.models.izhikevich import IzhikevichNeuronType
        
        for preset in IzhikevichNeuronType:
            neuron = IzhikevichNeuron.from_preset(preset)
            assert neuron is not None


class TestChemicalSynapse:
    """Tests for chemical synapses."""
    
    def test_initialization(self):
        """Test synapse initialization."""
        synapse = ChemicalSynapse(pre_id=0, post_id=1)
        assert synapse.pre_id == 0
        assert synapse.post_id == 1
    
    def test_conductance_update(self):
        """Test conductance updates."""
        synapse = ChemicalSynapse(pre_id=0, post_id=1)
        
        # Simulate presynaptic spike
        synapse.presynaptic_spike(10.0)
        
        # Update synapse
        for _ in range(100):
            synapse.update(0.1, -70.0, current_time=10.5)
        
        assert synapse.state.conductance >= 0
    
    def test_receptor_types(self):
        """Test different receptor types."""
        from src.synapses.chemical_synapse import SynapticReceptor
        
        for receptor in SynapticReceptor:
            synapse = ChemicalSynapse(receptor=receptor)
            assert synapse.receptor == receptor


class TestElectricalSynapse:
    """Tests for electrical synapses."""
    
    def test_initialization(self):
        """Test gap junction initialization."""
        gap = ElectricalSynapse(pre_id=0, post_id=1, conductance=2.0)
        assert gap.parameters.g_max == 2.0
    
    def test_bidirectional_current(self):
        """Test bidirectional current flow."""
        gap = ElectricalSynapse(pre_id=0, post_id=1, conductance=1.0)
        
        i_pre, i_post = gap.get_current(v_pre=-50.0, v_post=-70.0)
        
        # Currents should be equal and opposite
        assert i_pre == pytest.approx(-i_post, rel=0.001)


class TestNeuralNetwork:
    """Tests for neural networks."""
    
    def test_network_creation(self):
        """Test network creation."""
        config = NetworkConfig(
            num_neurons=50,
            topology=NetworkTopology.SPARSE_RANDOM
        )
        network = NeuralNetwork(config)
        
        assert len(network.neurons) == 50
        assert network.num_neurons == 50
    
    def test_network_step(self):
        """Test network simulation step."""
        config = NetworkConfig(num_neurons=20)
        network = NeuralNetwork(config)
        
        network.inject_current(np.ones(20) * 5.0)
        result = network.step(0.1)
        
        assert 'time' in result
        assert result['time'] == pytest.approx(0.1, rel=0.1)
    
    def test_connectivity(self):
        """Test network connectivity."""
        config = NetworkConfig(num_neurons=30, connection_probability=0.2)
        network = NeuralNetwork(config)
        
        assert len(network.synapses) > 0


class TestSimulator:
    """Tests for simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        config = SimulatorConfig(num_neurons=50)
        sim = Simulator(config)
        
        assert sim.config.num_neurons == 50
    
    def test_quick_simulation(self):
        """Test quick simulation function."""
        from src.simulator import run_simulation
        
        result = run_simulation(
            num_neurons=30,
            duration=100.0,
            dt=0.1,
            input_current=5.0
        )
        
        assert result is not None
        assert hasattr(result, 'spikes')


class TestSTDP:
    """Tests for STDP learning."""
    
    def test_stdp_creation(self):
        """Test STDP rule creation."""
        from src.learning import STDP
        
        stdp = STDP()
        assert stdp is not None
    
    def test_potentiation(self):
        """Test LTP."""
        from src.learning import STDP, STDPParameters
        
        params = STDPParameters(A_plus=0.01, A_minus=0.012)
        stdp = STDP(params)
        
        # Pre before post should cause potentiation
        stdp.record_presynaptic_spike('syn1', 0.0)
        stdp.record_postsynaptic_spike('syn1', 10.0)
        
        dw = stdp.compute_update('syn1', current_time=20.0)
        assert dw > 0  # Should be positive (potentiation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
