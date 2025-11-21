"""
Brain Simulation - Dynamic, Stateful Neural Network Simulation

This module implements a living, dynamic brain simulation that transitions
from the static atlas to a functioning spiking neural network.

Key features:
- Stateful neurons with dynamic voltage
- Synaptic connections with adjustable weights
- Simulation loop (heartbeat) for temporal evolution
- Spike detection and propagation
- Hebbian learning (synaptic plasticity)
- Sensory input mechanisms
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class NeuronType(Enum):
    """Types of neurons in the simulation"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    SENSORY = "sensory"
    MOTOR = "motor"


@dataclass
class IonConcentrations:
    """Ion concentrations inside and outside the neuron"""
    # Outside concentrations (mM)
    na_out: float = 145.0  # Sodium outside
    k_out: float = 4.0     # Potassium outside
    ca_out: float = 2.0    # Calcium outside
    cl_out: float = 110.0  # Chloride outside
    
    # Inside concentrations (mM)
    na_in: float = 12.0    # Sodium inside
    k_in: float = 155.0    # Potassium inside
    ca_in: float = 0.0001  # Calcium inside (very low at rest)
    cl_in: float = 4.0     # Chloride inside
    
    # Physical constants
    R: float = 8.314        # Gas constant (J/(mol·K))
    T: float = 310.15       # Temperature (37°C in Kelvin)
    F: float = 96485.0      # Faraday constant (C/mol)
    
    def calculate_nernst_potential(self, ion: str) -> float:
        """
        Calculate the Nernst equilibrium potential for a specific ion.
        E = (RT/zF) * ln([ion_out]/[ion_in])
        """
        z_values = {'na': 1, 'k': 1, 'ca': 2, 'cl': -1}  # Valence
        
        if ion == 'na':
            concentration_ratio = self.na_out / self.na_in
        elif ion == 'k':
            concentration_ratio = self.k_out / self.k_in
        elif ion == 'ca':
            concentration_ratio = self.ca_out / self.ca_in
        elif ion == 'cl':
            concentration_ratio = self.cl_out / self.cl_in
        else:
            raise ValueError(f"Unknown ion: {ion}")
        
        z = z_values[ion]
        # E = (RT/zF) * ln(ratio)
        # Convert to mV
        nernst_potential = (self.R * self.T / (z * self.F)) * np.log(concentration_ratio) * 1000
        return nernst_potential


class SimulatedNeuron:
    """
    A stateful neuron that simulates membrane potential dynamics.
    
    This is the core dynamic element that replaces static descriptions
    with actual voltage simulations.
    """
    
    def __init__(
        self,
        neuron_id: int,
        neuron_type: NeuronType = NeuronType.EXCITATORY,
        region: str = "Unknown"
    ):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.region = region
        
        # Dynamic state variables
        self.voltage_mv = -70.0  # Resting membrane potential
        self.ion_concentrations = IonConcentrations()
        
        # Spike mechanics
        self.threshold_mv = -55.0  # Spike threshold
        self.spike_peak_mv = 30.0  # Peak of action potential
        self.refractory_period_ms = 2.0  # Absolute refractory period
        self.time_since_spike_ms = 1000.0  # Large value initially (not refractory)
        self.is_spiking = False
        
        # Synaptic inputs (accumulated from incoming synapses)
        self.synaptic_input = 0.0
        
        # Membrane properties
        self.membrane_capacitance = 1.0  # μF/cm²
        self.leak_conductance = 0.3  # mS/cm²
        self.leak_reversal = -70.0  # mV
        
        # Spike history for analysis
        self.spike_times: List[float] = []
        self.voltage_history: List[float] = []
        
    def update(self, dt_ms: float) -> bool:
        """
        Update neuron state for one time step.
        
        Args:
            dt_ms: Time step in milliseconds
            
        Returns:
            True if neuron spiked in this time step
        """
        # Track time since last spike
        self.time_since_spike_ms += dt_ms
        
        # Check if in refractory period
        if self.time_since_spike_ms < self.refractory_period_ms:
            # During refractory period, voltage is clamped low
            self.voltage_mv = -80.0
            self.is_spiking = False
            self.voltage_history.append(self.voltage_mv)
            return False
        
        # Check if currently spiking (depolarization phase)
        if self.is_spiking:
            # Spike is happening - voltage rises to peak then immediately repolarizes
            if self.voltage_mv < self.spike_peak_mv:
                # Rising phase (very fast Na+ influx)
                self.voltage_mv += 100.0 * dt_ms  # Fast rise
            else:
                # Start repolarization
                self.is_spiking = False
                self.voltage_mv = self.spike_peak_mv
            self.voltage_history.append(self.voltage_mv)
            return False
        
        # Normal dynamics: integrate current to update voltage
        # I_total = I_leak + I_synaptic
        
        # Leak current (pulls voltage toward resting potential)
        i_leak = -self.leak_conductance * (self.voltage_mv - self.leak_reversal)
        
        # Synaptic current (from incoming connections)
        i_synaptic = self.synaptic_input
        
        # Total current
        i_total = i_leak + i_synaptic
        
        # Update voltage: dV/dt = I_total / C
        dv = (i_total / self.membrane_capacitance) * dt_ms
        self.voltage_mv += dv
        
        # Reset synaptic input for next time step
        self.synaptic_input = 0.0
        
        # Check for spike threshold crossing
        if self.voltage_mv >= self.threshold_mv:
            self.is_spiking = True
            self.time_since_spike_ms = 0.0
            self.spike_times.append(len(self.voltage_history) * dt_ms)
            self.voltage_history.append(self.voltage_mv)
            return True
        
        # Store voltage for history
        self.voltage_history.append(self.voltage_mv)
        return False
    
    def add_synaptic_input(self, current: float):
        """Add incoming synaptic current (can be positive or negative)"""
        self.synaptic_input += current
    
    def reset(self):
        """Reset neuron to resting state"""
        self.voltage_mv = -70.0
        self.time_since_spike_ms = 1000.0
        self.is_spiking = False
        self.synaptic_input = 0.0
        self.spike_times.clear()
        self.voltage_history.clear()


class Synapse:
    """
    Synaptic connection between two neurons with adjustable weight.
    
    Implements:
    - Synaptic transmission (spike -> current in post-synaptic neuron)
    - Hebbian learning (weight modification based on co-activity)
    """
    
    def __init__(
        self,
        pre_neuron: SimulatedNeuron,
        post_neuron: SimulatedNeuron,
        initial_weight: float = 1.0,
        delay_ms: float = 1.0
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = initial_weight  # Synaptic strength
        self.delay_ms = delay_ms  # Transmission delay
        
        # For Hebbian learning
        self.learning_rate = 0.01
        self.weight_max = 10.0
        self.weight_min = 0.0
        
        # Spike buffer for delay
        self.spike_buffer: List[Tuple[float, bool]] = []
        
    def transmit(self, current_time_ms: float, pre_spiked: bool):
        """
        Handle spike transmission with delay.
        
        Args:
            current_time_ms: Current simulation time
            pre_spiked: Whether presynaptic neuron spiked this time step
        """
        # Add new spike to buffer if it occurred
        if pre_spiked:
            self.spike_buffer.append((current_time_ms + self.delay_ms, True))
        
        # Check buffer for spikes that should arrive now
        arrived_spikes = [
            spike for spike in self.spike_buffer 
            if spike[0] <= current_time_ms
        ]
        
        # Remove arrived spikes from buffer
        self.spike_buffer = [
            spike for spike in self.spike_buffer 
            if spike[0] > current_time_ms
        ]
        
        # Apply synaptic current for each arrived spike
        for _ in arrived_spikes:
            # Synaptic current magnitude depends on weight and neuron type
            if self.pre_neuron.neuron_type == NeuronType.INHIBITORY:
                current = -self.weight * 8.0  # Inhibitory (negative current)
            else:  # Excitatory, Sensory, and Motor are all excitatory
                current = self.weight * 8.0  # Excitatory (positive current)
            
            self.post_neuron.add_synaptic_input(current)
    
    def apply_hebbian_learning(self, pre_spiked: bool, post_spiked: bool):
        """
        Apply Hebbian learning rule: "Cells that fire together, wire together"
        
        If pre and post neurons spike close in time, strengthen the connection.
        """
        # Only apply learning when there is presynaptic activity
        if not pre_spiked:
            return
        
        # Simple Hebbian rule: if both active, increase weight
        if post_spiked:
            # LTP: Long-Term Potentiation - both neurons spiked!
            self.weight += self.learning_rate * (self.weight_max - self.weight)
        else:
            # Very weak LTD: presynaptic activity without postsynaptic response
            # Make this much weaker to avoid overpowering LTP
            self.weight -= self.learning_rate * 0.005 * self.weight
        
        # Clamp weight to valid range
        self.weight = np.clip(self.weight, self.weight_min, self.weight_max)


class BrainSimulation:
    """
    The main simulation engine - the "heartbeat" of the brain.
    
    Manages:
    - Collection of neurons
    - Synaptic connections
    - Temporal evolution (simulation loop)
    - Sensory input
    - Analysis and visualization
    """
    
    def __init__(self, dt_ms: float = 0.1):
        """
        Initialize the brain simulation.
        
        Args:
            dt_ms: Time step for simulation in milliseconds
        """
        self.dt_ms = dt_ms
        self.current_time_ms = 0.0
        self.alive = True
        
        # Neural network components
        self.neurons: Dict[int, SimulatedNeuron] = {}
        self.synapses: List[Synapse] = []
        
        # Neuron groups by region (for organization)
        self.neuron_groups: Dict[str, List[int]] = {}
        
        # Simulation statistics
        self.total_spikes = 0
        self.step_count = 0
        
    def add_neuron(
        self,
        neuron_id: int,
        neuron_type: NeuronType = NeuronType.EXCITATORY,
        region: str = "Unknown"
    ) -> SimulatedNeuron:
        """Add a neuron to the simulation"""
        neuron = SimulatedNeuron(neuron_id, neuron_type, region)
        self.neurons[neuron_id] = neuron
        
        # Add to region group
        if region not in self.neuron_groups:
            self.neuron_groups[region] = []
        self.neuron_groups[region].append(neuron_id)
        
        return neuron
    
    def connect_neurons(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 1.0,
        delay_ms: float = 1.0
    ) -> Synapse:
        """Create a synaptic connection between two neurons"""
        if pre_id not in self.neurons or post_id not in self.neurons:
            raise ValueError("Both neurons must exist in the simulation")
        
        synapse = Synapse(
            self.neurons[pre_id],
            self.neurons[post_id],
            weight,
            delay_ms
        )
        self.synapses.append(synapse)
        return synapse
    
    def create_simple_network(
        self,
        n_sensory: int = 10,
        n_processing: int = 20,
        n_output: int = 5
    ):
        """
        Create a simple 3-layer network (sensory -> processing -> output)
        This mimics: Retina -> V1 -> Higher visual areas
        """
        neuron_id = 0
        
        # Layer 1: Sensory neurons (e.g., Retina)
        sensory_ids = []
        for i in range(n_sensory):
            self.add_neuron(neuron_id, NeuronType.SENSORY, "Retina")
            sensory_ids.append(neuron_id)
            neuron_id += 1
        
        # Layer 2: Processing neurons (e.g., V1 - Primary Visual Cortex)
        processing_ids = []
        for i in range(n_processing):
            # Mix of excitatory and inhibitory
            ntype = NeuronType.EXCITATORY if i % 5 != 0 else NeuronType.INHIBITORY
            self.add_neuron(neuron_id, ntype, "V1")
            processing_ids.append(neuron_id)
            neuron_id += 1
        
        # Layer 3: Output neurons (e.g., Higher visual areas)
        output_ids = []
        for i in range(n_output):
            self.add_neuron(neuron_id, NeuronType.EXCITATORY, "Visual_Association")
            output_ids.append(neuron_id)
            neuron_id += 1
        
        # Connect layers with random connectivity
        # Sensory -> Processing
        for pre_id in sensory_ids:
            # Each sensory neuron connects to ~50% of processing neurons
            for post_id in processing_ids:
                if np.random.random() < 0.5:
                    weight = np.random.uniform(1.5, 3.5)  # Stronger weights
                    self.connect_neurons(pre_id, post_id, weight, delay_ms=1.0)
        
        # Processing -> Output (need STRONG weights for deep propagation)
        for pre_id in processing_ids:
            # Each processing neuron connects to ALL output neurons for better propagation
            for post_id in output_ids:
                weight = np.random.uniform(3.0, 5.0)  # Very strong weights
                self.connect_neurons(pre_id, post_id, weight, delay_ms=1.0)
        
        print(f"Created simple network:")
        print(f"  Sensory neurons (Retina): {n_sensory}")
        print(f"  Processing neurons (V1): {n_processing}")
        print(f"  Output neurons (Visual Association): {n_output}")
        print(f"  Total synapses: {len(self.synapses)}")
    
    def inject_current(self, neuron_id: int, current: float):
        """Inject external current into a neuron (for sensory input)"""
        if neuron_id in self.neurons:
            self.neurons[neuron_id].add_synaptic_input(current)
    
    def stimulate_region(self, region: str, current: float):
        """Stimulate all neurons in a region with external current"""
        if region in self.neuron_groups:
            for neuron_id in self.neuron_groups[region]:
                self.inject_current(neuron_id, current)
    
    def simulate_step(self, enable_learning: bool = True) -> Dict[str, int]:
        """
        Execute one time step of the simulation.
        
        Returns:
            Dictionary with spike counts per region
        """
        spike_counts = {}
        neuron_spike_status = {}
        
        # Update all neurons
        for neuron_id, neuron in self.neurons.items():
            spiked = neuron.update(self.dt_ms)
            neuron_spike_status[neuron_id] = spiked
            
            if spiked:
                self.total_spikes += 1
                region = neuron.region
                spike_counts[region] = spike_counts.get(region, 0) + 1
        
        # Process all synapses
        for synapse in self.synapses:
            pre_id = synapse.pre_neuron.neuron_id
            post_id = synapse.post_neuron.neuron_id
            
            # Transmit spikes
            synapse.transmit(self.current_time_ms, neuron_spike_status[pre_id])
            
            # Apply learning if enabled
            if enable_learning:
                synapse.apply_hebbian_learning(
                    neuron_spike_status[pre_id],
                    neuron_spike_status[post_id]
                )
        
        # Advance time
        self.current_time_ms += self.dt_ms
        self.step_count += 1
        
        return spike_counts
    
    def run(self, duration_ms: float, enable_learning: bool = True, verbose: bool = False):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration_ms: How long to simulate (in milliseconds)
            enable_learning: Whether to apply Hebbian learning
            verbose: Print progress information
        """
        start_time = self.current_time_ms
        steps = int(duration_ms / self.dt_ms)
        
        if verbose:
            print(f"\nRunning simulation for {duration_ms} ms ({steps} steps)...")
            print(f"Time step: {self.dt_ms} ms")
            print(f"Learning: {'enabled' if enable_learning else 'disabled'}")
        
        total_spike_counts = {}
        
        for step in range(steps):
            spike_counts = self.simulate_step(enable_learning)
            
            # Accumulate spike counts
            for region, count in spike_counts.items():
                total_spike_counts[region] = total_spike_counts.get(region, 0) + count
            
            # Progress reporting
            if verbose and step % 1000 == 0:
                elapsed = self.current_time_ms - start_time
                print(f"  Step {step}/{steps} | Time: {elapsed:.1f} ms | Spikes this step: {sum(spike_counts.values())}")
        
        if verbose:
            print(f"\nSimulation complete!")
            print(f"Total spikes by region:")
            for region, count in sorted(total_spike_counts.items()):
                print(f"  {region}: {count}")
    
    def get_neuron_voltage_trace(self, neuron_id: int) -> List[float]:
        """Get the voltage history of a specific neuron"""
        if neuron_id in self.neurons:
            return self.neurons[neuron_id].voltage_history
        return []
    
    def get_spike_times(self, neuron_id: int) -> List[float]:
        """Get spike times for a specific neuron"""
        if neuron_id in self.neurons:
            return self.neurons[neuron_id].spike_times
        return []
    
    def get_statistics(self) -> Dict[str, any]:
        """Get simulation statistics"""
        return {
            'current_time_ms': self.current_time_ms,
            'total_neurons': len(self.neurons),
            'total_synapses': len(self.synapses),
            'total_spikes': self.total_spikes,
            'step_count': self.step_count,
            'average_firing_rate_hz': (self.total_spikes / len(self.neurons) / (self.current_time_ms / 1000.0)) if self.current_time_ms > 0 else 0
        }
    
    def reset(self):
        """Reset the simulation to initial state"""
        for neuron in self.neurons.values():
            neuron.reset()
        for synapse in self.synapses:
            synapse.spike_buffer.clear()
        self.current_time_ms = 0.0
        self.total_spikes = 0
        self.step_count = 0
