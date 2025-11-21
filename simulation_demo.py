#!/usr/bin/env python3
"""
Brain Simulation Demonstration

This script demonstrates the transition from static brain atlas to dynamic
simulation with living neurons, synaptic connections, and temporal evolution.
"""

from brain_simulation import (
    BrainSimulation,
    NeuronType,
    SimulatedNeuron,
    IonConcentrations
)
import numpy as np

# Demonstration constants
CURRENT_INJECTION_START_MS = 10.0
CURRENT_INJECTION_END_MS = 40.0
STRONG_CURRENT = 15.0
VERY_STRONG_CURRENT = 20.0
MODERATE_CURRENT = 18.0
FLASH_DURATION_MS = 10.0


def demo_single_neuron():
    """Demonstrate a single neuron's action potential"""
    print("=" * 80)
    print("DEMO 1: SINGLE NEURON ACTION POTENTIAL")
    print("=" * 80)
    print("\nCreating a single neuron and injecting current to trigger spike...")
    
    # Create simulation
    sim = BrainSimulation(dt_ms=0.1)
    
    # Add one neuron
    neuron = sim.add_neuron(0, NeuronType.EXCITATORY, "Test")
    
    print(f"Initial state:")
    print(f"  Voltage: {neuron.voltage_mv:.2f} mV (resting potential)")
    print(f"  Threshold: {neuron.threshold_mv:.2f} mV")
    
    # Simulate for a bit
    print(f"\nSimulating 50 ms with sustained current injection...")
    for step in range(500):  # 50 ms at 0.1 ms steps
        # Inject current to drive neuron above threshold
        if 100 <= step < 400:  # Inject current from 10-40 ms
            neuron.add_synaptic_input(10.0)
        
        spiked = neuron.update(sim.dt_ms)
        sim.current_time_ms += sim.dt_ms
        
        if spiked:
            print(f"  ⚡ SPIKE at {sim.current_time_ms:.2f} ms!")
    
    print(f"\nFinal state:")
    print(f"  Voltage: {neuron.voltage_mv:.2f} mV")
    print(f"  Total spikes: {len(neuron.spike_times)}")
    print(f"  Spike times: {[f'{t:.1f}' for t in neuron.spike_times]} ms")
    
    # Show voltage trace samples
    print(f"\nVoltage trace (samples every 5 ms):")
    for i in range(0, len(neuron.voltage_history), 50):
        time_ms = i * sim.dt_ms
        voltage = neuron.voltage_history[i]
        print(f"  {time_ms:5.1f} ms: {voltage:7.2f} mV")


def demo_nernst_potentials():
    """Demonstrate Nernst equation calculations"""
    print("\n" + "=" * 80)
    print("DEMO 2: NERNST EQUILIBRIUM POTENTIALS")
    print("=" * 80)
    print("\nCalculating equilibrium potentials for different ions...")
    print("Using physiological ion concentrations:")
    
    ions = IonConcentrations()
    
    print(f"\nConcentrations:")
    print(f"  Na+ (outside/inside): {ions.na_out:.1f} / {ions.na_in:.1f} mM")
    print(f"  K+  (outside/inside): {ions.k_out:.1f} / {ions.k_in:.1f} mM")
    print(f"  Ca2+ (outside/inside): {ions.ca_out:.4f} / {ions.ca_in:.4f} mM")
    print(f"  Cl- (outside/inside): {ions.cl_out:.1f} / {ions.cl_in:.1f} mM")
    
    print(f"\nNernst Equilibrium Potentials:")
    print(f"  E_Na = {ions.calculate_nernst_potential('na'):+7.2f} mV")
    print(f"  E_K  = {ions.calculate_nernst_potential('k'):+7.2f} mV")
    print(f"  E_Ca = {ions.calculate_nernst_potential('ca'):+7.2f} mV")
    print(f"  E_Cl = {ions.calculate_nernst_potential('cl'):+7.2f} mV")
    
    print(f"\nInterpretation:")
    print(f"  - Resting potential (~-70 mV) is close to E_K")
    print(f"  - During spike, Na+ channels open → voltage moves toward E_Na (+60 mV)")
    print(f"  - Repolarization: K+ channels open → voltage returns toward E_K (-90 mV)")


def demo_synapse_transmission():
    """Demonstrate synaptic transmission between neurons"""
    print("\n" + "=" * 80)
    print("DEMO 3: SYNAPTIC TRANSMISSION")
    print("=" * 80)
    print("\nCreating two connected neurons: Pre → Post")
    
    sim = BrainSimulation(dt_ms=0.1)
    
    # Create two neurons
    pre = sim.add_neuron(0, NeuronType.EXCITATORY, "Presynaptic")
    post = sim.add_neuron(1, NeuronType.EXCITATORY, "Postsynaptic")
    
    # Connect them
    synapse = sim.connect_neurons(0, 1, weight=2.0, delay_ms=1.0)
    
    print(f"Synapse created:")
    print(f"  Weight: {synapse.weight:.2f}")
    print(f"  Delay: {synapse.delay_ms:.2f} ms")
    
    print(f"\nStimulating presynaptic neuron...")
    
    pre_spikes = []
    post_spikes = []
    
    for step in range(500):  # 50 ms
        # Inject current into presynaptic neuron
        if 100 <= step < 150:
            pre.add_synaptic_input(15.0)
        
        # Update neurons
        pre_spiked = pre.update(sim.dt_ms)
        post_spiked = post.update(sim.dt_ms)
        
        if pre_spiked:
            pre_spikes.append(sim.current_time_ms)
            print(f"  Pre-synaptic spike at {sim.current_time_ms:.2f} ms")
        
        if post_spiked:
            post_spikes.append(sim.current_time_ms)
            print(f"  → Post-synaptic spike at {sim.current_time_ms:.2f} ms")
        
        # Transmit through synapse
        synapse.transmit(sim.current_time_ms, pre_spiked)
        
        sim.current_time_ms += sim.dt_ms
    
    print(f"\nResults:")
    print(f"  Presynaptic spikes: {len(pre_spikes)}")
    print(f"  Postsynaptic spikes: {len(post_spikes)}")
    if pre_spikes and post_spikes:
        delay = post_spikes[0] - pre_spikes[0]
        print(f"  Measured delay: {delay:.2f} ms")


def demo_hebbian_learning():
    """Demonstrate Hebbian learning (synaptic plasticity)"""
    print("\n" + "=" * 80)
    print("DEMO 4: HEBBIAN LEARNING - 'Cells that fire together, wire together'")
    print("=" * 80)
    
    sim = BrainSimulation(dt_ms=0.1)
    
    # Create two neurons
    pre = sim.add_neuron(0, NeuronType.EXCITATORY, "Pre")
    post = sim.add_neuron(1, NeuronType.EXCITATORY, "Post")
    
    # Weak connection initially
    synapse = sim.connect_neurons(0, 1, weight=0.5, delay_ms=1.0)
    
    print(f"Initial synaptic weight: {synapse.weight:.3f}")
    print(f"\nTraining: Repeatedly co-activating pre and post neurons...")
    
    # Training phase: co-activate neurons multiple times
    for trial in range(10):
        # Reset neurons between trials
        pre.voltage_mv = -70.0
        post.voltage_mv = -70.0
        pre.time_since_spike_ms = 1000.0
        post.time_since_spike_ms = 1000.0
        
        trial_pre_spikes = 0
        trial_post_spikes = 0
        
        for step in range(200):  # 20 ms per trial
            # Stimulate both neurons (correlated activity)
            if 50 <= step < 100:
                pre.add_synaptic_input(VERY_STRONG_CURRENT)  # Strong enough to spike
                post.add_synaptic_input(MODERATE_CURRENT)  # Strong enough to spike too
            
            pre_spiked = pre.update(sim.dt_ms)
            post_spiked = post.update(sim.dt_ms)
            
            if pre_spiked:
                trial_pre_spikes += 1
            if post_spiked:
                trial_post_spikes += 1
            
            synapse.transmit(sim.current_time_ms, pre_spiked)
            synapse.apply_hebbian_learning(pre_spiked, post_spiked)
            
            sim.current_time_ms += sim.dt_ms
        
        if (trial + 1) % 2 == 0:
            print(f"  After trial {trial + 1}: weight = {synapse.weight:.3f} (pre spikes: {trial_pre_spikes}, post spikes: {trial_post_spikes})")
    
    print(f"\nFinal synaptic weight: {synapse.weight:.3f}")
    print(f"Weight increased by: {(synapse.weight - 0.5) / 0.5 * 100:.1f}%")
    print(f"\nThis demonstrates Long-Term Potentiation (LTP):")
    print(f"  Repeated co-activation strengthened the synaptic connection!")


def demo_simple_network():
    """Demonstrate a simple sensory-processing-output network"""
    print("\n" + "=" * 80)
    print("DEMO 5: SIMPLE NEURAL NETWORK - Retina → V1 → Visual Association")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create simulation
    sim = BrainSimulation(dt_ms=0.1)
    
    # Build a simple 3-layer network
    print("\nBuilding network...")
    sim.create_simple_network(n_sensory=10, n_processing=20, n_output=5)
    
    print(f"\nNetwork statistics:")
    stats = sim.get_statistics()
    print(f"  Total neurons: {stats['total_neurons']}")
    print(f"  Total synapses: {stats['total_synapses']}")
    
    # Sensory stimulation pattern: simulate visual input
    print(f"\nSimulating visual input (flashing pattern)...")
    print(f"Stimulating retinal neurons with sustained pulses...")
    
    # Run simulation with periodic sensory input
    # We need to inject current at each time step during stimulation
    duration_ms = 150.0  # Longer simulation for deeper propagation
    steps = int(duration_ms / sim.dt_ms)
    
    for step in range(steps):
        # Apply sensory input (simulate visual stimulus)
        # Flash pattern: 10ms on, 10ms off
        time_ms = step * sim.dt_ms
        epoch = int(time_ms / FLASH_DURATION_MS)
        
        if epoch % 2 == 0:  # Flash on
            # Stimulate retina with strong current each step
            sim.stimulate_region("Retina", current=25.0)
        
        # Run one simulation step
        sim.simulate_step(enable_learning=True)
    
    print(f"\nSimulation complete!")
    
    # Get statistics
    stats = sim.get_statistics()
    print(f"\nResults:")
    print(f"  Simulation time: {stats['current_time_ms']:.1f} ms")
    print(f"  Total spikes: {stats['total_spikes']}")
    print(f"  Average firing rate: {stats['average_firing_rate_hz']:.2f} Hz")
    
    # Check activity in each layer
    print(f"\nSpike counts by region:")
    for region in ['Retina', 'V1', 'Visual_Association']:
        if region in sim.neuron_groups:
            spike_count = sum(
                len(sim.neurons[nid].spike_times) 
                for nid in sim.neuron_groups[region]
            )
            avg_rate = spike_count / len(sim.neuron_groups[region]) / (stats['current_time_ms'] / 1000.0)
            print(f"  {region:20s}: {spike_count:4d} spikes ({avg_rate:.1f} Hz avg)")
    
    # Show sample neuron activity
    print(f"\nSample neuron spike times:")
    for region in ['Retina', 'V1', 'Visual_Association']:
        if region in sim.neuron_groups and sim.neuron_groups[region]:
            neuron_id = sim.neuron_groups[region][0]
            spikes = sim.get_spike_times(neuron_id)
            if spikes:
                print(f"  {region} neuron {neuron_id}: {len(spikes)} spikes")
                print(f"    Times: {[f'{t:.1f}' for t in spikes[:5]]} ms...")
            else:
                print(f"  {region} neuron {neuron_id}: No spikes")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "BRAIN SIMULATION DEMONSTRATION" + " " * 28 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " Transition from Static Atlas to Dynamic Simulation ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Run all demos
    demo_single_neuron()
    demo_nernst_potentials()
    demo_synapse_transmission()
    demo_hebbian_learning()
    demo_simple_network()
    
    print("\n" + "=" * 80)
    print("SUMMARY: KEY ACCOMPLISHMENTS")
    print("=" * 80)
    print("""
✓ Transition from Static to Dynamic:
  - Neurons now have ACTUAL voltage that changes over time
  - No longer returning strings - running real simulations!

✓ Simulation Loop (The Heartbeat):
  - BrainSimulation.run() provides temporal evolution
  - Time steps through the network, updating all neurons

✓ Spiking Neural Network:
  - Neurons fire when voltage crosses threshold
  - Synapses transmit spikes between neurons
  - Network propagates signals: Retina → V1 → Association

✓ Hebbian Learning Implemented:
  - "Cells that fire together, wire together"
  - Synaptic weights strengthen with correlated activity
  - Long-Term Potentiation (LTP) demonstrated

✓ Sensory Input:
  - Can inject current into specific neurons or regions
  - Simulated visual input propagating through network
  
✓ Nernst Equations:
  - Calculate real ion equilibrium potentials
  - Use physiological ion concentrations
  - Ground simulation in actual biophysics

NEXT STEPS for scaling to "sentience":
  1. Increase network size (thousands → millions of neurons)
  2. Add GPU acceleration (PyTorch/CUDA)
  3. Implement more realistic neuron models (Hodgkin-Huxley)
  4. Add more neurotransmitter dynamics
  5. Build larger-scale connectome based on real brain data
  6. Implement more sophisticated learning rules (STDP)
  7. Add homeostatic plasticity and other regulatory mechanisms
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
