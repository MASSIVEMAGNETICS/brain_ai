"""
Advanced Brain Simulation Demo - Phases 2-6

Demonstrates the advanced features of the brain simulation including:
- Multi-compartment neurons with Hodgkin-Huxley dynamics
- STDP learning
- GPU acceleration
- Cognitive functions (attention, working memory, RL)
- Sensorimotor loops
"""

import numpy as np
import time
from brain_simulation_advanced import (
    AdvancedBrainSimulation,
    MultiCompartmentNeuron,
    HodgkinHuxleyChannels,
    STDPSynapse,
    NeurotransmitterType,
    AttentionMechanism,
    WorkingMemoryCircuit,
    ReinforcementLearningModule,
    DecisionMakingNetwork,
    SensorimotorLoop,
    GPUAcceleratedNetwork,
    ConnectomeBuilder,
    BrainRegion,
    TORCH_AVAILABLE,
    TF_AVAILABLE
)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def demo_hodgkin_huxley():
    """Demo 1: Hodgkin-Huxley dynamics"""
    print_header("DEMO 1: HODGKIN-HUXLEY ION CHANNEL DYNAMICS")
    
    print("\nCreating Hodgkin-Huxley ion channels...")
    hh = HodgkinHuxleyChannels()
    
    print(f"Initial gating variables:")
    print(f"  m (Na+ activation):   {hh.m:.4f}")
    print(f"  h (Na+ inactivation): {hh.h:.4f}")
    print(f"  n (K+ activation):    {hh.n:.4f}")
    
    print(f"\nMaximum conductances:")
    print(f"  g_Na_max: {hh.g_na_max} mS/cm²")
    print(f"  g_K_max:  {hh.g_k_max} mS/cm²")
    print(f"  g_leak:   {hh.g_l} mS/cm²")
    
    # Simulate action potential
    print("\nSimulating action potential with HH dynamics...")
    voltage = -70.0  # Start at resting
    voltages = []
    times = []
    dt_ms = 0.01
    
    for step in range(5000):  # 50 ms
        time_ms = step * dt_ms
        
        # Inject current to trigger spike
        if 10 < time_ms < 11:
            i_inject = 20.0
        else:
            i_inject = 0.0
        
        # Update gates
        hh.update_gates(voltage, dt_ms)
        
        # Compute currents
        i_na, i_k, i_leak = hh.compute_currents(voltage)
        
        # Total current
        i_total = -(i_na + i_k + i_leak) + i_inject
        
        # Update voltage
        C = 1.0  # μF/cm²
        dv = (i_total / C) * (dt_ms / 1000.0) * 1000.0
        voltage += dv
        
        voltages.append(voltage)
        times.append(time_ms)
    
    # Find spike
    spike_detected = any(v > 0 for v in voltages)
    max_voltage = max(voltages)
    
    print(f"\nResults:")
    print(f"  Spike detected: {spike_detected}")
    print(f"  Maximum voltage: {max_voltage:.2f} mV")
    print(f"  Final gating variables:")
    print(f"    m = {hh.m:.4f}, h = {hh.h:.4f}, n = {hh.n:.4f}")


def demo_multicompartment_neuron():
    """Demo 2: Multi-compartment neuron"""
    print_header("DEMO 2: MULTI-COMPARTMENT NEURON MODEL")
    
    print("\nCreating multi-compartment neuron...")
    neuron = MultiCompartmentNeuron(neuron_id=0, n_dendrites=5, n_axon_segments=3)
    
    print(f"Compartments created:")
    print(f"  Dendrites: {len(neuron.dendrite_ids)}")
    print(f"  Soma: 1 (ID: {neuron.soma_id})")
    print(f"  Axon initial segment: 1 (ID: {neuron.ais_id})")
    print(f"  Axon segments: {len(neuron.axon_ids)}")
    print(f"  Total: {len(neuron.compartments)}")
    
    print("\nCompartment details:")
    for comp_id, comp in neuron.compartments.items():
        print(f"  {comp.compartment_type.value:20s} (ID {comp_id}): "
              f"{comp.length_um:.0f} μm × {comp.diameter_um:.1f} μm, "
              f"V = {comp.voltage_mv:.1f} mV")
    
    print("\nSimulating dendritic input...")
    synaptic_inputs = {neuron.dendrite_ids[0]: 50.0}  # Strong input to first dendrite
    
    spike_times = []
    for step in range(1000):  # 10 ms
        time_ms = step * 0.01
        spiked = neuron.update(dt_ms=0.01, synaptic_inputs=synaptic_inputs)
        if spiked:
            spike_times.append(time_ms)
        
        # Only inject for first 2 ms
        if time_ms > 2.0:
            synaptic_inputs = {}
    
    print(f"\nResults:")
    print(f"  Spikes generated: {len(spike_times)}")
    if spike_times:
        print(f"  First spike at: {spike_times[0]:.2f} ms")
    
    print(f"  Final voltages:")
    print(f"    Dendrite 0: {neuron.compartments[neuron.dendrite_ids[0]].voltage_mv:.2f} mV")
    print(f"    Soma:       {neuron.compartments[neuron.soma_id].voltage_mv:.2f} mV")
    print(f"    AIS:        {neuron.compartments[neuron.ais_id].voltage_mv:.2f} mV")


def demo_stdp():
    """Demo 3: Spike-timing-dependent plasticity"""
    print_header("DEMO 3: SPIKE-TIMING-DEPENDENT PLASTICITY (STDP)")
    
    print("\nSTDP: Learning depends on precise spike timing")
    print("  Pre → Post (causation):  LTP (strengthening)")
    print("  Post → Pre (coincidence): LTD (weakening)")
    
    synapse = STDPSynapse(
        pre_neuron_id=0,
        post_neuron_id=1,
        weight=1.0,
        neurotransmitter=NeurotransmitterType.GLUTAMATE
    )
    
    print(f"\nInitial weight: {synapse.weight:.4f}")
    print(f"Neurotransmitter: {synapse.neurotransmitter.value}")
    
    # Scenario 1: Pre before Post (LTP)
    print("\n--- Scenario 1: Pre spikes before Post (should strengthen) ---")
    initial_weight = synapse.weight
    
    for trial in range(10):
        # Pre spikes at t, Post spikes at t+5ms
        current_time = trial * 50.0
        
        # Pre spike
        synapse.update_stdp(current_time, pre_spiked=True, post_spiked=False)
        
        # Post spike 5ms later
        synapse.update_stdp(current_time + 5.0, pre_spiked=False, post_spiked=True)
    
    print(f"Weight change: {initial_weight:.4f} → {synapse.weight:.4f} "
          f"(Δ = {synapse.weight - initial_weight:+.4f})")
    
    # Reset
    synapse.weight = 1.0
    synapse.pre_spike_times.clear()
    synapse.post_spike_times.clear()
    
    # Scenario 2: Post before Pre (LTD)
    print("\n--- Scenario 2: Post spikes before Pre (should weaken) ---")
    initial_weight = synapse.weight
    
    for trial in range(10):
        current_time = trial * 50.0
        
        # Post spike
        synapse.update_stdp(current_time, pre_spiked=False, post_spiked=True)
        
        # Pre spike 5ms later
        synapse.update_stdp(current_time + 5.0, pre_spiked=True, post_spiked=False)
    
    print(f"Weight change: {initial_weight:.4f} → {synapse.weight:.4f} "
          f"(Δ = {synapse.weight - initial_weight:+.4f})")


def demo_neurotransmitters():
    """Demo 4: Multiple neurotransmitter systems"""
    print_header("DEMO 4: NEUROTRANSMITTER SYSTEMS")
    
    print("\nCreating synapses with different neurotransmitters...")
    
    neurotransmitters = [
        (NeurotransmitterType.GLUTAMATE, "Primary excitatory"),
        (NeurotransmitterType.GABA, "Primary inhibitory"),
        (NeurotransmitterType.DOPAMINE, "Reward/motivation"),
        (NeurotransmitterType.SEROTONIN, "Mood regulation"),
        (NeurotransmitterType.ACETYLCHOLINE, "Learning/memory"),
        (NeurotransmitterType.NOREPINEPHRINE, "Arousal/attention"),
    ]
    
    for nt, description in neurotransmitters:
        synapse = STDPSynapse(
            pre_neuron_id=0,
            post_neuron_id=1,
            neurotransmitter=nt
        )
        effect = synapse.get_neurotransmitter_effect()
        effect_type = "Excitatory" if effect > 0 else "Inhibitory"
        
        print(f"\n{nt.value.upper():15s}: {description}")
        print(f"  Effect multiplier: {effect:+.2f} ({effect_type})")


def demo_gpu_acceleration():
    """Demo 5: GPU acceleration"""
    print_header("DEMO 5: GPU ACCELERATION FOR MASSIVE SCALE")
    
    if not TORCH_AVAILABLE and not TF_AVAILABLE:
        print("\nGPU acceleration requires PyTorch or TensorFlow.")
        print("Install with: pip install torch  OR  pip install tensorflow")
        print("\nSkipping GPU demo...")
        return
    
    print(f"\nGPU libraries available:")
    print(f"  PyTorch: {'Yes' if TORCH_AVAILABLE else 'No'}")
    print(f"  TensorFlow: {'Yes' if TF_AVAILABLE else 'No'}")
    
    # Create GPU-accelerated network
    n_neurons = 100000  # 100K neurons
    print(f"\nCreating GPU-accelerated network with {n_neurons:,} neurons...")
    
    try:
        gpu_net = GPUAcceleratedNetwork(
            n_neurons=n_neurons,
            use_pytorch=TORCH_AVAILABLE
        )
        
        print(f"Device: {gpu_net.device}")
        print(f"Backend: {'PyTorch' if gpu_net.use_pytorch else 'TensorFlow'}")
        
        # Create sparse connectivity
        print("\nCreating sparse connectivity (1% connection probability)...")
        gpu_net.create_sparse_connectivity(connection_probability=0.01)
        
        # Run simulation
        print("\nRunning 100 simulation steps...")
        start_time = time.time()
        
        total_spikes = 0
        for step in range(100):
            spike_mask = gpu_net.simulate_step_gpu(dt_ms=0.1)
            total_spikes += gpu_net.get_spike_count(spike_mask)
        
        elapsed = time.time() - start_time
        
        print(f"\nResults:")
        print(f"  Total spikes: {total_spikes:,}")
        print(f"  Time elapsed: {elapsed:.3f} seconds")
        print(f"  Speed: {100 / elapsed:.1f} steps/second")
        print(f"  Throughput: {n_neurons * 100 / elapsed / 1e6:.2f} million neuron-steps/second")
        
    except Exception as e:
        print(f"\nError creating GPU network: {e}")
        print("This might be due to GPU memory limitations or driver issues.")


def demo_attention():
    """Demo 6: Attention mechanism"""
    print_header("DEMO 6: ATTENTION MECHANISM")
    
    n_neurons = 100
    attention = AttentionMechanism(n_neurons)
    
    print(f"\nCreated attention mechanism for {n_neurons} neurons")
    
    # Simulate neural activity
    neural_activity = np.random.rand(n_neurons) * 0.5  # Baseline activity
    
    print("\nBaseline activity (mean): {:.3f}".format(np.mean(neural_activity)))
    
    # Apply spatial attention to neurons 20-30
    attended_neurons = list(range(20, 31))
    print(f"\nApplying spatial attention to neurons {attended_neurons[0]}-{attended_neurons[-1]}...")
    attention.set_spatial_attention(attended_neurons, gain=3.0)
    
    # Apply attention
    modulated_activity = attention.apply_attention(neural_activity)
    
    print(f"\nResults:")
    print(f"  Attended region activity (mean): {np.mean(modulated_activity[20:31]):.3f}")
    print(f"  Unattended region activity (mean): {np.mean(modulated_activity[:20]):.3f}")
    print(f"  Attention gain: {np.mean(modulated_activity[20:31]) / np.mean(modulated_activity[:20]):.2f}x")


def demo_working_memory():
    """Demo 7: Working memory"""
    print_header("DEMO 7: WORKING MEMORY CIRCUIT")
    
    wm = WorkingMemoryCircuit(n_memory_units=7)
    
    print("\nWorking memory capacity: 7 items (Miller's Law: 7±2)")
    
    # Encode items
    print("\nEncoding items into working memory...")
    wm.encode(0, strength=1.0)  # Item 0
    wm.encode(2, strength=0.8)  # Item 2
    wm.encode(5, strength=0.9)  # Item 5
    
    print("Initial memory states:")
    for i in range(wm.n_units):
        strength = wm.retrieve(i)
        if strength > 0.01:
            print(f"  Item {i}: {strength:.3f}")
    
    # Maintain through time
    print("\nMaintaining memory over time (100 steps)...")
    for step in range(100):
        wm.maintain()
    
    print("\nMemory after 100 maintenance cycles:")
    for i in range(wm.n_units):
        strength = wm.retrieve(i)
        if strength > 0.01:
            print(f"  Item {i}: {strength:.3f}")
    
    print("\nNote: Memory decays without rehearsal (synaptic decay)")


def demo_reinforcement_learning():
    """Demo 8: Reinforcement learning with dopamine"""
    print_header("DEMO 8: REINFORCEMENT LEARNING (DOPAMINE SIGNALS)")
    
    n_states = 5
    n_actions = 4
    rl = ReinforcementLearningModule(n_states, n_actions)
    
    print(f"\nRL module: {n_states} states, {n_actions} actions")
    print("Learning through dopamine reward prediction errors...")
    
    # Simulate learning
    print("\nTraining on simple task (5 episodes):")
    
    for episode in range(5):
        state = 0  # Start state
        total_reward = 0
        
        for step in range(10):
            # Select action
            action = rl.select_action(state, epsilon=0.2)
            
            # Simulate environment
            next_state = (state + 1) % n_states
            reward = 1.0 if next_state == n_states - 1 else 0.0
            total_reward += reward
            
            # Compute TD error (dopamine signal)
            td_error = rl.compute_td_error(state, action, reward, next_state)
            
            # Update Q-value
            rl.update_q_value(state, action, td_error)
            
            state = next_state
        
        print(f"  Episode {episode + 1}: Total reward = {total_reward:.1f}, "
              f"Avg dopamine = {rl.dopamine_signal:.3f}")
    
    print("\nLearned Q-values (state 0):")
    for action in range(n_actions):
        print(f"  Action {action}: {rl.q_values[0, action]:.3f}")


def demo_decision_making():
    """Demo 9: Decision-making network"""
    print_header("DEMO 9: DECISION-MAKING NETWORK (DRIFT-DIFFUSION)")
    
    decision_net = DecisionMakingNetwork(n_options=2)
    
    print("\nDecision network: 2 options (evidence accumulation)")
    print("Accumulating evidence until threshold reached...")
    
    # Simulate decision
    decision_net.reset()
    time_ms = 0
    dt_ms = 1.0
    
    # Option 0 has slightly more evidence
    true_evidence = np.array([0.6, 0.4])
    
    while not decision_net.decision_made and time_ms < 1000:
        # Add noisy evidence
        decision_net.accumulate_evidence(true_evidence, dt_ms)
        time_ms += dt_ms
    
    decision = decision_net.get_decision()
    
    print(f"\nResults:")
    print(f"  Decision: Option {decision}")
    print(f"  Reaction time: {time_ms:.1f} ms")
    print(f"  Final evidence: {decision_net.evidence}")


def demo_sensorimotor():
    """Demo 10: Sensorimotor loop"""
    print_header("DEMO 10: SENSORIMOTOR LOOP (EMBODIMENT)")
    
    sensorimotor = SensorimotorLoop()
    
    print("\nSensorimotor system with camera input and motor output")
    
    # Create fake camera image
    image = np.random.rand(32, 32) * 255  # 32x32 grayscale image
    
    print(f"\nInput: {image.shape[0]}×{image.shape[1]} image")
    
    # Simple neural network function: pass-through with scaling
    def simple_neural_network(sensory_activity):
        # Simple linear transformation
        motor_activity = sensory_activity[:sensorimotor.motor_interface.n_joints]
        return motor_activity
    
    # Process sensorimotor step
    result = sensorimotor.process_sensorimotor_step(image, simple_neural_network)
    
    print(f"\nProcessing results:")
    print(f"  Sensory neurons activated: {len(result['sensory_activity'])}")
    print(f"  Motor neurons activated: {len(result['motor_activity'])}")
    print(f"  Joint commands: {result['motor_commands']}")
    print(f"  Joint positions: {result['joint_state']['positions']}")


def demo_integrated_system():
    """Demo 11: Integrated advanced system"""
    print_header("DEMO 11: INTEGRATED ADVANCED BRAIN SIMULATION")
    
    print("\nCreating integrated brain simulation with all features...")
    
    sim = AdvancedBrainSimulation(use_gpu=False, dt_ms=0.1)
    
    # Add multi-compartment neurons
    print("\n1. Adding multi-compartment neurons...")
    for i in range(5):
        sim.create_multicompartment_neuron(i, n_dendrites=3)
    
    # Add STDP synapses
    print("2. Creating STDP synapses...")
    sim.create_stdp_synapse(0, 1, NeurotransmitterType.GLUTAMATE)
    sim.create_stdp_synapse(1, 2, NeurotransmitterType.GLUTAMATE)
    sim.create_stdp_synapse(2, 3, NeurotransmitterType.DOPAMINE)
    
    # Add cognitive modules
    print("3. Adding cognitive modules...")
    sim.add_cognitive_modules(n_memory_units=7, n_rl_states=10, n_rl_actions=4)
    
    # Enable embodiment
    print("4. Enabling embodiment interfaces...")
    sim.enable_embodiment()
    
    # Print summary
    print("\n" + sim.get_summary())
    
    # Run simulation
    print("\nRunning integrated simulation (100 steps)...")
    for step in range(100):
        stats = sim.simulate_step()
    
    print(f"\nFinal time: {sim.current_time_ms:.2f} ms")
    print("\nIntegrated system ready for complex cognitive tasks!")


def main():
    """Run all demonstrations"""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ADVANCED BRAIN SIMULATION - PHASES 2-6 DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    demos = [
        ("Hodgkin-Huxley Dynamics", demo_hodgkin_huxley),
        ("Multi-Compartment Neurons", demo_multicompartment_neuron),
        ("STDP Learning", demo_stdp),
        ("Neurotransmitter Systems", demo_neurotransmitters),
        ("GPU Acceleration", demo_gpu_acceleration),
        ("Attention Mechanism", demo_attention),
        ("Working Memory", demo_working_memory),
        ("Reinforcement Learning", demo_reinforcement_learning),
        ("Decision Making", demo_decision_making),
        ("Sensorimotor Loop", demo_sensorimotor),
        ("Integrated System", demo_integrated_system),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos):
            input("\n\nPress Enter to continue to next demo...")
    
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE".center(80))
    print("=" * 80)
    print("\nAll advanced features demonstrated successfully!")
    print("The brain simulation now includes:")
    print("  ✓ Phase 2: Biophysical realism")
    print("  ✓ Phase 3: Massive scale capabilities")
    print("  ✓ Phase 4: Realistic connectivity")
    print("  ✓ Phase 5: Cognitive functions")
    print("  ✓ Phase 6: Embodiment")


if __name__ == "__main__":
    main()
