# Advanced Brain Simulation Documentation - Phases 2-6

## Overview

This document describes the advanced features implemented in `brain_simulation_advanced.py`, covering Phases 2-6 of the brain simulation roadmap. These features build upon the basic simulation to provide biophysical realism, massive scale capabilities, realistic connectivity, cognitive functions, and embodiment.

---

## Phase 2: Biophysical Realism

### Multi-Compartment Neuron Models

Multi-compartment neurons provide spatial resolution within individual neurons, modeling dendrites, soma, and axon as separate electrical compartments.

#### MultiCompartmentNeuron Class

```python
from brain_simulation_advanced import MultiCompartmentNeuron

neuron = MultiCompartmentNeuron(
    neuron_id=0,
    n_dendrites=5,      # Number of dendritic compartments
    n_axon_segments=3   # Number of axon segments
)
```

**Compartment Types:**
- **Dendrites**: Receive synaptic inputs, longer and thinner
- **Soma**: Cell body, integrates dendritic signals
- **Axon Initial Segment (AIS)**: Spike initiation zone with high Na+ channel density
- **Axon segments**: Propagate spikes to other neurons

**Key Features:**
- Realistic compartment geometry (length and diameter in micrometers)
- Electrical coupling between neighboring compartments
- Voltage propagation through the neuron
- Spike initiation at the AIS (most excitable region)

### Hodgkin-Huxley Dynamics

The Hodgkin-Huxley model provides detailed ion channel kinetics for realistic action potentials.

#### HodgkinHuxleyChannels Class

```python
from brain_simulation_advanced import HodgkinHuxleyChannels

hh = HodgkinHuxleyChannels()

# Update gating variables
hh.update_gates(voltage=-70.0, dt_ms=0.1)

# Compute ionic currents
i_na, i_k, i_leak = hh.compute_currents(voltage=-70.0)
```

**Gating Variables:**
- **m**: Na+ channel activation (fast)
- **h**: Na+ channel inactivation (medium)
- **n**: K+ channel activation (slow)

**Ion Channels:**
- **Sodium (Na+)**: Rapid depolarization during action potential
- **Potassium (K+)**: Repolarization and afterhyperpolarization
- **Leak**: Maintains resting potential

**Equations:**
```
I_Na = g_Na_max * m³ * h * (V - E_Na)
I_K = g_K_max * n⁴ * (V - E_K)
I_leak = g_leak * (V - E_leak)
```

### Multiple Neurotransmitter Systems

Synapses can use different neurotransmitters, each with distinct effects.

#### NeurotransmitterType Enum

```python
from brain_simulation_advanced import NeurotransmitterType, STDPSynapse

synapse = STDPSynapse(
    pre_neuron_id=0,
    post_neuron_id=1,
    neurotransmitter=NeurotransmitterType.DOPAMINE
)
```

**Available Neurotransmitters:**

| Neurotransmitter | Effect | Function |
|------------------|--------|----------|
| Glutamate | +1.0 (excitatory) | Primary excitatory neurotransmitter |
| GABA | -0.8 (inhibitory) | Primary inhibitory neurotransmitter |
| Dopamine | +1.2 (modulatory) | Reward, motivation, learning |
| Serotonin | +0.9 (modulatory) | Mood, sleep regulation |
| Acetylcholine | +1.1 (modulatory) | Learning, memory, attention |
| Norepinephrine | +1.15 (modulatory) | Arousal, attention, stress |

### Spike-Timing-Dependent Plasticity (STDP)

STDP is a learning rule where synaptic strength depends on the precise timing of pre- and post-synaptic spikes.

#### STDPSynapse Class

```python
from brain_simulation_advanced import STDPSynapse

synapse = STDPSynapse(
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    delay_ms=1.0
)

# Update during simulation
synapse.update_stdp(
    current_time_ms=10.0,
    pre_spiked=True,
    post_spiked=False
)
```

**Learning Rules:**
- **Pre → Post (causation)**: Strengthening (LTP - Long-Term Potentiation)
  - Δw = A+ * exp(-Δt/τ+)
- **Post → Pre**: Weakening (LTD - Long-Term Depression)
  - Δw = -A- * exp(Δt/τ-)

**Parameters:**
- `a_plus`: LTP amplitude (default: 0.005)
- `a_minus`: LTD amplitude (default: 0.00525)
- `tau_plus`: LTP time constant (default: 20 ms)
- `tau_minus`: LTD time constant (default: 20 ms)

### Homeostatic Plasticity

Homeostatic plasticity maintains network stability by adjusting synaptic strengths to keep firing rates within target ranges.

#### HomeostaticPlasticity Class

```python
from brain_simulation_advanced import HomeostaticPlasticity

homeostasis = HomeostaticPlasticity(
    target_firing_rate=5.0,  # Target: 5 Hz
    scaling_rate=0.001
)

# Apply synaptic scaling
updated_weights = homeostasis.update(
    neuron_spiked=True,
    synaptic_weights=[1.0, 1.5, 0.8],
    dt_ms=0.1
)
```

**Mechanisms:**
- **Synaptic scaling**: Global adjustment of all synaptic strengths
- **Target firing rate**: Desired average activity level
- **Feedback regulation**: Increases weights if too quiet, decreases if too active

---

## Phase 3: Massive Scale

### GPU Acceleration

GPU acceleration enables simulation of millions of neurons in parallel using PyTorch or TensorFlow.

#### GPUAcceleratedNetwork Class

```python
from brain_simulation_advanced import GPUAcceleratedNetwork

# Create GPU network
gpu_net = GPUAcceleratedNetwork(
    n_neurons=1000000,  # 1 million neurons
    use_pytorch=True,    # Use PyTorch (or TensorFlow if False)
    device='auto'        # Auto-detect GPU/CPU
)

# Create sparse connectivity
gpu_net.create_sparse_connectivity(connection_probability=0.01)

# Simulate
spike_mask = gpu_net.simulate_step_gpu(dt_ms=0.1)
n_spikes = gpu_net.get_spike_count(spike_mask)
```

**Features:**
- **Parallel processing**: All neurons updated simultaneously on GPU
- **Sparse connectivity**: Memory-efficient sparse tensor representation
- **Device flexibility**: Automatic GPU detection with CPU fallback
- **Multiple backends**: PyTorch or TensorFlow support

**Performance:**
- CPU: ~1,000 neurons at real-time
- GPU: 100,000+ neurons at real-time
- Large-scale: Millions of neurons (depends on GPU memory)

**Requirements:**
```bash
# PyTorch (recommended for GPU)
pip install torch

# OR TensorFlow
pip install tensorflow
```

### Sparse Connectivity Matrices

Realistic brain connectivity is sparse (~1% connection probability), requiring efficient data structures.

**Implementation:**
- **COO format**: Coordinate list (indices + weights)
- **Memory efficiency**: Only store non-zero connections
- **Fast operations**: Optimized sparse matrix multiplication

---

## Phase 4: Realistic Connectivity

### Brain Regions

Define anatomically realistic brain regions with specific properties.

#### BrainRegion Class

```python
from brain_simulation_advanced import BrainRegion

v1 = BrainRegion(
    name="V1",
    neuron_count=200_000_000,  # ~200M neurons
    region_type="cortical",
    excitatory_ratio=0.8,      # 80% excitatory
    inhibitory_ratio=0.2,      # 20% inhibitory
    position=(0, -50, 0)       # Position in mm
)
```

**Properties:**
- Neuron count and type distribution
- Spatial location for distance-dependent connectivity
- Connection probabilities
- Synaptic density (~10,000 synapses per neuron)

### Connectome Builder

Build networks based on connectome data and anatomical principles.

#### ConnectomeBuilder Class

```python
from brain_simulation_advanced import ConnectomeBuilder, BrainRegion

builder = ConnectomeBuilder()

# Add regions
builder.add_region(BrainRegion(name="V1", neuron_count=1000, ...))
builder.add_region(BrainRegion(name="PFC", neuron_count=2000, ...))

# Add pathways
builder.add_pathway(
    source_region="V1",
    target_region="PFC",
    connection_prob=0.01,
    delay_ms=10.0
)
```

### White Matter Tract Delays

Realistic transmission delays based on axonal conduction velocity and distance.

```python
delay_ms = builder.calculate_white_matter_delay(
    source_pos=(0, 0, 0),
    target_pos=(50, 0, 0),  # 50mm away
    conduction_velocity_m_s=4.0  # Myelinated axon
)
```

**Typical Values:**
- **Myelinated axons**: 4-10 m/s
- **Unmyelinated axons**: 0.5-2 m/s
- **Synaptic delay**: 0.5-1.0 ms
- **Long-range connections**: 10-50 ms total delay

---

## Phase 5: Cognitive Functions

### Attention Mechanism

Top-down attention modulates sensory processing through gain control.

#### AttentionMechanism Class

```python
from brain_simulation_advanced import AttentionMechanism

attention = AttentionMechanism(n_neurons=1000)

# Focus attention on specific neurons
attended_neurons = list(range(100, 200))
attention.set_spatial_attention(attended_neurons, gain=2.0)

# Apply to neural activity
modulated_activity = attention.apply_attention(neural_activity)
```

**Types:**
- **Spatial attention**: Enhance specific locations
- **Feature-based attention**: Enhance specific features
- **Gain modulation**: Multiplicative scaling of responses

### Working Memory Circuit

Persistent neural activity maintains information during delay periods.

#### WorkingMemoryCircuit Class

```python
from brain_simulation_advanced import WorkingMemoryCircuit

wm = WorkingMemoryCircuit(n_memory_units=7)  # 7±2 capacity

# Encode items
wm.encode(item_index=0, strength=1.0)
wm.encode(item_index=2, strength=0.8)

# Maintain through recurrent activity
for _ in range(100):
    wm.maintain()

# Retrieve
strength = wm.retrieve(item_index=0)
```

**Features:**
- **Capacity limits**: 7±2 items (Miller's Law)
- **Recurrent excitation**: Self-sustaining activity
- **Gradual decay**: Without rehearsal, memory fades
- **Item strength**: Variable memory precision

### Reinforcement Learning Module

Dopamine-based reward learning using temporal difference (TD) algorithms.

#### ReinforcementLearningModule Class

```python
from brain_simulation_advanced import ReinforcementLearningModule

rl = ReinforcementLearningModule(
    n_states=10,
    n_actions=4
)

# Learning loop
state = 0
action = rl.select_action(state, epsilon=0.1)

# Get reward from environment
reward = environment.step(action)
next_state = environment.get_state()

# Compute TD error (dopamine signal)
td_error = rl.compute_td_error(state, action, reward, next_state)

# Update Q-values
rl.update_q_value(state, action, td_error)

# Get dopamine signal
dopamine = rl.get_dopamine_signal()
```

**Components:**
- **Q-learning**: State-action value function
- **TD error**: Reward prediction error (dopamine)
- **Epsilon-greedy**: Exploration vs. exploitation
- **Value function**: Learned action values

### Decision-Making Network

Evidence accumulation for perceptual and cognitive decisions.

#### DecisionMakingNetwork Class

```python
from brain_simulation_advanced import DecisionMakingNetwork

decision_net = DecisionMakingNetwork(n_options=2)

# Accumulate evidence
while not decision_net.decision_made:
    evidence = np.array([0.6, 0.4])  # Evidence for each option
    decision_net.accumulate_evidence(evidence, dt_ms=1.0)

# Get decision
choice = decision_net.get_decision()
```

**Model:**
- **Drift-diffusion**: Evidence accumulates to threshold
- **Noisy accumulation**: Realistic variability
- **Reaction time**: Time to reach threshold
- **Confidence**: Evidence strength at decision time

---

## Phase 6: Embodiment

### Sensory Input Interface

Process real-world sensory data into neural activity patterns.

#### SensoryInputInterface Class

```python
from brain_simulation_advanced import SensoryInputInterface

sensory = SensoryInputInterface(input_type="camera")

# Process camera image
import numpy as np
image = np.random.rand(480, 640, 3)  # RGB image
spike_rates = sensory.process_camera_input(image)

# Process audio
audio_samples = np.random.randn(44100)  # 1 second at 44.1kHz
cochlear_activity = sensory.process_audio_input(audio_samples)
```

**Modalities:**
- **Vision**: Image → spike rates (retinal ganglion cells)
- **Audition**: Sound → frequency bands (cochlear channels)
- **Custom**: Extensible for other modalities

### Motor Output Interface

Decode motor neuron activity into robot control commands.

#### MotorOutputInterface Class

```python
from brain_simulation_advanced import MotorOutputInterface

motor = MotorOutputInterface(n_joints=6)

# Decode motor commands
motor_neuron_activity = np.random.rand(6)
joint_velocities = motor.decode_motor_commands(motor_neuron_activity)

# Update joint state
motor.update_joints(joint_velocities, dt_s=0.001)

# Get state
joint_state = motor.get_joint_state()
print(joint_state['positions'])
print(joint_state['velocities'])
```

**Features:**
- **Population vector decoding**: Neural activity → joint commands
- **Joint control**: Position and velocity
- **Scalable**: Arbitrary number of joints/actuators

### Sensorimotor Loop

Closed-loop integration of perception and action.

#### SensorimotorLoop Class

```python
from brain_simulation_advanced import SensorimotorLoop

loop = SensorimotorLoop()

# Define neural processing function
def neural_network(sensory_activity):
    # Your neural network here
    motor_activity = process(sensory_activity)
    return motor_activity

# Process sensorimotor step
sensory_input = camera.capture()
result = loop.process_sensorimotor_step(sensory_input, neural_network)

print(result['motor_commands'])
print(result['joint_state'])
```

**Components:**
- **Sensory processing**: Input → neural activity
- **Neural computation**: User-defined network
- **Motor decoding**: Neural activity → commands
- **Feedback**: Current state influences next input

---

## Integrated System

### AdvancedBrainSimulation Class

The main class integrating all advanced features.

```python
from brain_simulation_advanced import (
    AdvancedBrainSimulation,
    NeurotransmitterType
)

# Create simulation
sim = AdvancedBrainSimulation(use_gpu=False, dt_ms=0.1)

# Add multi-compartment neurons
for i in range(10):
    sim.create_multicompartment_neuron(i, n_dendrites=5)

# Add STDP synapses
sim.create_stdp_synapse(0, 1, NeurotransmitterType.GLUTAMATE)
sim.create_stdp_synapse(1, 2, NeurotransmitterType.DOPAMINE)

# Add cognitive modules
sim.add_cognitive_modules(
    n_memory_units=7,
    n_rl_states=10,
    n_rl_actions=4
)

# Enable embodiment
sim.enable_embodiment()

# Enable GPU (optional)
# sim.enable_gpu_acceleration(n_neurons=100000)

# Run simulation
for step in range(1000):
    stats = sim.simulate_step()
    print(f"Time: {stats['time_ms']:.1f} ms, Spikes: {stats['spikes']}")

# Get summary
print(sim.get_summary())
```

---

## Examples and Demos

### Running the Demos

```bash
# Run all demos interactively
python advanced_demo.py

# Run specific demo
python -c "from advanced_demo import demo_stdp; demo_stdp()"
```

### Available Demos

1. **Hodgkin-Huxley Dynamics**: Realistic action potentials
2. **Multi-Compartment Neurons**: Spatial voltage propagation
3. **STDP Learning**: Spike-timing-dependent plasticity
4. **Neurotransmitter Systems**: Different transmitter effects
5. **GPU Acceleration**: Large-scale parallel simulation
6. **Attention Mechanism**: Top-down modulation
7. **Working Memory**: Persistent activity maintenance
8. **Reinforcement Learning**: Dopamine-based learning
9. **Decision Making**: Evidence accumulation
10. **Sensorimotor Loop**: Embodied cognition
11. **Integrated System**: All features together

---

## Performance Considerations

### Computational Cost

| Feature | Neurons | Synapses | Speed (relative to basic) |
|---------|---------|----------|---------------------------|
| Basic integrate-and-fire | 1000 | 10000 | 1x |
| Multi-compartment | 100 | 1000 | 10x slower |
| Hodgkin-Huxley | 1000 | 10000 | 5x slower |
| GPU acceleration | 1M | 10M | 100x faster |

### Memory Requirements

- **Basic neuron**: ~100 bytes
- **Multi-compartment neuron**: ~1 KB (10 compartments)
- **GPU network**: ~4 bytes per neuron (float32)
- **Sparse connectivity**: ~12 bytes per synapse

### Scaling Recommendations

- **Small networks (<1000 neurons)**: CPU, multi-compartment OK
- **Medium networks (1K-100K)**: GPU recommended
- **Large networks (>100K)**: GPU required, sparse connectivity essential

---

## Future Enhancements

### Planned Features

1. **Distributed computing**: Multi-GPU and multi-node support
2. **HCP data integration**: Load real human connectome data
3. **Online learning**: Continuous adaptation during operation
4. **Neuromodulation**: More detailed dopamine/serotonin effects
5. **Plasticity rules**: BCM, metaplasticity, structural plasticity
6. **Ion concentration dynamics**: Calcium-dependent plasticity
7. **Astrocyte models**: Glial modulation of synapses
8. **Energy models**: ATP consumption and metabolic constraints

---

## References

### Phase 2: Biophysical Realism
- Hodgkin & Huxley (1952): "A quantitative description of membrane current"
- Bi & Poo (1998): "Synaptic modifications in cultured hippocampal neurons"
- Turrigiano & Nelson (2004): "Homeostatic plasticity in the developing nervous system"

### Phase 3: Massive Scale
- Izhikevich & Edelman (2008): "Large-scale model of mammalian thalamocortical systems"
- Markram et al. (2015): "Reconstruction and simulation of neocortical microcircuitry"

### Phase 4: Realistic Connectivity
- Van Essen et al. (2013): "The WU-Minn Human Connectome Project"
- Oh et al. (2014): "A mesoscale connectome of the mouse brain"

### Phase 5: Cognitive Functions
- Desimone & Duncan (1995): "Neural mechanisms of selective visual attention"
- Baddeley (2000): "The episodic buffer: a new component of working memory?"
- Schultz (1998): "Predictive reward signal of dopamine neurons"

### Phase 6: Embodiment
- Wolpert & Kawato (1998): "Multiple paired forward and inverse models"
- Rizzolatti & Craighero (2004): "The mirror-neuron system"

---

## Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should be True

# Or for TensorFlow
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Out of Memory

- Reduce `n_neurons`
- Decrease `connection_probability`
- Use sparse connectivity
- Enable gradient checkpointing (for learning)

### Slow Performance

- Enable GPU acceleration
- Reduce compartments per neuron
- Use simpler neuron models for large networks
- Batch process sensory inputs

---

## License

Educational and research use. See main repository LICENSE.

---

## Contact and Contributions

For questions, bug reports, or contributions, please open an issue on the GitHub repository.
