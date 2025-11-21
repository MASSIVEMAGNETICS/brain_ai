# Quick Start Guide - Advanced Brain Simulation

## Installation

```bash
# Clone repository
git clone https://github.com/MASSIVEMAGNETICS/brain_ai.git
cd brain_ai

# Install base requirements
pip install -r requirements.txt

# Optional: Install GPU support (choose one)
pip install torch          # For PyTorch GPU acceleration
# OR
pip install tensorflow     # For TensorFlow GPU acceleration
```

## Quick Examples

### 1. Multi-Compartment Neuron with Hodgkin-Huxley Dynamics

```python
from brain_simulation_advanced import MultiCompartmentNeuron

# Create neuron with realistic compartments
neuron = MultiCompartmentNeuron(
    neuron_id=0,
    n_dendrites=5,      # 5 dendritic branches
    n_axon_segments=3   # 3 axon segments
)

# Simulate with synaptic input to dendrite
synaptic_inputs = {neuron.dendrite_ids[0]: 50.0}  # Current injection
for step in range(1000):
    spiked = neuron.update(dt_ms=0.1, synaptic_inputs=synaptic_inputs)
    if spiked:
        print(f"Spike at {step * 0.1} ms!")
```

### 2. STDP Learning

```python
from brain_simulation_advanced import STDPSynapse, NeurotransmitterType

# Create synapse with dopamine neurotransmitter
synapse = STDPSynapse(
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    neurotransmitter=NeurotransmitterType.DOPAMINE
)

# Simulate spike-timing-dependent learning
current_time = 0.0
for trial in range(10):
    # Pre spikes before Post → LTP (strengthening)
    synapse.update_stdp(current_time, pre_spiked=True, post_spiked=False)
    synapse.update_stdp(current_time + 5.0, pre_spiked=False, post_spiked=True)
    current_time += 50.0

print(f"Final weight: {synapse.weight:.4f}")
```

### 3. GPU-Accelerated Large Network

```python
from brain_simulation_advanced import GPUAcceleratedNetwork

# Create network with 100,000 neurons
gpu_net = GPUAcceleratedNetwork(
    n_neurons=100000,
    use_pytorch=True,    # Use PyTorch backend
    device='auto'        # Auto-detect GPU
)

# Create sparse random connectivity (1% connection probability)
gpu_net.create_sparse_connectivity(connection_probability=0.01)

# Run simulation
for step in range(1000):
    spike_mask = gpu_net.simulate_step_gpu(dt_ms=0.1)
    n_spikes = gpu_net.get_spike_count(spike_mask)
    if step % 100 == 0:
        print(f"Step {step}: {n_spikes} spikes")
```

### 4. Attention Mechanism

```python
from brain_simulation_advanced import AttentionMechanism
import numpy as np

# Create attention mechanism for 1000 neurons
attention = AttentionMechanism(n_neurons=1000)

# Simulate baseline neural activity
activity = np.random.rand(1000) * 0.5

# Apply spatial attention to neurons 100-150
attended_neurons = list(range(100, 151))
attention.set_spatial_attention(attended_neurons, gain=3.0)

# Modulate activity
modulated_activity = attention.apply_attention(activity)

# Compare attended vs unattended regions
print(f"Attended region: {modulated_activity[100:151].mean():.3f}")
print(f"Unattended region: {modulated_activity[:100].mean():.3f}")
```

### 5. Working Memory

```python
from brain_simulation_advanced import WorkingMemoryCircuit

# Create working memory with 7 slots (Miller's Law)
wm = WorkingMemoryCircuit(n_memory_units=7)

# Encode items
wm.encode(0, strength=1.0)  # Item 0
wm.encode(2, strength=0.8)  # Item 2
wm.encode(5, strength=0.9)  # Item 5

# Maintain through recurrent activity
for _ in range(100):
    wm.maintain()

# Retrieve items
for i in range(7):
    strength = wm.retrieve(i)
    if strength > 0.01:
        print(f"Item {i}: strength = {strength:.3f}")
```

### 6. Reinforcement Learning

```python
from brain_simulation_advanced import ReinforcementLearningModule

# Create RL module
rl = ReinforcementLearningModule(
    n_states=10,
    n_actions=4
)

# Training loop
for episode in range(100):
    state = 0
    for step in range(20):
        # Select action
        action = rl.select_action(state, epsilon=0.1)
        
        # Environment step (example)
        next_state = (state + 1) % 10
        reward = 1.0 if next_state == 9 else 0.0
        
        # Learn from experience
        td_error = rl.compute_td_error(state, action, reward, next_state)
        rl.update_q_value(state, action, td_error)
        
        state = next_state

# Check learned policy
print(f"Q-values for state 0: {rl.q_values[0, :]}")
print(f"Best action: {rl.q_values[0, :].argmax()}")
```

### 7. Decision Making

```python
from brain_simulation_advanced import DecisionMakingNetwork
import numpy as np

# Create decision network (2 options)
decision_net = DecisionMakingNetwork(n_options=2)

# Reset for new decision
decision_net.reset()

# Accumulate evidence
evidence = np.array([0.6, 0.4])  # Option 0 has more evidence
time_ms = 0.0

while not decision_net.decision_made and time_ms < 1000:
    decision_net.accumulate_evidence(evidence, dt_ms=1.0)
    time_ms += 1.0

# Get decision
choice = decision_net.get_decision()
print(f"Decision: Option {choice}")
print(f"Reaction time: {time_ms:.1f} ms")
```

### 8. Sensorimotor Loop

```python
from brain_simulation_advanced import SensorimotorLoop
import numpy as np

# Create sensorimotor system
loop = SensorimotorLoop()

# Simulate camera input
camera_image = np.random.rand(64, 64)  # 64x64 grayscale image

# Define simple neural processing
def neural_network(sensory_activity):
    # Simple pass-through for demo
    n_motor = loop.motor_interface.n_joints
    return sensory_activity[:n_motor] * 2.0

# Process one sensorimotor step
result = loop.process_sensorimotor_step(camera_image, neural_network)

print(f"Motor commands: {result['motor_commands']}")
print(f"Joint positions: {result['joint_state']['positions']}")
```

### 9. Complete Integrated System

```python
from brain_simulation_advanced import (
    AdvancedBrainSimulation,
    NeurotransmitterType
)

# Create advanced simulation
sim = AdvancedBrainSimulation(use_gpu=False, dt_ms=0.1)

# Add multi-compartment neurons
for i in range(10):
    sim.create_multicompartment_neuron(i, n_dendrites=5)

# Create network with different neurotransmitters
sim.create_stdp_synapse(0, 1, NeurotransmitterType.GLUTAMATE)
sim.create_stdp_synapse(1, 2, NeurotransmitterType.GLUTAMATE)
sim.create_stdp_synapse(2, 3, NeurotransmitterType.DOPAMINE)
sim.create_stdp_synapse(3, 4, NeurotransmitterType.GABA)

# Add cognitive modules
sim.add_cognitive_modules(
    n_memory_units=7,
    n_rl_states=10,
    n_rl_actions=4
)

# Enable embodiment
sim.enable_embodiment()

# Optional: Enable GPU for larger networks
# sim.enable_gpu_acceleration(n_neurons=100000)

# Run simulation
print("Running integrated simulation...")
for step in range(1000):
    stats = sim.simulate_step()
    if step % 100 == 0:
        print(f"  Step {step}: {stats['spikes']} spikes")

# Get summary
print("\n" + sim.get_summary())
```

## Run the Demos

```bash
# Run all 11 comprehensive demos
python advanced_demo.py

# Or run specific demos programmatically
python -c "from advanced_demo import demo_stdp; demo_stdp()"
```

## Available Demos

1. **Hodgkin-Huxley Dynamics** - Realistic action potentials
2. **Multi-Compartment Neurons** - Spatial voltage propagation
3. **STDP Learning** - Spike-timing-dependent plasticity
4. **Neurotransmitter Systems** - 6 different neurotransmitters
5. **GPU Acceleration** - Large-scale parallel simulation
6. **Attention Mechanism** - Top-down modulation
7. **Working Memory** - Persistent activity
8. **Reinforcement Learning** - Dopamine-based learning
9. **Decision Making** - Evidence accumulation
10. **Sensorimotor Loop** - Embodied cognition
11. **Integrated System** - All features together

## Documentation

- **README.md** - Project overview and quick start
- **ADVANCED_DOCUMENTATION.md** - Complete API reference
- **IMPLEMENTATION_PHASES_2-6.md** - Implementation details
- **SIMULATION_DOCUMENTATION.md** - Basic simulation guide

## Common Use Cases

### Research & Education
```python
# Study STDP learning rules
from advanced_demo import demo_stdp
demo_stdp()
```

### Computational Neuroscience
```python
# Model realistic neurons
from advanced_demo import demo_multicompartment_neuron
demo_multicompartment_neuron()
```

### AI/ML Applications
```python
# Spiking neural networks with learning
from advanced_demo import demo_reinforcement_learning
demo_reinforcement_learning()
```

### Robotics & Embodiment
```python
# Sensorimotor control
from advanced_demo import demo_sensorimotor
demo_sensorimotor()
```

## Performance Tips

1. **For small networks (<1000 neurons)**: Use CPU, multi-compartment OK
2. **For medium networks (1K-100K)**: Enable GPU acceleration
3. **For large networks (>100K)**: GPU required, use sparse connectivity
4. **For learning**: Use STDP for temporal precision, Hebbian for simplicity
5. **For embodiment**: Process sensory input in batches

## Troubleshooting

**GPU not detected?**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**Out of memory?**
- Reduce number of neurons
- Decrease connection probability
- Use sparse connectivity
- Simplify neuron model

**Slow performance?**
- Enable GPU acceleration
- Reduce compartments per neuron
- Use vectorized operations
- Batch process inputs

## Next Steps

1. Read the [ADVANCED_DOCUMENTATION.md](ADVANCED_DOCUMENTATION.md) for complete API details
2. Explore the demos in `advanced_demo.py`
3. Build your own cognitive architecture
4. Experiment with different neurotransmitters
5. Try GPU acceleration on large networks

## Support

- **Documentation**: See `ADVANCED_DOCUMENTATION.md`
- **Examples**: Check `advanced_demo.py`
- **Issues**: Open a GitHub issue
- **Questions**: Refer to inline code documentation

---

Happy simulating! 🧠✨
