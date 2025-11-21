# Brain Simulation - Dynamic Neural Network

## Overview

The `brain_simulation.py` module transforms the brain_ai repository from a static knowledge atlas into a **living, dynamic simulation** with actual spiking neurons, synaptic connections, and temporal evolution.

## Key Differences: Static vs. Dynamic

### Before (Static Atlas)
```python
# Returns a string description
action_potential = ActionPotential.get_depolarization()
# Output: "Depolarization Phase: Threshold: -55 mV..."
```

### After (Dynamic Simulation)
```python
# Runs actual voltage simulation
neuron = SimulatedNeuron(0)
neuron.add_synaptic_input(10.0)
spiked = neuron.update(dt_ms=0.1)
# neuron.voltage_mv changes in real-time: -70 → -55 → +30 mV
```

## Core Components

### 1. SimulatedNeuron

A stateful neuron that simulates membrane potential dynamics.

**Key Features:**
- **Dynamic voltage**: Changes over time based on currents
- **Ion concentrations**: Na+, K+, Ca2+, Cl- with Nernst potentials
- **Spike mechanics**: Threshold crossing, refractory period
- **Realistic dynamics**: Based on Hodgkin-Huxley principles

**State Variables:**
```python
neuron.voltage_mv           # Current membrane potential
neuron.ion_concentrations   # Ion concentrations (inside/outside)
neuron.threshold_mv         # Spike threshold (-55 mV)
neuron.refractory_period_ms # Absolute refractory period (2 ms)
neuron.spike_times          # History of spike times
```

**Usage:**
```python
from brain_simulation import SimulatedNeuron, NeuronType

# Create a neuron
neuron = SimulatedNeuron(
    neuron_id=0,
    neuron_type=NeuronType.EXCITATORY,
    region="Hippocampus"
)

# Inject current and update
neuron.add_synaptic_input(15.0)
spiked = neuron.update(dt_ms=0.1)

if spiked:
    print(f"Neuron spiked at {neuron.voltage_mv} mV!")
```

### 2. Synapse

Connects two neurons with adjustable synaptic weight.

**Key Features:**
- **Spike transmission**: Pre-synaptic spike → post-synaptic current
- **Transmission delay**: Realistic axonal conduction time
- **Hebbian learning**: Weight changes based on co-activity
- **Synaptic plasticity**: LTP and LTD

**Parameters:**
```python
synapse.weight              # Synaptic strength (0.0 - 10.0)
synapse.delay_ms            # Transmission delay (typically 1-5 ms)
synapse.learning_rate       # Plasticity rate (default 0.01)
```

**Usage:**
```python
from brain_simulation import Synapse

# Connect two neurons
pre = SimulatedNeuron(0, NeuronType.EXCITATORY)
post = SimulatedNeuron(1, NeuronType.EXCITATORY)

synapse = Synapse(pre, post, initial_weight=2.0, delay_ms=1.0)

# Transmit spike
synapse.transmit(current_time_ms=10.0, pre_spiked=True)

# Apply learning
synapse.apply_hebbian_learning(pre_spiked=True, post_spiked=True)
# → weight increases (LTP)
```

### 3. BrainSimulation

The main simulation engine - provides the "heartbeat" of the brain.

**Key Features:**
- **Temporal evolution**: Advances simulation time step by step
- **Network management**: Handles collections of neurons and synapses
- **Sensory input**: Inject external currents
- **Learning**: Optional Hebbian plasticity
- **Statistics**: Track spikes, firing rates, etc.

**Core Methods:**

#### Creating Networks
```python
from brain_simulation import BrainSimulation

sim = BrainSimulation(dt_ms=0.1)  # 0.1 ms time step

# Add individual neurons
neuron_id = sim.add_neuron(0, NeuronType.EXCITATORY, region="V1")

# Connect neurons
sim.connect_neurons(pre_id=0, post_id=1, weight=2.0, delay_ms=1.0)

# Or create a simple 3-layer network
sim.create_simple_network(
    n_sensory=10,      # Retina
    n_processing=20,   # V1
    n_output=5         # Visual Association
)
```

#### Running Simulations
```python
# Single time step
spike_counts = sim.simulate_step(enable_learning=True)

# Run for duration
sim.run(duration_ms=100.0, enable_learning=True, verbose=True)
```

#### Sensory Input
```python
# Inject current into specific neuron
sim.inject_current(neuron_id=0, current=15.0)

# Stimulate entire region
sim.stimulate_region("Retina", current=25.0)
```

#### Analysis
```python
# Get statistics
stats = sim.get_statistics()
print(f"Total spikes: {stats['total_spikes']}")
print(f"Average firing rate: {stats['average_firing_rate_hz']} Hz")

# Get voltage trace
voltage_trace = sim.get_neuron_voltage_trace(neuron_id=0)

# Get spike times
spike_times = sim.get_spike_times(neuron_id=0)
```

## Biophysical Realism

### Nernst Equilibrium Potentials

The simulation uses real ion concentrations and the Nernst equation:

```
E_ion = (RT/zF) * ln([ion]_out / [ion]_in)
```

**Typical values:**
- E_Na ≈ +66 mV (drives depolarization)
- E_K ≈ -98 mV (drives repolarization)
- E_Ca ≈ +132 mV (important for plasticity)
- E_Cl ≈ -89 mV (inhibition)

```python
from brain_simulation import IonConcentrations

ions = IonConcentrations()
e_na = ions.calculate_nernst_potential('na')
e_k = ions.calculate_nernst_potential('k')

print(f"Sodium equilibrium: {e_na:.2f} mV")
print(f"Potassium equilibrium: {e_k:.2f} mV")
```

### Action Potential Mechanism

The neuron model captures the essential phases:

1. **Resting State**: V_m = -70 mV
2. **Depolarization**: Voltage rises above threshold (-55 mV)
3. **Spike**: Rapid rise to +30 mV
4. **Repolarization**: Return to resting
5. **Refractory Period**: Temporary inability to spike (2 ms)

### Hebbian Learning

**"Cells that fire together, wire together"**

```python
if pre_spiked and post_spiked:
    # Long-Term Potentiation (LTP)
    weight += learning_rate * (weight_max - weight)
elif pre_spiked and not post_spiked:
    # Long-Term Depression (LTD)
    weight -= learning_rate * 0.005 * weight
```

## Example: Visual Pathway Simulation

```python
import numpy as np
from brain_simulation import BrainSimulation

# Set random seed for reproducibility
np.random.seed(42)

# Create simulation
sim = BrainSimulation(dt_ms=0.1)

# Build Retina → V1 → Visual Association network
sim.create_simple_network(n_sensory=10, n_processing=20, n_output=5)

# Simulate visual input (flashing pattern)
duration_ms = 150.0
steps = int(duration_ms / sim.dt_ms)

for step in range(steps):
    time_ms = step * sim.dt_ms
    epoch = int(time_ms / 10.0)
    
    # Flash on/off every 10 ms
    if epoch % 2 == 0:
        sim.stimulate_region("Retina", current=25.0)
    
    # Run one step
    sim.simulate_step(enable_learning=True)

# Check results
stats = sim.get_statistics()
print(f"Total spikes: {stats['total_spikes']}")
print(f"Average firing rate: {stats['average_firing_rate_hz']:.2f} Hz")

# Spike counts by region
for region in ['Retina', 'V1', 'Visual_Association']:
    spike_count = sum(
        len(sim.neurons[nid].spike_times) 
        for nid in sim.neuron_groups[region]
    )
    avg_rate = spike_count / len(sim.neuron_groups[region]) / (duration_ms / 1000.0)
    print(f"{region}: {spike_count} spikes ({avg_rate:.1f} Hz avg)")
```

**Expected Output:**
```
Total spikes: 636
Average firing rate: 121.14 Hz
Retina: 470 spikes (313.3 Hz avg)
V1: 151 spikes (50.3 Hz avg)
Visual_Association: 15 spikes (20.0 Hz avg)
```

This demonstrates **signal propagation** through the network!

## Running the Demonstrations

The `simulation_demo.py` script includes 5 comprehensive demonstrations:

```bash
python simulation_demo.py
```

**Demos included:**
1. **Single Neuron**: Action potential generation
2. **Nernst Potentials**: Ion equilibrium calculations
3. **Synaptic Transmission**: Pre → Post spike propagation
4. **Hebbian Learning**: Synaptic weight modification
5. **Network Simulation**: Multi-layer signal propagation

## Performance Considerations

### Current Capabilities
- **Scale**: Hundreds to thousands of neurons
- **Speed**: Real-time for small networks
- **Precision**: 0.1 ms time steps

### Scaling to Larger Networks

For networks with millions of neurons (approaching brain-scale), you would need:

1. **GPU Acceleration**: Use PyTorch or TensorFlow
```python
# Future implementation
import torch
voltages = torch.tensor(voltages, device='cuda')
```

2. **Vectorization**: Batch operations instead of loops
```python
# Instead of: for neuron in neurons: neuron.update()
# Use: voltages += (currents / capacitance) * dt
```

3. **Sparse Connectivity**: Use sparse matrices for synapses
```python
from scipy.sparse import csr_matrix
connectivity = csr_matrix((weights, (pre_ids, post_ids)))
```

4. **Specialized Libraries**: 
   - **Brian2**: High-level Python SNN simulator
   - **NEST**: Large-scale brain simulations
   - **Nengo**: Neural engineering framework

## Comparison with Static Atlas

Both systems coexist for different purposes:

| Feature | Static Atlas | Dynamic Simulation |
|---------|--------------|-------------------|
| Purpose | Knowledge reference | Actual computation |
| Neurons | Text descriptions | Stateful objects |
| Connectivity | Listed pathways | Functional connections |
| Time | No temporal aspect | Time steps forward |
| Learning | Describes LTP/LTD | Implements LTP/LTD |
| Output | Strings, LaTeX | Spike trains, voltages |
| Use case | Education, reference | Research, AI modeling |

## Future Enhancements

To move toward "sentient" behavior:

1. **Larger Scale**: Millions of neurons with GPU acceleration
2. **Realistic Morphology**: Multi-compartment neuron models
3. **Multiple Neurotransmitters**: Dopamine, serotonin dynamics
4. **Detailed Connectome**: Real brain connectivity data
5. **STDP**: Spike-timing-dependent plasticity
6. **Homeostasis**: Activity-dependent regulation
7. **Neuromodulation**: Global state changes (arousal, attention)
8. **Sensory Integration**: Multi-modal inputs
9. **Motor Output**: Actuator control
10. **Cognitive Loops**: Attention, memory, decision-making

## References

### Neuroscience
- Hodgkin & Huxley (1952): Action potential mechanism
- Hebb (1949): "Cells that fire together, wire together"
- Nernst (1888): Ion equilibrium potentials

### Computational Neuroscience
- Gerstner & Kistler (2002): Spiking Neuron Models
- Dayan & Abbott (2001): Theoretical Neuroscience
- Izhikevich (2007): Dynamical Systems in Neuroscience

### Software
- Brian2: https://brian2.readthedocs.io/
- NEST: https://nest-simulator.org/
- Nengo: https://www.nengo.ai/

## License

This implementation synthesizes knowledge from computational neuroscience, systems biology, and neural engineering research for educational and research purposes.
