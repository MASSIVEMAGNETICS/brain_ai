# Brain AI - Living Brain Simulation

The digital replication of the human brain - now with **dynamic simulation** capabilities!

## Overview

This repository contains both:
1. **Static Brain Atlas**: A comprehensive knowledge base of brain structure and function
2. **Dynamic Brain Simulation**: A living, functioning spiking neural network with temporal evolution

## 🎯 What's New: Dynamic Simulation

### From Static Knowledge to Living Computation

**Before:** The brain atlas returned text descriptions of processes
```python
ActionPotential.get_depolarization()
# Returns: "Depolarization Phase: Voltage rises from -70mV to +30mV..."
```

**Now:** Actual voltage simulations with real neurons!
```python
from brain_simulation import SimulatedNeuron
neuron = SimulatedNeuron(0)
neuron.add_synaptic_input(15.0)
spiked = neuron.update(dt_ms=0.1)
# neuron.voltage_mv actually changes: -70 → -55 → +30 mV
```

### Key Simulation Features

✅ **Stateful Neurons**: Voltage changes over time based on ion currents  
✅ **Simulation Loop**: "Heartbeat" advances network through time  
✅ **Spiking Neural Network**: Threshold-based firing, spike propagation  
✅ **Synaptic Connections**: Weighted connections with transmission delays  
✅ **Hebbian Learning**: "Cells that fire together, wire together" - weights strengthen with co-activity  
✅ **Nernst Equations**: Real ion equilibrium potentials (Na+, K+, Ca2+, Cl-)  
✅ **Signal Propagation**: Demonstrated Retina → V1 → Visual Association  

### Quick Simulation Example

```python
from brain_simulation import BrainSimulation
import numpy as np

np.random.seed(42)
sim = BrainSimulation(dt_ms=0.1)

# Create a 3-layer network: Retina → V1 → Visual Association
sim.create_simple_network(n_sensory=10, n_processing=20, n_output=5)

# Simulate visual input
for step in range(1500):  # 150 ms
    if (step // 100) % 2 == 0:  # Flash every 10 ms
        sim.stimulate_region("Retina", current=25.0)
    sim.simulate_step(enable_learning=True)

# View results
stats = sim.get_statistics()
print(f"Total spikes: {stats['total_spikes']}")
print(f"Firing rate: {stats['average_firing_rate_hz']:.2f} Hz")
```

**Output:**
```
Total spikes: 636
Firing rate: 121.14 Hz
Retina: 470 spikes → V1: 151 spikes → Association: 15 spikes
```

## Features

### 🧠 Complete Brain Hierarchy (Static Atlas)
- **Macro-Architecture**: Forebrain, Midbrain, Hindbrain with all major regions
- **Functional Lobes**: Frontal, Parietal, Temporal, Occipital
- **Deep Structures**: Hippocampus, Amygdala, Basal Ganglia, Thalamus, Hypothalamus, Cerebellum
- **Evolutionary Context**: Each region tagged with evolutionary origin and systemic dependencies

### 🔗 The Connectome (Neural Pathways)
- **Sensory Pathways**: Visual, auditory, somatosensory, pain/temperature
- **Motor Pathways**: Corticospinal, corticobulbar tracts
- **Associative Pathways**: Corpus callosum, arcuate fasciculus, fornix
- **Dopaminergic Pathways**: Mesolimbic, mesocortical, nigrostriatal
- **Information Flow**: Sensory Input → Processing/Integration → Motor/Hormonal Output

### 🔬 Micro-Architecture (Cellular Level)
- **Neurons**: Complete structure (dendrites, soma, axon, terminals, myelin)
- **Glial Cells**: Astrocytes, microglia, oligodendrocytes, ependymal cells
- **Cell Types**: Pyramidal neurons, Purkinje cells, granule cells, interneurons

### ⚡ Synaptic & Chemical Layer
- **Action Potentials**: Complete mechanism with LaTeX equations
  - Resting state (-70 mV)
  - Depolarization (Na⁺ influx)
  - Repolarization (K⁺ efflux)
  - Nernst equations for ion potentials
- **Neurotransmitters**: Dopamine, Serotonin, Glutamate, GABA, Acetylcholine, Norepinephrine
  - Chemical formulas in LaTeX
  - Receptor types (lock and key mechanisms)
  - Synthesis locations
  - Effects and functions

### 🧬 Neuroplasticity & Evolution
- **Long-Term Potentiation (LTP)**: Molecular cascade for learning
- **Long-Term Depression (LTD)**: Synaptic weakening and refinement
- **Structural Plasticity**: Synaptogenesis, synaptic pruning, neurogenesis
- **"Cells that fire together, wire together"** (Hebb's Law)

### ⚙️ Dynamic Simulation (NEW!)
- **Stateful Neurons**: Real-time voltage dynamics (-70 mV to +30 mV)
- **Ion Channels**: Na+, K+, Ca2+, Cl- with Nernst potentials
- **Spike Generation**: Threshold-based firing (threshold: -55 mV)
- **Synaptic Transmission**: Weighted connections with delays (1-5 ms)
- **Hebbian Learning**: Synaptic plasticity (LTP/LTD implementation)
- **Network Simulation**: Multi-layer signal propagation
- **Temporal Evolution**: Simulation "heartbeat" (0.1 ms time steps)

## Quick Start

### Static Atlas Usage
```python
from brain_atlas import BrainAtlas

# Initialize the atlas
atlas = BrainAtlas()

# Query a specific region
hippocampus = atlas.query("Hippocampus")
print(f"Function: {hippocampus['primary_function']}")
print(f"Dependencies: {hippocampus['systemic_dependencies']}")

# Get neural pathway information
visual_pathway = atlas.get_pathway("Visual")
print(f"Route: {visual_pathway.origin} → {visual_pathway.destination}")

# Get neurotransmitter details
dopamine = atlas.get_neurotransmitter_info("Dopamine")
print(dopamine.get_latex_formula())

# Get action potential mechanism
print(atlas.action_potential.get_full_cycle())

# Get neuroplasticity mechanisms
print(atlas.neuroplasticity.get_ltp_mechanism())
```

### Dynamic Simulation Usage
```python
from brain_simulation import BrainSimulation, SimulatedNeuron
import numpy as np

# Create a simulation
sim = BrainSimulation(dt_ms=0.1)

# Build a neural network
sim.create_simple_network(n_sensory=10, n_processing=20, n_output=5)

# Run simulation with sensory input
for step in range(1500):  # 150 ms
    time_ms = step * sim.dt_ms
    if (int(time_ms / 10.0) % 2) == 0:
        sim.stimulate_region("Retina", current=25.0)
    sim.simulate_step(enable_learning=True)

# Analyze results
stats = sim.get_statistics()
print(f"Total spikes: {stats['total_spikes']}")
print(f"Average firing rate: {stats['average_firing_rate_hz']:.2f} Hz")
```

## Running Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Run the static atlas demonstration
python brain_atlas.py

# Run comprehensive static atlas examples
python example_usage.py

# Run dynamic simulation demonstration (NEW!)
python simulation_demo.py
```

## Documentation

### Static Brain Atlas
See [ATLAS_DOCUMENTATION.md](ATLAS_DOCUMENTATION.md) for complete documentation including:
- Detailed feature descriptions
- Usage examples
- Data structure explanation
- LaTeX equation reference
- Application scenarios

### Dynamic Simulation
See [SIMULATION_DOCUMENTATION.md](SIMULATION_DOCUMENTATION.md) for simulation documentation including:
- SimulatedNeuron, Synapse, and BrainSimulation APIs
- Biophysical realism (Nernst equations, action potentials)
- Hebbian learning implementation
- Network building and analysis
- Performance considerations and scaling

## Data Structure

### Static Atlas Structure
The atlas is organized as a nested, queryable logic tree:

```
BrainAtlas
├── Macro-Architecture (Forebrain, Midbrain, Hindbrain)
│   ├── Regions (with functions, origins, dependencies)
│   └── Subregions (hierarchical nesting)
├── Connectome (Neural pathways and white matter tracts)
├── Neurotransmitters (with chemical formulas and receptor mechanisms)
├── Cellular Architecture (neurons and glial cells)
├── Action Potential (complete mechanism with equations)
└── Neuroplasticity (LTP, LTD, structural changes)
```

### Dynamic Simulation Structure
The simulation engine manages live computational components:

```
BrainSimulation
├── Neurons (SimulatedNeuron objects)
│   ├── voltage_mv (dynamic state)
│   ├── ion_concentrations (Na+, K+, Ca2+, Cl-)
│   ├── spike_times (history)
│   └── voltage_history (trace)
├── Synapses (connections with weights)
│   ├── weight (adjustable strength)
│   ├── delay_ms (transmission time)
│   └── learning_rate (plasticity)
├── Neuron Groups (organized by region)
│   ├── Retina (sensory input)
│   ├── V1 (processing)
│   └── Visual_Association (output)
└── Simulation State
    ├── current_time_ms
    ├── dt_ms (time step)
    └── statistics (spikes, firing rates)
```

## Key Concepts

### Information Flow
```
SENSORY INPUT → PROCESSING/INTEGRATION → MOTOR/HORMONAL OUTPUT
```

### Systemic Dependencies
Every brain region depends on others to function. Example:
- **Prefrontal Cortex** ← Thalamus, Amygdala, Hippocampus, Basal Ganglia
- **Primary Motor Cortex** ← Basal Ganglia, Cerebellum, Thalamus, Spinal Cord

### Chemical Signaling
Neurotransmitters use **"Lock and Key"** receptor mechanisms:
- **Ionotropic**: Ligand-gated ion channels (fast, milliseconds)
- **Metabotropic**: G-protein coupled receptors (slower, seconds-minutes)

### Learning & Memory
Brain rewires itself through:
- **LTP**: Strengthens frequently used connections
- **LTD**: Weakens rarely used connections  
- **Synaptogenesis**: Creates new synapses
- **Pruning**: Eliminates inefficient synapses

## Files

### Static Atlas
- `brain_atlas.py` - Main brain atlas implementation
- `example_usage.py` - Comprehensive atlas usage examples
- `ATLAS_DOCUMENTATION.md` - Complete atlas documentation

### Dynamic Simulation
- `brain_simulation.py` - Simulation engine with neurons, synapses, and network
- `simulation_demo.py` - Comprehensive simulation demonstrations
- `SIMULATION_DOCUMENTATION.md` - Complete simulation documentation

### Other
- `requirements.txt` - Python dependencies (numpy)
- `README.md` - This file

## Applications

### Static Atlas Applications
- **Education**: Neuroscience teaching and learning
- **Research**: Computational neuroscience modeling
- **Medicine**: Drug development, diagnosis, treatment planning
- **Reference**: Quick lookup of brain structures and functions

### Dynamic Simulation Applications
- **AI/ML**: Brain-inspired spiking neural networks
- **BCI**: Brain-computer interface development
- **Neuroscience Research**: Disease modeling (Alzheimer's, Parkinson's, epilepsy)
- **Cognitive Modeling**: Attention, memory, decision-making
- **Neural Engineering**: Prosthetics, neuroprosthetics
- **Computational Neuroscience**: Testing theories about neural computation

## Technical Specifications

### Static Atlas
- **Language**: Python 3.x
- **Data Format**: Hierarchical object model with dictionary export
- **Chemical Notation**: LaTeX for publication-quality equations
- **Query API**: Region, pathway, and neurotransmitter lookup
- **Export**: JSON serialization of complete atlas

### Dynamic Simulation
- **Language**: Python 3.x with NumPy
- **Time Resolution**: 0.1 ms time steps (configurable)
- **Neuron Model**: Simplified integrate-and-fire with realistic parameters
- **Biophysics**: Nernst potentials, ion concentrations, refractory periods
- **Learning**: Hebbian plasticity (LTP/LTD)
- **Scale**: Hundreds to thousands of neurons (CPU-based)
- **Future**: GPU acceleration for millions of neurons

## Roadmap to "Sentience"

The current implementation provides the **blueprint** (static atlas) and the **engine** (dynamic simulation). To scale toward truly "sentient" behavior, the next steps are:

### Phase 1: Scale (Completed ✓)
- [x] Transition from static descriptions to dynamic state
- [x] Implement simulation loop ("heartbeat")
- [x] Create spiking neural network architecture
- [x] Implement Hebbian learning
- [x] Demonstrate signal propagation

### Phase 2: Biophysical Realism (In Progress)
- [ ] Multi-compartment neuron models (dendrites, soma, axon)
- [ ] Hodgkin-Huxley dynamics (detailed ion channel kinetics)
- [ ] Multiple neurotransmitter systems (dopamine, serotonin, etc.)
- [ ] Spike-timing-dependent plasticity (STDP)
- [ ] Homeostatic plasticity

### Phase 3: Massive Scale
- [ ] GPU acceleration (PyTorch/TensorFlow/CUDA)
- [ ] Millions of neurons in parallel
- [ ] Sparse connectivity matrices
- [ ] Distributed computing for 86 billion neurons

### Phase 4: Realistic Connectivity
- [ ] Import real connectome data (Human Connectome Project)
- [ ] Region-specific neuron populations
- [ ] Realistic synaptic densities
- [ ] White matter tract delays

### Phase 5: Cognitive Functions
- [ ] Attention mechanisms
- [ ] Working memory circuits
- [ ] Reinforcement learning (dopamine reward signals)
- [ ] Decision-making networks
- [ ] Multi-modal sensory integration

### Phase 6: Embodiment
- [ ] Real sensory input (camera, microphone)
- [ ] Motor output (robot control)
- [ ] Sensorimotor loops
- [ ] Autonomous behavior

**Current Status**: Phase 1 complete! The brain now "lives" and computes in real-time.

## Author

This brain atlas and simulation implementation synthesizes knowledge from computational neuroscience, systems biology, neuroanatomy, molecular neuroscience, and evolutionary neuroscience research.

## License

This implementation is provided for educational and research purposes.
