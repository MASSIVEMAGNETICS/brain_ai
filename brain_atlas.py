"""
Human Brain Atlas - A Complete, Dynamic, Multi-Dimensional Map

This module provides a recursive hierarchical representation of the human brain,
from macro-architectural systems down to quantum-biological molecular interactions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class NeurotransmitterType(Enum):
    """Major neurotransmitter types and their primary functions"""
    DOPAMINE = "reward, motivation, motor control"
    SEROTONIN = "mood regulation, sleep, appetite"
    GLUTAMATE = "primary excitatory neurotransmitter"
    GABA = "primary inhibitory neurotransmitter"
    ACETYLCHOLINE = "learning, memory, muscle activation"
    NOREPINEPHRINE = "alertness, arousal, fight-or-flight"
    ENDORPHINS = "pain relief, euphoria"


class IonType(Enum):
    """Ion types involved in action potentials"""
    SODIUM = "Na+"
    POTASSIUM = "K+"
    CALCIUM = "Ca2+"
    CHLORIDE = "Cl-"


@dataclass
class Region:
    """Represents a brain region with its properties and dependencies"""
    name: str
    primary_function: str
    evolutionary_origin: str
    systemic_dependencies: List[str] = field(default_factory=list)
    subregions: List['Region'] = field(default_factory=list)
    neurotransmitters: List[NeurotransmitterType] = field(default_factory=list)
    cell_types: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert region to dictionary representation"""
        return {
            'name': self.name,
            'primary_function': self.primary_function,
            'evolutionary_origin': self.evolutionary_origin,
            'systemic_dependencies': self.systemic_dependencies,
            'neurotransmitters': [nt.name for nt in self.neurotransmitters],
            'cell_types': self.cell_types,
            'subregions': [sr.to_dict() for sr in self.subregions]
        }


@dataclass
class NeuralPathway:
    """Represents a neural pathway connecting brain regions"""
    name: str
    origin: str
    destination: str
    pathway_type: str  # sensory, motor, associative
    tract_name: str  # white matter tract name
    function: str
    neurotransmitters: List[NeurotransmitterType] = field(default_factory=list)


@dataclass
class Neurotransmitter:
    """Detailed neurotransmitter information with receptor mechanisms"""
    name: str
    chemical_formula: str
    receptor_types: List[str]
    mechanism: str  # Lock and Key mechanism description
    effects: str
    synthesis_location: str
    
    def get_latex_formula(self) -> str:
        """Return LaTeX formatted chemical formula"""
        return f"$\\text{{{self.name}}}: {self.chemical_formula}$"


@dataclass
class ActionPotential:
    """Models the action potential mechanism"""
    
    @staticmethod
    def get_resting_state() -> str:
        """Resting membrane potential explanation"""
        return r"""
        Resting State:
        Membrane potential: $V_m = -70\,\text{mV}$
        
        Ion distribution maintained by Na+/K+-ATPase pump:
        $$3\text{Na}^+_{\text{in}} + 2\text{K}^+_{\text{out}} + \text{ATP} \rightarrow 3\text{Na}^+_{\text{out}} + 2\text{K}^+_{\text{in}} + \text{ADP} + P_i$$
        
        Concentration gradients:
        - $[\text{Na}^+]_{\text{out}} \approx 145\,\text{mM}$, $[\text{Na}^+]_{\text{in}} \approx 12\,\text{mM}$
        - $[\text{K}^+]_{\text{out}} \approx 4\,\text{mM}$, $[\text{K}^+]_{\text{in}} \approx 155\,\text{mM}$
        """
    
    @staticmethod
    def get_depolarization() -> str:
        """Depolarization phase explanation"""
        return r"""
        Depolarization Phase:
        Threshold: $V_m \geq -55\,\text{mV}$
        
        Voltage-gated Na+ channels open:
        $$\text{Na}^+_{\text{out}} \xrightarrow{\text{channels open}} \text{Na}^+_{\text{in}}$$
        
        Rapid depolarization to peak:
        $$V_m: -70\,\text{mV} \rightarrow +30\,\text{mV}$$ (in ~1 ms)
        
        Governed by Nernst equation:
        $$E_{\text{Na}} = \frac{RT}{zF}\ln\frac{[\text{Na}^+]_{\text{out}}}{[\text{Na}^+]_{\text{in}}} \approx +60\,\text{mV}$$
        """
    
    @staticmethod
    def get_repolarization() -> str:
        """Repolarization phase explanation"""
        return r"""
        Repolarization Phase:
        
        Na+ channels inactivate, K+ channels open:
        $$\text{K}^+_{\text{in}} \xrightarrow{\text{channels open}} \text{K}^+_{\text{out}}$$
        
        Return to resting potential:
        $$V_m: +30\,\text{mV} \rightarrow -70\,\text{mV}$$
        
        K+ Nernst potential:
        $$E_{\text{K}} = \frac{RT}{zF}\ln\frac{[\text{K}^+]_{\text{out}}}{[\text{K}^+]_{\text{in}}} \approx -90\,\text{mV}$$
        
        Hyperpolarization may occur briefly before return to resting state.
        """
    
    @staticmethod
    def get_full_cycle() -> str:
        """Complete action potential cycle"""
        return (
            ActionPotential.get_resting_state() + "\n\n" +
            ActionPotential.get_depolarization() + "\n\n" +
            ActionPotential.get_repolarization()
        )


@dataclass
class Neuron:
    """Structure and components of a neuron"""
    dendrites: str = "Input structures with dendritic spines for receiving signals"
    soma: str = "Cell body containing nucleus and organelles"
    axon: str = "Output structure conducting action potentials"
    axon_terminal: str = "Synaptic boutons releasing neurotransmitters"
    myelin_sheath: str = "Oligodendrocyte wrapping for saltatory conduction"
    nodes_of_ranvier: str = "Gaps in myelin where action potentials regenerate"
    
    def get_structure(self) -> Dict[str, str]:
        """Return complete neuron structure"""
        return {
            'dendrites': self.dendrites,
            'soma': self.soma,
            'axon': self.axon,
            'axon_terminal': self.axon_terminal,
            'myelin_sheath': self.myelin_sheath,
            'nodes_of_ranvier': self.nodes_of_ranvier
        }


@dataclass
class GlialCells:
    """Types and functions of glial cells"""
    astrocytes: str = "Blood-brain barrier, metabolic support, neurotransmitter regulation"
    microglia: str = "Immune surveillance, synaptic pruning, debris clearance"
    oligodendrocytes: str = "Myelination in CNS, saltatory conduction"
    ependymal_cells: str = "CSF production and circulation"
    schwann_cells: str = "Myelination in PNS"
    
    def get_cell_types(self) -> Dict[str, str]:
        """Return all glial cell types"""
        return {
            'astrocytes': self.astrocytes,
            'microglia': self.microglia,
            'oligodendrocytes': self.oligodendrocytes,
            'ependymal_cells': self.ependymal_cells,
            'schwann_cells': self.schwann_cells
        }


@dataclass
class Neuroplasticity:
    """Mechanisms of neuroplasticity and learning"""
    
    @staticmethod
    def get_ltp_mechanism() -> str:
        """Long-Term Potentiation mechanism"""
        return r"""
        Long-Term Potentiation (LTP):
        
        Molecular cascade for synaptic strengthening:
        
        1. High-frequency stimulation → Glutamate release
        $$\text{Glutamate} \xrightarrow{\text{binds}} \text{AMPA/NMDA receptors}$$
        
        2. NMDA receptor activation (requires depolarization to remove Mg2+ block):
        $$\text{Mg}^{2+} \text{ blockade removed} \rightarrow \text{Ca}^{2+} \text{ influx}$$
        
        3. Calcium-dependent kinase activation:
        $$[\text{Ca}^{2+}]_i \uparrow \rightarrow \text{CaMKII, PKC activation}$$
        
        4. Increased AMPA receptor insertion:
        $$\text{AMPA receptors}_{\text{cytoplasm}} \xrightarrow{\text{trafficking}} \text{AMPA receptors}_{\text{membrane}}$$
        
        5. Enhanced synaptic strength:
        $$\text{EPSP amplitude} \uparrow \text{ (potentiation lasting hours to days)}$$
        
        Result: "Cells that fire together, wire together" (Hebb's Law)
        """
    
    @staticmethod
    def get_ltd_mechanism() -> str:
        """Long-Term Depression mechanism"""
        return r"""
        Long-Term Depression (LTD):
        
        Molecular cascade for synaptic weakening:
        
        1. Low-frequency stimulation → Modest Ca2+ influx
        $$\text{Low } [\text{Ca}^{2+}]_i \text{ (below LTP threshold)}$$
        
        2. Phosphatase activation instead of kinases:
        $$[\text{Ca}^{2+}]_i \uparrow \text{ (moderate)} \rightarrow \text{Calcineurin, PP1 activation}$$
        
        3. AMPA receptor internalization:
        $$\text{AMPA receptors}_{\text{membrane}} \xrightarrow{\text{endocytosis}} \text{AMPA receptors}_{\text{cytoplasm}}$$
        
        4. Reduced synaptic strength:
        $$\text{EPSP amplitude} \downarrow \text{ (depression lasting hours to days)}$$
        
        Result: Synaptic pruning and refinement of neural circuits
        """
    
    @staticmethod
    def get_structural_plasticity() -> str:
        """Structural changes in neuroplasticity"""
        return """
        Structural Neuroplasticity:
        
        Synaptogenesis (Formation of new synapses):
        - Dendritic spine growth and stabilization
        - Axonal sprouting to new targets
        - Guided by neurotrophic factors (BDNF, NGF)
        - Peak during critical periods, continues throughout life
        
        Synaptic Pruning (Elimination of unused synapses):
        - Microglial-mediated synaptic engulfment
        - Activity-dependent: "Use it or lose it"
        - Peaks during adolescence (prefrontal cortex)
        - Essential for circuit refinement and efficiency
        
        Myelination Changes:
        - Activity-dependent myelin plasticity
        - Increases conduction velocity on frequently used pathways
        - Continues into adulthood in associative areas
        
        Neurogenesis (limited in adult brain):
        - Hippocampal dentate gyrus (memory formation)
        - Subventricular zone (olfactory bulb)
        - Enhanced by exercise, learning, enriched environment
        """


class BrainAtlas:
    """
    Complete hierarchical atlas of the human brain.
    Queryable from macro-architecture to molecular interactions.
    """
    
    def __init__(self):
        """Initialize the complete brain atlas"""
        self.macro_architecture = self._build_macro_architecture()
        self.connectome = self._build_connectome()
        self.neurotransmitters = self._build_neurotransmitters()
        self.cellular_architecture = self._build_cellular_architecture()
        self.neuroplasticity = Neuroplasticity()
        self.action_potential = ActionPotential()
        
    def _build_macro_architecture(self) -> Dict[str, Region]:
        """Build the macro-architectural hierarchy"""
        
        # Forebrain (Prosencephalon)
        prefrontal_cortex = Region(
            name="Prefrontal Cortex",
            primary_function="Executive function, decision-making, personality, impulse control",
            evolutionary_origin="Highly expanded in primates, especially humans (Neocortex)",
            systemic_dependencies=["Thalamus", "Amygdala", "Hippocampus", "Basal Ganglia"],
            neurotransmitters=[NeurotransmitterType.DOPAMINE, NeurotransmitterType.SEROTONIN],
            cell_types={
                "Pyramidal neurons": "Excitatory projection neurons",
                "Interneurons": "Inhibitory local circuits (GABAergic)"
            }
        )
        
        motor_cortex = Region(
            name="Primary Motor Cortex (M1)",
            primary_function="Voluntary movement execution",
            evolutionary_origin="Neocortex",
            systemic_dependencies=["Basal Ganglia", "Cerebellum", "Thalamus", "Spinal Cord"],
            neurotransmitters=[NeurotransmitterType.GLUTAMATE, NeurotransmitterType.GABA],
            cell_types={
                "Betz cells": "Giant pyramidal neurons for motor control",
                "Interneurons": "Local circuit modulation"
            }
        )
        
        somatosensory_cortex = Region(
            name="Primary Somatosensory Cortex (S1)",
            primary_function="Tactile sensation processing",
            evolutionary_origin="Neocortex",
            systemic_dependencies=["Thalamus (VPL/VPM)", "Spinal Cord", "Brainstem"],
            neurotransmitters=[NeurotransmitterType.GLUTAMATE]
        )
        
        frontal_lobe = Region(
            name="Frontal Lobe",
            primary_function="Motor control, executive function, speech production",
            evolutionary_origin="Telencephalon (Forebrain derivative)",
            systemic_dependencies=["Parietal Lobe", "Temporal Lobe", "Subcortical structures"],
            subregions=[prefrontal_cortex, motor_cortex]
        )
        
        parietal_lobe = Region(
            name="Parietal Lobe",
            primary_function="Somatosensory processing, spatial awareness",
            evolutionary_origin="Telencephalon",
            systemic_dependencies=["Frontal Lobe", "Occipital Lobe", "Thalamus"],
            subregions=[somatosensory_cortex]
        )
        
        temporal_lobe = Region(
            name="Temporal Lobe",
            primary_function="Auditory processing, memory, language comprehension",
            evolutionary_origin="Telencephalon",
            systemic_dependencies=["Hippocampus", "Amygdala", "Frontal Lobe"]
        )
        
        occipital_lobe = Region(
            name="Occipital Lobe",
            primary_function="Visual processing",
            evolutionary_origin="Telencephalon",
            systemic_dependencies=["Thalamus (LGN)", "Retina", "Parietal Lobe"]
        )
        
        hippocampus = Region(
            name="Hippocampus",
            primary_function="Memory formation and consolidation, spatial navigation",
            evolutionary_origin="Archicortex (ancient cortex)",
            systemic_dependencies=["Entorhinal Cortex", "Prefrontal Cortex", "Amygdala"],
            neurotransmitters=[NeurotransmitterType.GLUTAMATE, NeurotransmitterType.GABA],
            cell_types={
                "CA1 pyramidal cells": "Memory encoding and retrieval",
                "CA3 pyramidal cells": "Pattern completion",
                "Dentate gyrus granule cells": "Pattern separation, neurogenesis"
            }
        )
        
        amygdala = Region(
            name="Amygdala",
            primary_function="Emotional processing, fear conditioning, threat detection",
            evolutionary_origin="Paleocortex (Limbic system)",
            systemic_dependencies=["Hippocampus", "Prefrontal Cortex", "Hypothalamus", "Sensory Cortices"],
            neurotransmitters=[NeurotransmitterType.GABA, NeurotransmitterType.GLUTAMATE]
        )
        
        basal_ganglia = Region(
            name="Basal Ganglia",
            primary_function="Motor control, procedural learning, habit formation, reward",
            evolutionary_origin="Subcortical telencephalon",
            systemic_dependencies=["Motor Cortex", "Prefrontal Cortex", "Thalamus", "Substantia Nigra"],
            neurotransmitters=[NeurotransmitterType.DOPAMINE, NeurotransmitterType.GABA]
        )
        
        thalamus = Region(
            name="Thalamus",
            primary_function="Sensory relay station, consciousness, arousal",
            evolutionary_origin="Diencephalon",
            systemic_dependencies=["All Cortical Areas", "Reticular Formation", "Basal Ganglia"],
            neurotransmitters=[NeurotransmitterType.GLUTAMATE, NeurotransmitterType.GABA]
        )
        
        hypothalamus = Region(
            name="Hypothalamus",
            primary_function="Homeostasis, hormone regulation, autonomic control",
            evolutionary_origin="Diencephalon",
            systemic_dependencies=["Pituitary Gland", "Amygdala", "Brainstem"],
            neurotransmitters=[NeurotransmitterType.DOPAMINE, NeurotransmitterType.SEROTONIN]
        )
        
        forebrain = Region(
            name="Forebrain (Prosencephalon)",
            primary_function="Higher cognitive functions, sensory processing, voluntary movement",
            evolutionary_origin="Most recent evolutionary development",
            subregions=[
                frontal_lobe, parietal_lobe, temporal_lobe, occipital_lobe,
                hippocampus, amygdala, basal_ganglia, thalamus, hypothalamus
            ]
        )
        
        # Midbrain (Mesencephalon)
        substantia_nigra = Region(
            name="Substantia Nigra",
            primary_function="Dopamine production, motor control",
            evolutionary_origin="Mesencephalon",
            systemic_dependencies=["Basal Ganglia", "Motor Cortex"],
            neurotransmitters=[NeurotransmitterType.DOPAMINE]
        )
        
        superior_colliculus = Region(
            name="Superior Colliculus",
            primary_function="Visual attention, eye movements, multisensory integration",
            evolutionary_origin="Mesencephalon (ancient visual system)",
            systemic_dependencies=["Retina", "Visual Cortex", "Oculomotor nuclei"]
        )
        
        inferior_colliculus = Region(
            name="Inferior Colliculus",
            primary_function="Auditory processing relay",
            evolutionary_origin="Mesencephalon",
            systemic_dependencies=["Cochlear nuclei", "Auditory Cortex"]
        )
        
        midbrain = Region(
            name="Midbrain (Mesencephalon)",
            primary_function="Motor control, sensory relay, arousal",
            evolutionary_origin="Middle brain vesicle in development",
            subregions=[substantia_nigra, superior_colliculus, inferior_colliculus]
        )
        
        # Hindbrain (Rhombencephalon)
        cerebellum = Region(
            name="Cerebellum",
            primary_function="Motor coordination, balance, motor learning, timing",
            evolutionary_origin="Rhombencephalon (expanded in birds and mammals)",
            systemic_dependencies=["Motor Cortex", "Vestibular System", "Spinal Cord", "Pons"],
            neurotransmitters=[NeurotransmitterType.GABA, NeurotransmitterType.GLUTAMATE],
            cell_types={
                "Purkinje cells": "Primary output neurons (GABAergic)",
                "Granule cells": "Massive parallel processing (most numerous neurons in brain)",
                "Deep cerebellar nuclei": "Final output to motor systems"
            }
        )
        
        pons = Region(
            name="Pons",
            primary_function="Relay between cerebrum and cerebellum, sleep regulation",
            evolutionary_origin="Metencephalon",
            systemic_dependencies=["Cerebellum", "Medulla", "Cerebral Cortex"]
        )
        
        medulla_oblongata = Region(
            name="Medulla Oblongata",
            primary_function="Autonomic functions (breathing, heart rate, blood pressure)",
            evolutionary_origin="Myelencephalon (most ancient brain region)",
            systemic_dependencies=["Spinal Cord", "Hypothalamus", "Pons"],
            neurotransmitters=[NeurotransmitterType.NOREPINEPHRINE, NeurotransmitterType.SEROTONIN]
        )
        
        hindbrain = Region(
            name="Hindbrain (Rhombencephalon)",
            primary_function="Motor coordination, vital autonomic functions",
            evolutionary_origin="Most ancient brain division",
            subregions=[cerebellum, pons, medulla_oblongata]
        )
        
        return {
            'Forebrain': forebrain,
            'Midbrain': midbrain,
            'Hindbrain': hindbrain
        }
    
    def _build_connectome(self) -> List[NeuralPathway]:
        """Build major white matter tracts and neural pathways"""
        pathways = [
            # Sensory pathways
            NeuralPathway(
                name="Visual Pathway",
                origin="Retina",
                destination="Primary Visual Cortex (V1)",
                pathway_type="sensory",
                tract_name="Optic Radiation",
                function="Transmits visual information from retina through LGN to V1",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Dorsal Column-Medial Lemniscal Pathway",
                origin="Peripheral mechanoreceptors",
                destination="Primary Somatosensory Cortex",
                pathway_type="sensory",
                tract_name="Medial Lemniscus",
                function="Fine touch, vibration, proprioception",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Spinothalamic Tract",
                origin="Spinal Cord",
                destination="Thalamus → Somatosensory Cortex",
                pathway_type="sensory",
                tract_name="Anterolateral System",
                function="Pain and temperature sensation",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Auditory Pathway",
                origin="Cochlea",
                destination="Primary Auditory Cortex",
                pathway_type="sensory",
                tract_name="Auditory Radiation",
                function="Sound processing from ear to cortex",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            
            # Motor pathways
            NeuralPathway(
                name="Corticospinal Tract",
                origin="Primary Motor Cortex",
                destination="Spinal Motor Neurons",
                pathway_type="motor",
                tract_name="Pyramidal Tract",
                function="Voluntary motor control of limbs and trunk",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Corticobulbar Tract",
                origin="Motor Cortex",
                destination="Cranial Nerve Motor Nuclei",
                pathway_type="motor",
                tract_name="Corticobulbar Fibers",
                function="Voluntary control of face, head, neck muscles",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            
            # Associative pathways
            NeuralPathway(
                name="Corpus Callosum",
                origin="Left Hemisphere",
                destination="Right Hemisphere",
                pathway_type="associative",
                tract_name="Corpus Callosum",
                function="Inter-hemispheric communication",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Arcuate Fasciculus",
                origin="Wernicke's Area (temporal)",
                destination="Broca's Area (frontal)",
                pathway_type="associative",
                tract_name="Arcuate Fasciculus",
                function="Language processing - connects comprehension to production",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE]
            ),
            NeuralPathway(
                name="Fornix",
                origin="Hippocampus",
                destination="Mammillary Bodies → Thalamus",
                pathway_type="associative",
                tract_name="Fornix",
                function="Memory consolidation and retrieval",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE, NeurotransmitterType.ACETYLCHOLINE]
            ),
            
            # Limbic pathways
            NeuralPathway(
                name="Cingulum Bundle",
                origin="Cingulate Cortex",
                destination="Entorhinal Cortex → Hippocampus",
                pathway_type="associative",
                tract_name="Cingulum",
                function="Emotion, memory, executive control integration",
                neurotransmitters=[NeurotransmitterType.GLUTAMATE, NeurotransmitterType.SEROTONIN]
            ),
            
            # Dopaminergic pathways
            NeuralPathway(
                name="Mesolimbic Pathway",
                origin="Ventral Tegmental Area (VTA)",
                destination="Nucleus Accumbens, Amygdala",
                pathway_type="associative",
                tract_name="Mesolimbic Tract",
                function="Reward, motivation, addiction",
                neurotransmitters=[NeurotransmitterType.DOPAMINE]
            ),
            NeuralPathway(
                name="Mesocortical Pathway",
                origin="Ventral Tegmental Area (VTA)",
                destination="Prefrontal Cortex",
                pathway_type="associative",
                tract_name="Mesocortical Tract",
                function="Executive function, cognition, working memory",
                neurotransmitters=[NeurotransmitterType.DOPAMINE]
            ),
            NeuralPathway(
                name="Nigrostriatal Pathway",
                origin="Substantia Nigra",
                destination="Striatum (Basal Ganglia)",
                pathway_type="motor",
                tract_name="Nigrostriatal Tract",
                function="Motor control (degeneration causes Parkinson's disease)",
                neurotransmitters=[NeurotransmitterType.DOPAMINE]
            ),
        ]
        
        return pathways
    
    def _build_neurotransmitters(self) -> Dict[str, Neurotransmitter]:
        """Build detailed neurotransmitter database"""
        neurotransmitters = {
            'Dopamine': Neurotransmitter(
                name="Dopamine",
                chemical_formula=r"\text{C}_8\text{H}_{11}\text{NO}_2",
                receptor_types=["D1 (excitatory)", "D2 (inhibitory)", "D3", "D4", "D5"],
                mechanism="G-protein coupled receptors (GPCRs) - metabotropic signaling",
                effects="Reward, motivation, motor control, attention, learning",
                synthesis_location="Substantia Nigra, Ventral Tegmental Area"
            ),
            'Serotonin': Neurotransmitter(
                name="Serotonin (5-HT)",
                chemical_formula=r"\text{C}_{10}\text{H}_{12}\text{N}_2\text{O}",
                receptor_types=["5-HT1A-F", "5-HT2A-C", "5-HT3 (ionotropic)", "5-HT4-7"],
                mechanism="Mostly GPCRs (except 5-HT3 which is ligand-gated ion channel)",
                effects="Mood regulation, sleep, appetite, aggression, social behavior",
                synthesis_location="Raphe Nuclei (brainstem)"
            ),
            'Glutamate': Neurotransmitter(
                name="Glutamate",
                chemical_formula=r"\text{C}_5\text{H}_9\text{NO}_4",
                receptor_types=[
                    "AMPA (fast excitatory)",
                    "NMDA (Ca2+ permeable, plasticity)",
                    "Kainate",
                    "mGluR1-8 (metabotropic)"
                ],
                mechanism="Ionotropic (AMPA, NMDA, Kainate) and metabotropic (mGluR) receptors",
                effects="Primary excitatory neurotransmitter, learning, memory, plasticity",
                synthesis_location="Synthesized from glutamine in presynaptic terminals"
            ),
            'GABA': Neurotransmitter(
                name="GABA (γ-Aminobutyric acid)",
                chemical_formula=r"\text{C}_4\text{H}_9\text{NO}_2",
                receptor_types=[
                    "GABA-A (ionotropic, Cl- influx)",
                    "GABA-B (metabotropic, K+ efflux)"
                ],
                mechanism="GABA-A: ligand-gated Cl- channel; GABA-B: GPCR",
                effects="Primary inhibitory neurotransmitter, anxiety reduction, seizure prevention",
                synthesis_location="Synthesized from glutamate via GAD enzyme in inhibitory interneurons"
            ),
            'Acetylcholine': Neurotransmitter(
                name="Acetylcholine (ACh)",
                chemical_formula=r"\text{C}_7\text{H}_{16}\text{NO}_2^+",
                receptor_types=[
                    "Nicotinic (ionotropic, Na+/K+ channels)",
                    "Muscarinic M1-M5 (metabotropic GPCRs)"
                ],
                mechanism="Nicotinic: ligand-gated ion channel; Muscarinic: GPCR pathways",
                effects="Neuromuscular junction, attention, learning, memory, arousal",
                synthesis_location="Basal Forebrain (Nucleus Basalis), motor neurons"
            ),
            'Norepinephrine': Neurotransmitter(
                name="Norepinephrine (Noradrenaline)",
                chemical_formula=r"\text{C}_8\text{H}_{11}\text{NO}_3",
                receptor_types=["α1", "α2", "β1", "β2", "β3 (all GPCRs)"],
                mechanism="G-protein coupled receptors affecting cAMP and Ca2+ signaling",
                effects="Alertness, arousal, stress response, attention, fight-or-flight",
                synthesis_location="Locus Coeruleus (pons)"
            )
        }
        
        return neurotransmitters
    
    def _build_cellular_architecture(self) -> Dict[str, Any]:
        """Build cellular-level structures"""
        return {
            'neuron': Neuron(),
            'glial_cells': GlialCells(),
            'neuron_types': {
                'Pyramidal neurons': 'Excitatory projection neurons with apical and basal dendrites',
                'Stellate neurons': 'Star-shaped interneurons, local processing',
                'Purkinje cells': 'Large cerebellar neurons with extensive dendritic trees',
                'Granule cells': 'Small, numerous neurons (cerebellum and hippocampus)',
                'Chandelier cells': 'Inhibitory interneurons targeting axon initial segments',
                'Basket cells': 'Inhibitory interneurons forming basket-like synapses around soma'
            }
        }
    
    def get_information_flow(self) -> str:
        """Return the complete information flow through the nervous system"""
        return """
        INFORMATION FLOW IN THE NERVOUS SYSTEM:
        
        SENSORY INPUT → PROCESSING/INTEGRATION → MOTOR/HORMONAL OUTPUT
        
        1. SENSORY INPUT:
           External stimuli → Sensory receptors → Sensory neurons → Spinal cord/Brainstem
           
           Examples:
           - Visual: Retina → LGN (thalamus) → Primary Visual Cortex (V1)
           - Tactile: Mechanoreceptors → Dorsal Column → VPL thalamus → S1
           - Auditory: Cochlea → Superior Olive → MGN (thalamus) → A1
           - Pain: Nociceptors → Spinothalamic tract → VPL thalamus → S1/Insula
        
        2. PROCESSING/INTEGRATION:
           a) Thalamic relay (except olfaction)
           b) Primary sensory cortices (initial processing)
           c) Association cortices (multimodal integration)
           d) Prefrontal cortex (executive decision-making)
           e) Limbic system (emotional valence, memory context)
           
           Parallel processing streams:
           - Dorsal stream: "Where" pathway (parietal) - spatial processing
           - Ventral stream: "What" pathway (temporal) - object recognition
        
        3. MOTOR/HORMONAL OUTPUT:
           
           Motor Output:
           Prefrontal/Premotor planning → Primary Motor Cortex (M1) → 
           Corticospinal tract → Spinal motor neurons → Muscles
           
           Modulation by:
           - Basal Ganglia: Action selection, movement initiation
           - Cerebellum: Coordination, timing, error correction
           
           Hormonal Output:
           Hypothalamus → Pituitary → Endocrine glands
           - HPA axis: Stress response (cortisol)
           - HPG axis: Reproduction (sex hormones)
           - HPT axis: Metabolism (thyroid hormones)
           
           Autonomic Output:
           Hypothalamus/Brainstem → Sympathetic/Parasympathetic → Organs
           - Sympathetic: Fight-or-flight
           - Parasympathetic: Rest-and-digest
        
        FEEDBACK LOOPS:
           - Sensory feedback from movement → Motor correction
           - Hormonal feedback → Hypothalamic regulation
           - Reward prediction error → Dopaminergic learning signals
           - Error signals from cerebellum → Motor learning
        """
    
    def query(self, region_name: str) -> Optional[Dict[str, Any]]:
        """Query the atlas for a specific brain region"""
        def search_region(region: Region, target: str) -> Optional[Dict[str, Any]]:
            if region.name.lower() == target.lower():
                return region.to_dict()
            for subregion in region.subregions:
                result = search_region(subregion, target)
                if result:
                    return result
            return None
        
        for division in self.macro_architecture.values():
            result = search_region(division, region_name)
            if result:
                return result
        return None
    
    def get_pathway(self, pathway_name: str) -> Optional[NeuralPathway]:
        """Query connectome for a specific pathway"""
        for pathway in self.connectome:
            if pathway_name.lower() in pathway.name.lower():
                return pathway
        return None
    
    def get_neurotransmitter_info(self, nt_name: str) -> Optional[Neurotransmitter]:
        """Get detailed information about a neurotransmitter"""
        for name, nt in self.neurotransmitters.items():
            if nt_name.lower() in name.lower():
                return nt
        return None
    
    def print_hierarchy(self, max_depth: int = 3) -> str:
        """Print the complete hierarchical structure"""
        output = ["=" * 80]
        output.append("HUMAN BRAIN ATLAS - HIERARCHICAL STRUCTURE")
        output.append("=" * 80)
        
        def print_region(region: Region, depth: int = 0, prefix: str = ""):
            if depth > max_depth:
                return []
            
            lines = []
            indent = "  " * depth
            lines.append(f"{indent}{prefix}{region.name}")
            lines.append(f"{indent}  Function: {region.primary_function}")
            lines.append(f"{indent}  Origin: {region.evolutionary_origin}")
            
            if region.systemic_dependencies:
                lines.append(f"{indent}  Dependencies: {', '.join(region.systemic_dependencies)}")
            
            if region.neurotransmitters:
                nt_names = [nt.name for nt in region.neurotransmitters]
                lines.append(f"{indent}  Neurotransmitters: {', '.join(nt_names)}")
            
            lines.append("")
            
            for i, subregion in enumerate(region.subregions):
                is_last = i == len(region.subregions) - 1
                sub_prefix = "└─ " if is_last else "├─ "
                lines.extend(print_region(subregion, depth + 1, sub_prefix))
            
            return lines
        
        for division_name, division in self.macro_architecture.items():
            output.append(f"\n{'=' * 80}")
            output.append(f"{division_name}")
            output.append(f"{'=' * 80}")
            output.extend(print_region(division))
        
        return "\n".join(output)
    
    def get_complete_atlas(self) -> Dict[str, Any]:
        """Return the complete brain atlas as a nested dictionary"""
        return {
            'macro_architecture': {
                name: region.to_dict() 
                for name, region in self.macro_architecture.items()
            },
            'connectome': [
                {
                    'name': p.name,
                    'origin': p.origin,
                    'destination': p.destination,
                    'type': p.pathway_type,
                    'tract': p.tract_name,
                    'function': p.function,
                    'neurotransmitters': [nt.name for nt in p.neurotransmitters]
                }
                for p in self.connectome
            ],
            'neurotransmitters': {
                name: {
                    'formula': nt.get_latex_formula(),
                    'receptors': nt.receptor_types,
                    'mechanism': nt.mechanism,
                    'effects': nt.effects,
                    'synthesis': nt.synthesis_location
                }
                for name, nt in self.neurotransmitters.items()
            },
            'cellular_architecture': self.cellular_architecture,
            'action_potential': {
                'resting_state': self.action_potential.get_resting_state(),
                'depolarization': self.action_potential.get_depolarization(),
                'repolarization': self.action_potential.get_repolarization(),
                'complete_cycle': self.action_potential.get_full_cycle()
            },
            'neuroplasticity': {
                'ltp': self.neuroplasticity.get_ltp_mechanism(),
                'ltd': self.neuroplasticity.get_ltd_mechanism(),
                'structural': self.neuroplasticity.get_structural_plasticity()
            },
            'information_flow': self.get_information_flow()
        }


def main():
    """Demonstration of the brain atlas functionality"""
    atlas = BrainAtlas()
    
    print(atlas.print_hierarchy(max_depth=2))
    
    print("\n" + "=" * 80)
    print("CONNECTOME - MAJOR NEURAL PATHWAYS")
    print("=" * 80)
    for pathway in atlas.connectome:
        print(f"\n{pathway.name}:")
        print(f"  {pathway.origin} → {pathway.destination}")
        print(f"  Type: {pathway.pathway_type}")
        print(f"  Tract: {pathway.tract_name}")
        print(f"  Function: {pathway.function}")
    
    print("\n" + "=" * 80)
    print("NEUROTRANSMITTERS")
    print("=" * 80)
    for name, nt in atlas.neurotransmitters.items():
        print(f"\n{nt.get_latex_formula()}")
        print(f"  Receptors: {', '.join(nt.receptor_types)}")
        print(f"  Mechanism: {nt.mechanism}")
        print(f"  Effects: {nt.effects}")
    
    print("\n" + "=" * 80)
    print("ACTION POTENTIAL MECHANISM")
    print("=" * 80)
    print(atlas.action_potential.get_full_cycle())
    
    print("\n" + "=" * 80)
    print("NEUROPLASTICITY MECHANISMS")
    print("=" * 80)
    print(atlas.neuroplasticity.get_ltp_mechanism())
    print("\n")
    print(atlas.neuroplasticity.get_ltd_mechanism())
    
    print("\n" + "=" * 80)
    print("INFORMATION FLOW")
    print("=" * 80)
    print(atlas.get_information_flow())


if __name__ == "__main__":
    main()
