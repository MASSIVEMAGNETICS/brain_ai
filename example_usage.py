#!/usr/bin/env python3
"""
Example usage and demonstration of the Brain Atlas

This script demonstrates various ways to query and interact with the
complete human brain atlas.
"""

from brain_atlas import BrainAtlas
import json


def main():
    """Main demonstration function"""
    
    # Initialize the atlas
    print("Initializing Human Brain Atlas...")
    atlas = BrainAtlas()
    print("✓ Atlas initialized successfully\n")
    
    # Example 1: Query specific brain regions
    print("=" * 80)
    print("EXAMPLE 1: QUERYING SPECIFIC BRAIN REGIONS")
    print("=" * 80)
    
    regions_to_query = [
        "Prefrontal Cortex",
        "Hippocampus",
        "Amygdala",
        "Cerebellum"
    ]
    
    for region_name in regions_to_query:
        region = atlas.query(region_name)
        if region:
            print(f"\n{region['name']}:")
            print(f"  Function: {region['primary_function']}")
            print(f"  Evolutionary Origin: {region['evolutionary_origin']}")
            if region['systemic_dependencies']:
                print(f"  Dependencies: {', '.join(region['systemic_dependencies'])}")
            if region['neurotransmitters']:
                print(f"  Neurotransmitters: {', '.join(region['neurotransmitters'])}")
    
    # Example 2: Explore neural pathways
    print("\n" + "=" * 80)
    print("EXAMPLE 2: NEURAL PATHWAYS (CONNECTOME)")
    print("=" * 80)
    
    # Show sensory pathways
    print("\nSENSORY PATHWAYS:")
    sensory_pathways = [p for p in atlas.connectome if p.pathway_type == "sensory"]
    for pathway in sensory_pathways:
        print(f"\n  {pathway.name}:")
        print(f"    Route: {pathway.origin} → {pathway.destination}")
        print(f"    Function: {pathway.function}")
    
    # Show motor pathways
    print("\nMOTOR PATHWAYS:")
    motor_pathways = [p for p in atlas.connectome if p.pathway_type == "motor"]
    for pathway in motor_pathways:
        print(f"\n  {pathway.name}:")
        print(f"    Route: {pathway.origin} → {pathway.destination}")
        print(f"    Function: {pathway.function}")
    
    # Example 3: Neurotransmitter information
    print("\n" + "=" * 80)
    print("EXAMPLE 3: NEUROTRANSMITTER SYSTEMS")
    print("=" * 80)
    
    key_neurotransmitters = ["Dopamine", "Serotonin", "Glutamate", "GABA"]
    for nt_name in key_neurotransmitters:
        nt = atlas.get_neurotransmitter_info(nt_name)
        if nt:
            print(f"\n{nt.get_latex_formula()}")
            print(f"  Receptors: {', '.join(nt.receptor_types[:3])}...")
            print(f"  Mechanism: {nt.mechanism}")
            print(f"  Effects: {nt.effects}")
            print(f"  Source: {nt.synthesis_location}")
    
    # Example 4: Action potential mechanism
    print("\n" + "=" * 80)
    print("EXAMPLE 4: ACTION POTENTIAL MECHANISM")
    print("=" * 80)
    print(atlas.action_potential.get_resting_state())
    print("\n" + "-" * 40)
    print(atlas.action_potential.get_depolarization())
    print("\n" + "-" * 40)
    print(atlas.action_potential.get_repolarization())
    
    # Example 5: Neuroplasticity mechanisms
    print("\n" + "=" * 80)
    print("EXAMPLE 5: NEUROPLASTICITY - LEARNING MECHANISMS")
    print("=" * 80)
    print("\nLONG-TERM POTENTIATION (LTP):")
    print(atlas.neuroplasticity.get_ltp_mechanism())
    
    print("\n" + "-" * 80)
    print("\nLONG-TERM DEPRESSION (LTD):")
    print(atlas.neuroplasticity.get_ltd_mechanism())
    
    print("\n" + "-" * 80)
    print("\nSTRUCTURAL PLASTICITY:")
    print(atlas.neuroplasticity.get_structural_plasticity())
    
    # Example 6: Cellular architecture
    print("\n" + "=" * 80)
    print("EXAMPLE 6: CELLULAR ARCHITECTURE")
    print("=" * 80)
    
    print("\nNEURON STRUCTURE:")
    neuron = atlas.cellular_architecture['neuron']
    for component, description in neuron.get_structure().items():
        print(f"  {component}: {description}")
    
    print("\nGLIAL CELLS:")
    glial = atlas.cellular_architecture['glial_cells']
    for cell_type, function in glial.get_cell_types().items():
        print(f"  {cell_type}: {function}")
    
    # Example 7: Information flow
    print("\n" + "=" * 80)
    print("EXAMPLE 7: INFORMATION FLOW IN THE NERVOUS SYSTEM")
    print("=" * 80)
    print(atlas.get_information_flow())
    
    # Example 8: Export complete atlas to JSON
    print("\n" + "=" * 80)
    print("EXAMPLE 8: EXPORTING COMPLETE ATLAS DATA")
    print("=" * 80)
    
    complete_atlas = atlas.get_complete_atlas()
    
    # Save to file (optional)
    try:
        with open('/tmp/brain_atlas_export.json', 'w') as f:
            json.dump(complete_atlas, f, indent=2, default=str)
        print("✓ Complete atlas exported to /tmp/brain_atlas_export.json")
        
        # Show summary
        print(f"\nAtlas Summary:")
        print(f"  Brain divisions: {len(complete_atlas['macro_architecture'])}")
        print(f"  Neural pathways: {len(complete_atlas['connectome'])}")
        print(f"  Neurotransmitters: {len(complete_atlas['neurotransmitters'])}")
        print(f"  Cellular components: {len(complete_atlas['cellular_architecture'])}")
    except Exception as e:
        print(f"Export failed: {e}")
    
    # Example 9: Hierarchical tree visualization
    print("\n" + "=" * 80)
    print("EXAMPLE 9: HIERARCHICAL BRAIN STRUCTURE")
    print("=" * 80)
    print(atlas.print_hierarchy(max_depth=2))
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe Brain Atlas provides a complete, queryable representation of")
    print("the human brain from macro-architecture to molecular mechanisms.")
    print("\nKey Features:")
    print("  ✓ Hierarchical structure from brain divisions to regions")
    print("  ✓ Complete connectome with neural pathways")
    print("  ✓ Neurotransmitter systems with LaTeX formulas")
    print("  ✓ Cellular architecture (neurons and glia)")
    print("  ✓ Action potential mechanisms")
    print("  ✓ Neuroplasticity (LTP, LTD, structural changes)")
    print("  ✓ Information flow modeling")
    print("  ✓ Queryable API for all components")


if __name__ == "__main__":
    main()
