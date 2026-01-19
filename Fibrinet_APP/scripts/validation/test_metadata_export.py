"""
Test metadata export for publication reporting.

This demonstrates how to include numerical guards and assumptions
in your output files for peer review defense.
"""

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
import json


def test_metadata_export():
    """Test that metadata includes all required fields."""

    print("=" * 70)
    print("PUBLICATION METADATA TEST")
    print("=" * 70)

    # Create adapter (no network needed for metadata structure test)
    adapter = CoreV2GUIAdapter()

    # Get metadata
    metadata = adapter.get_simulation_metadata()

    # Verify critical sections exist
    required_sections = [
        'physics_engine',
        'integrator',
        'force_model',
        'rupture_model',
        'guards',  # CRITICAL for peer review
        'assumptions',  # CRITICAL for peer review
        'physical_constants',
        'validation'
    ]

    print("\nVerifying metadata structure...")
    all_present = True
    for section in required_sections:
        if section in metadata:
            print(f"  [{section:20s}] PRESENT")
        else:
            print(f"  [{section:20s}] MISSING!")
            all_present = False

    if not all_present:
        print("\nERROR: Missing required metadata sections")
        return False

    # Show critical fields
    print("\n" + "=" * 70)
    print("NUMERICAL GUARDS (Peer Review Defense)")
    print("=" * 70)
    guards = metadata['guards']
    for key, value in guards.items():
        if key == 'rationale':
            print(f"  Rationale: {value}")
        else:
            print(f"  {key:20s} = {value}")

    print("\n" + "=" * 70)
    print("MODEL ASSUMPTIONS (Peer Review Defense)")
    print("=" * 70)
    for i, assumption in enumerate(metadata['assumptions'], 1):
        print(f"  {i}. {assumption}")

    print("\n" + "=" * 70)
    print("VALIDATION STATUS")
    print("=" * 70)
    for test, status in metadata['validation'].items():
        print(f"  {test:30s} {status}")

    # Export to JSON
    output_file = "test_metadata.json"
    adapter.export_metadata_to_file(output_file)

    print("\n" + "=" * 70)
    print("USAGE IN PUBLICATION")
    print("=" * 70)
    print("""
When publishing results:

1. Export metadata with every simulation:
   adapter.export_metadata_to_file('experiment_001_metadata.json')

2. Include in your paper's Methods section:
   "Simulations used FibriNet Core V2 with WLC mechanics and
    stress-based Bell rupture model. Numerical guards (S_floor=0.05,
    max_strain=0.99) prevented overflow while preserving physics
    in the accessible regime. See supplementary metadata.json."

3. Attach metadata.json as supplementary material

4. When reviewer asks "Why didn't you model diffusion?":
   Point to assumptions[1]: "Uniform enzyme distribution (mean-field)"

5. When reviewer asks "What about your numerical clamps?":
   Point to guards section: All documented with rationale
""")

    print("=" * 70)
    print(f"[PASS] Metadata export test complete")
    print(f"       Output: {output_file}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_metadata_export()
    sys.exit(0 if success else 1)
