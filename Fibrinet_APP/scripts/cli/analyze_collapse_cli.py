import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.managers.network.collapse_analysis_manager import CollapseAnalysisManager


def main():
    print("\n=== FibriNet Collapse Analyzer ===\n")
    
    # Parse arguments
    excel_path = None
    output_dir = None
    max_steps = None
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--out-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 1
        elif arg == "--max-steps" and i + 1 < len(args):
            try:
                max_steps = int(args[i + 1])
            except ValueError:
                print(f">>> Error: Invalid max-steps value: {args[i + 1]}")
                sys.exit(1)
            i += 1
        elif not arg.startswith("--"):
            excel_path = arg
        i += 1

    # Get input file
    if not excel_path:
        excel_path = input("Enter path to Excel (.xlsx) network file: ").strip()

    if not excel_path:
        print(">>> Error: No file path provided.")
        sys.exit(1)

    if not os.path.exists(excel_path):
        print(f">>> Error: File not found: {excel_path}")
        sys.exit(1)

    # Set output directory
    if not output_dir:
        base = os.path.splitext(os.path.basename(excel_path))[0]
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(_project_root, "exports", f"{base}_collapse_{ts}")

    try:
        manager = CollapseAnalysisManager()
        
        print("Starting iterative collapse analysis...")
        print(f"Input file: {excel_path}")
        print(f"Output directory: {output_dir}")
        if max_steps:
            print(f"Max steps: {max_steps}")
        print()
        
        result = manager.iterate_constrained_cuts(excel_path, output_dir, max_steps)
        
        if "error" in result:
            print(f">>> Error: {result['error']}")
            sys.exit(1)
        
        print("\n--- Analysis Complete ---")
        print(f"Total steps completed: {result['total_steps']}")
        print(f"Edges removed: {len(result['cumulative_removed'])}")
        print(f"Removal order: {', '.join(str(e) for e in result['cumulative_removed'])}")
        print(f"Final LCC nodes: {result['final_lcc_nodes']}")
        print(f"Final LCC edges: {result['final_lcc_edges']}")
        print(f"CSV log: {result['csv_path']}")
        print(f"Initial image: {result['initial_image']}")
        print("\nDone.\n")
        
    except Exception as ex:
        print(f">>> Fatal error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()