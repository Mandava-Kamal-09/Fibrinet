
import sys
import os
from src.managers.network.collapse_analysis_manager import CollapseAnalysisManager


def main():
    print("\n=== FibriNet Collapse Analyzer ===\n")
    fast = False
    excel_path = None
    constrain = False
    image_path = None
    iterate = False
    output_dir = None
    save_csv = False
    max_steps = None

    # Args: [excel_path] [--fast] [--constrain-center] [--image path] [--iterate] [--out-dir path] [--save-csv] [--max-steps N]
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        low = arg.lower()
        if low == "--fast":
            fast = True
        elif low == "--constrain-center":
            constrain = True
        elif low == "--iterate":
            iterate = True
        elif low == "--save-csv":
            save_csv = True
        elif low == "--image":
            if i + 1 < len(args):
                image_path = args[i + 1]
                i += 1
        elif low == "--out-dir":
            if i + 1 < len(args):
                output_dir = args[i + 1]
                i += 1
        elif low == "--max-steps":
            if i + 1 < len(args):
                try:
                    max_steps = int(args[i + 1])
                except ValueError:
                    print(f">>> Error: Invalid max-steps value: {args[i + 1]}")
                    sys.exit(1)
                i += 1
        else:
            excel_path = arg
        i += 1

    if not excel_path:
        excel_path = input("Enter path to Excel (.xlsx) network file: ").strip()

    if not excel_path:
        print(">>> Error: No file path provided.")
        sys.exit(1)

    if not os.path.exists(excel_path):
        print(f">>> Error: File not found: {excel_path}")
        sys.exit(1)

    try:
        manager = CollapseAnalysisManager()

        if iterate:
            # Iterative mode
            if not output_dir:
                base = os.path.splitext(os.path.basename(excel_path))[0]
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(os.path.dirname(__file__), "exports", f"{base}_iter_{ts}")
            
            print("Starting iterative analysis...")
            print(f"Output directory: {output_dir}")
            if max_steps:
                print(f"Max steps: {max_steps}")
            print()
            
            result = manager.iterate_constrained_cuts(excel_path, output_dir, max_steps)
            
            if "error" in result:
                print(f">>> Error: {result['error']}")
                sys.exit(1)
            
            print("\n--- Iterative Analysis Complete ---")
            print(f"Total steps completed: {result['total_steps']}")
            print(f"Edges removed: {len(result['cumulative_removed'])}")
            print(f"Removal order: {', '.join(str(e) for e in result['cumulative_removed'])}")
            print(f"Final LCC nodes: {result['final_lcc_nodes']}")
            print(f"Final LCC edges: {result['final_lcc_edges']}")
            print(f"CSV log: {result['csv_path']}")
            print("\nDone.\n")
        else:
            # Single analysis mode
            if constrain and not image_path:
                try:
                    base = os.path.splitext(os.path.basename(excel_path))[0]
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    exports_dir = os.path.join(os.path.dirname(__file__), "exports")
                    os.makedirs(exports_dir, exist_ok=True)
                    image_path = os.path.join(exports_dir, f"{base}_flush_{ts}.png")
                except Exception as _ex:
                    image_path = None
            
            result = manager.find_minimum_degradations_to_collapse(
                excel_path,
                fast=fast,
                constrain_to_center_flush_region=constrain,
                output_image_path=image_path,
            )
            print("\n--- Analysis Result ---")
            print(f"Minimum edge removals to collapse: {result.minimum_degradations}")
            if result.removal_order:
                print("Removal order (edge IDs):", ", ".join(str(e) for e in result.removal_order))
            else:
                print("Removal order (edge IDs): []")
            print("Metrics:")
            for k, v in result.collapse_metrics.items():
                print(f"  - {k}: {v}")
            print("\nDone.\n")
    except Exception as ex:
        print(f">>> Fatal error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()


