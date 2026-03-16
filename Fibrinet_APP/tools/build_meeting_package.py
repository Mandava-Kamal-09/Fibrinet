"""
Build March 11 meeting figures package for Dr. Bannish.

Copies selected figures into results/meeting_march11/ with sequential naming.
If a source file does not exist, prints a warning and skips.
"""

import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE, 'results', 'meeting_march11')

COPIES = [
    ('results/figures/fig_01_critical_strain.png',
     '01_critical_strain.png'),
    ('results/figures/fig_02_three_hypotheses.png',
     '02_three_hypotheses.png'),
    ('results/validation/fig_01_validation_4panel.png',
     '03_validation_4panel.png'),
    ('results/combination_matrix/fig_combination_matrix_no_cascade.png',
     '04_matrix_no_cascade.png'),
    ('results/combination_matrix/fig_combination_matrix.png',
     '05_matrix_cascade.png'),
    ('results/figures/fig_07_single_fiber.png',
     '06_single_fiber.png'),
    ('results/figures/cascade_threshold_sweep.png',
     '07_threshold_sweep.png'),
]

README_TEXT = """\
FibriNet Meeting Package — March 11, 2026
Dr. Brittany Bannish

Figure Order:
01_critical_strain.png    — ε*=23% biphasic discovery
                            (F-test p=0.003, Cohen's d=1.07)
02_three_hypotheses.png   — Three mechanochemical coupling modes
03_validation_4panel.png  — 4/4 validation benchmarks
04_matrix_no_cascade.png  — 3×3 hypothesis matrix, pure
                            enzymatic signal (N=10 seeds)
05_matrix_cascade.png     — 3×3 hypothesis matrix, mechanical
                            cascade ON (N=10 seeds)
06_single_fiber.png       — Single fiber Lynch 2022 comparison
                            (3/3 PASS, 9% error at 0% strain)
07_threshold_sweep.png    — Cascade threshold calibration
                            (best R²=−0.7416 at threshold=0.30)

Current State (March 8, 2026):
- Tests: 73/73 passing
- Validation: 4/4 benchmarks passing
- k_cat dynamic updating: COMPLETE (C1 gap closed)
- Spatial prestrain: implemented, amplitude calibration pending
- 3×3 matrix: 2160 runs complete, all 9 combinations
  F-test p=0.000000
- ε* = 23.3% (p=0.003, d=1.07)
- Novelty score: 8.75/10
"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    copied = 0
    warnings = []

    for src_rel, dst_name in COPIES:
        src = os.path.join(BASE, src_rel)
        dst = os.path.join(OUTDIR, dst_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  COPIED  {src_rel}  ->  {dst_name}")
            copied += 1
        else:
            msg = f"  WARNING: source not found, skipping: {src_rel}"
            print(msg)
            warnings.append(src_rel)

    # Write README
    readme_path = os.path.join(OUTDIR, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(README_TEXT)
    print(f"  WROTE   README.txt")

    print()
    print(f"  Summary: {copied}/{len(COPIES)} figures copied, "
          f"{len(warnings)} warning(s)")
    if warnings:
        for w in warnings:
            print(f"    - missing: {w}")
    print(f"  Output:  {OUTDIR}")


if __name__ == '__main__':
    main()
