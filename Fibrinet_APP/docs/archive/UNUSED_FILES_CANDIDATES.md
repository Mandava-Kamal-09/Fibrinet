# Unused Files Candidates - Fibrinet_APP

**Generated:** 2026-01-19
**Purpose:** Identify files that may be candidates for removal or archival

---

## Evidence Legend

- **GREP_ZERO**: No imports or references found via grep
- **UNTRACKED**: File is not tracked in git
- **DUPLICATE**: File appears to be a duplicate
- **OBSOLETE**: Directory/file is obsolete based on README or content
- **OUTPUT_ARTIFACT**: Generated output that should be gitignored
- **BACKUP_FILE**: Backup copy of another file
- **DEVELOPMENT_ONLY**: Used only during development, not for production

---

## 1. Obsolete Directories

| Directory | Evidence | Action |
|-----------|----------|--------|
| `Single_Fibrinet/` | OBSOLETE - Contains only README.md pointing to `projects/single_fiber/` | **DELETE** - Superseded by projects/single_fiber |

---

## 2. Duplicate/Backup Files

| File | Evidence | Action |
|------|----------|--------|
| `test/input_data/Hangman - Copy.xlsx` | DUPLICATE of `Hangman.xlsx` | **DELETE** |
| `src/views/tkinter_view/research_simulation_page.py.backup` | BACKUP_FILE | **DELETE** |

---

## 3. Output Artifacts (Should be Gitignored)

| File/Directory | Evidence | Action |
|----------------|----------|--------|
| `exports/fibrin_network_big_collapse_20251217_081605/` | OUTPUT_ARTIFACT - timestamped export | **GITIGNORE** |
| `output/` | OUTPUT_ARTIFACT - CLI output | **GITIGNORE** |
| `test_output/` | OUTPUT_ARTIFACT - test output | **GITIGNORE** |
| `generated_figures/` | OUTPUT_ARTIFACT - generated figures | **GITIGNORE** (or archive separately) |
| `validation_results/` | OUTPUT_ARTIFACT - validation output | **GITIGNORE** |
| `boosted_strain_sweep_results.csv` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `gentle_strain_sweep_results.csv` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `hangman_gentle_sweep_results.csv` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `ultra_gentle_sweep_results.csv` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `fibrin_network_big_strain_sweep_results.csv` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `plot1_golden_curve.png` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `ultra_gentle_strain_protection_curve.png` | OUTPUT_ARTIFACT | **GITIGNORE** |
| `nul` | OUTPUT_ARTIFACT - Windows null device | **DELETE** |
| `test/input_data/Output/` | OUTPUT_ARTIFACT - test output | **GITIGNORE** |
| `test/input_data/4F83D810` | OUTPUT_ARTIFACT - temp file | **DELETE** |
| `test/input_data/794C1910` | OUTPUT_ARTIFACT - temp file | **DELETE** |

---

## 4. Diagnostic Scripts (Development Only)

These scripts are development/debugging tools, not production code. Consider moving to a `dev/` or `scripts/diagnostics/` directory.

| File | Evidence | Action |
|------|----------|--------|
| `diagnostic_fiber_strains.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |
| `diagnostic_strain_coupling.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |
| `diagnostic_fiber_strain_distribution.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |
| `diagnostic_unit_scaling.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |
| `diagnostic_strain_sweep_boosted.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |
| `diagnostic_node_movement.py` | UNTRACKED, DEVELOPMENT_ONLY | **MOVE** to `scripts/diagnostics/` |

---

## 5. Research/Generation Scripts (Consider Archiving)

These are one-off research scripts. Consider moving to a `scripts/research/` directory or archiving.

| File | Evidence | Action |
|------|----------|--------|
| `generate_generated_figures.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `generate_winner_poster_plots.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `generate_mechanochemical_coupling_figures.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `generate_clean_golden_curve.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `run_strain_sweep.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `run_boosted_strain_sweep.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `run_gentle_strain_sweep.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `run_ultra_gentle_strain_sweep.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `run_production_clearance_study.py` | UNTRACKED | **MOVE** to `scripts/research/` |
| `prove_mechanochemical_coupling.py` | UNTRACKED | **MOVE** to `scripts/research/` |

---

## 6. Validation/Test Scripts (Consider Consolidating)

These root-level test scripts should be moved into proper test directories.

| File | Evidence | Action |
|------|----------|--------|
| `test_core_v2_integration.py` | UNTRACKED | **MOVE** to `test/` |
| `test_reproducibility.py` | UNTRACKED | **MOVE** to `test/` |
| `test_manual_batch.py` | UNTRACKED | **MOVE** to `test/` or DELETE if obsolete |
| `test_metadata_export.py` | UNTRACKED | **MOVE** to `test/` |
| `test_wlc_force.py` | UNTRACKED | **MOVE** to `tests/` (with force law tests) |
| `test_strain_0p3.py` | UNTRACKED | **MOVE** to `test/` |
| `validate_generated_ready.py` | UNTRACKED | **MOVE** to `scripts/validation/` |
| `readme/test_binding_simple.py` | GREP_ZERO - orphaned in readme/ | **MOVE** to `test/` or DELETE |

---

## 7. Documentation Candidates for Consolidation

Many `.md` files at root level appear to be historical or redundant. Consider archiving.

| File | Evidence | Action |
|------|----------|--------|
| `COMPLETE_CODEBASE_DOCUMENTATION.md` | UNTRACKED - large doc | **ARCHIVE** to `docs/archive/` |
| `CONNECTIVITY_AND_TRACKING_FEATURES.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `CORE_V2_INTEGRATION_STATUS.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `CRITICAL_FIXES_DETERMINISM_AND_BFS.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `CRITICAL_FIXES_SUMMARY.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `EDGE_CASE_CRASH_ANALYSIS.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `FIBRINET_CORE_V2_COMPLETE_DOCUMENTATION.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `FIBRINET_FUNCTIONAL_DOCUMENTATION.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `FIBRINET_TECHNICAL_DOCUMENTATION.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `FIXES_VERIFICATION.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `IMPLEMENTATION_COMPLETE.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `LAUNCH_CHECKLIST.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `MODEL_VALIDATION.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `QUICK_REFERENCE_FORMULA_SHEET.md` | UNTRACKED - may be useful | **KEEP** or **MOVE** to `docs/` |
| `READY_FOR_TRANSPLANT.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `RELAXED_NETWORK_IMPLEMENTATION_SUMMARY.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `REPRODUCIBILITY.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `RIGOROUS_SANITY_CHECK.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `ROADMAP_2026.md` | UNTRACKED - may be useful | **KEEP** or **MOVE** to `docs/` |
| `SANITY_CHECK_SUMMARY.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `TESTING_GUIDE.md` | UNTRACKED - may be useful | **KEEP** or **MOVE** to `docs/` |
| `Theoritical Explanation and justification.md` | UNTRACKED | **ARCHIVE** to `docs/archive/` |
| `USER_GUIDE_CORE_V2.md` | UNTRACKED - may be useful | **KEEP** or **MOVE** to `docs/` |
| `readme/IMPLEMENTATION_SUMMARY_v5.0_*.md` (7 files) | Historical | **ARCHIVE** to `docs/archive/` |

---

## 8. Template/Example Files

| File | Evidence | Action |
|------|----------|--------|
| `GUI_INTEGRATION_TEMPLATE.py` | UNTRACKED - example code | **MOVE** to `examples/` or DELETE |

---

## 9. Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| Obsolete directories | 1 | DELETE |
| Duplicate/backup files | 2 | DELETE |
| Output artifacts | 17 | GITIGNORE |
| Diagnostic scripts | 6 | MOVE to scripts/diagnostics/ |
| Research scripts | 10 | MOVE to scripts/research/ |
| Root-level test scripts | 8 | MOVE to test/ or tests/ |
| Documentation to archive | 23+ | ARCHIVE to docs/archive/ |
| Templates | 1 | MOVE or DELETE |

---

## 10. Files to KEEP (Not Candidates for Removal)

These files are actively used and should NOT be removed:

### Core Entry Points
- `FibriNet.py` - Main GUI
- `cli_main.py` - Main CLI
- `analyze_collapse_cli.py` - Collapse analysis

### Core Source Code
- All files in `src/` (62 files)
- All files in `projects/single_fiber/src/` (28 files)
- All files in `projects/single_fiber/benchmarks/` (2 files)

### Tests
- All files in `test/` (21 files)
- All files in `tests/` (1 file)
- All files in `projects/single_fiber/tests/` (24 files)

### Configuration
- `requirements.txt`
- `pyproject.toml` (if exists)
- `.gitignore`

### Documentation (Tracked)
- `README.md`
- `Guide for CLI/` directory
- `projects/single_fiber/README.md`
- `projects/single_fiber/ROOKIE_GUIDE.md`

### Test Data
- `test/input_data/*.xlsx` (except duplicates)
- `test/input_data/synthetic_research_network/`

### Project Resources
- `src/views/tkinter_view/images/` (GUI icons)
- `projects/single_fiber/examples/` (YAML configs)
- `projects/single_fiber/protocols/` (research protocols)
