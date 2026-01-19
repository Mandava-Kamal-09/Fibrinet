# Repository Map - Fibrinet_APP

**Generated:** 2026-01-19
**Purpose:** Comprehensive inventory of repository structure for cleanup planning

---

## 1. Directory Structure Overview

```
Fibrinet_APP/
├── src/                          # NETWORK CODE - Main FibriNet library
│   ├── config/                   # Feature flags, config
│   ├── controllers/              # System controller (MVC pattern)
│   ├── core/                     # Core physics engine (v2)
│   │   └── force_laws/           # Shared force laws (Hookean, WLC)
│   ├── managers/                 # Business logic managers
│   │   ├── export/               # PNG/Excel export strategies
│   │   ├── input/                # Excel input parsing
│   │   ├── network/              # Network state, degradation engine
│   │   │   ├── degradation_engine/
│   │   │   ├── edges/
│   │   │   ├── networks/
│   │   │   └── nodes/
│   │   └── view/                 # View management
│   ├── models/                   # Data models (plasmin, system state)
│   └── views/                    # UI implementations
│       ├── cli_view/             # CLI interface
│       └── tkinter_view/         # Tkinter GUI
│           ├── images/           # GUI icons (20 PNG files)
│           ├── modify_page_managers/
│           └── new_network_page_managers/
│
├── projects/                     # SINGLE FIBER CODE - Separate project
│   └── single_fiber/
│       ├── benchmarks/           # Performance benchmarks
│       ├── examples/             # YAML config examples (6 files)
│       ├── protocols/            # Research protocol templates (4 files)
│       ├── src/single_fiber/     # Core library
│       │   ├── enzyme_models/    # Enzyme kinetics
│       │   └── gui/              # DearPyGui interface
│       └── tests/                # Unit tests (24 files)
│
├── test/                         # NETWORK TESTS - Test suite for network code
│   └── input_data/               # Test data files
│       ├── Output/
│       └── synthetic_research_network/
│
├── tests/                        # FORCE LAW TESTS - Shared force law tests
│
├── utils/                        # Utility scripts
│   └── logger/                   # Logging framework
│
├── exports/                      # Generated output (should be gitignored)
├── output/                       # CLI output (should be gitignored)
├── generated_figures/          # Generated figures
├── validation_results/           # Validation output
├── test_output/                  # Test output
│
├── Guide for CLI/                # CLI documentation
├── readme/                       # Implementation summaries (historical)
├── .github/                      # GitHub config
├── .claude/                      # Claude Code settings
│
├── Single_Fibrinet/              # OBSOLETE - Only contains README.md
└── .venv/                        # Python virtual environment
```

---

## 2. Entry Points (Main Scripts)

### Primary Entry Points
| File | Purpose | Status |
|------|---------|--------|
| `FibriNet.py` | Main GUI application entry | **Active** |
| `cli_main.py` | CLI entry point | **Active** |
| `analyze_collapse_cli.py` | Collapse analysis CLI | **Active** |

### Single Fiber Entry Points
| File | Purpose | Status |
|------|---------|--------|
| `projects/single_fiber/src/single_fiber/cli.py` | Single fiber CLI | **Active** |
| `projects/single_fiber/src/single_fiber/gui_cli.py` | Single fiber GUI | **Active** |
| `projects/single_fiber/benchmarks/benchmark_performance.py` | Performance benchmark | **Active** |

### Diagnostic Scripts (Root Level - UNTRACKED)
| File | Purpose | Status |
|------|---------|--------|
| `diagnostic_fiber_strains.py` | Debug fiber strains | Development |
| `diagnostic_strain_coupling.py` | Debug strain coupling | Development |
| `diagnostic_fiber_strain_distribution.py` | Debug strain distribution | Development |
| `diagnostic_unit_scaling.py` | Debug unit scaling | Development |
| `diagnostic_strain_sweep_boosted.py` | Debug boosted sweep | Development |
| `diagnostic_node_movement.py` | Debug node movement | Development |

### Generation Scripts (Root Level - UNTRACKED)
| File | Purpose | Status |
|------|---------|--------|
| `generate_generated_figures.py` | Generate pub figures | Research |
| `generate_winner_poster_plots.py` | Generate poster plots | Research |
| `generate_mechanochemical_coupling_figures.py` | Generate coupling figures | Research |
| `generate_clean_golden_curve.py` | Generate golden curve | Research |

### Run Scripts (Root Level - UNTRACKED)
| File | Purpose | Status |
|------|---------|--------|
| `run_strain_sweep.py` | Run strain sweep | Research |
| `run_boosted_strain_sweep.py` | Run boosted sweep | Research |
| `run_gentle_strain_sweep.py` | Run gentle sweep | Research |
| `run_ultra_gentle_strain_sweep.py` | Run ultra gentle sweep | Research |
| `run_production_clearance_study.py` | Run clearance study | Research |

### Validation/Test Scripts (Root Level - UNTRACKED)
| File | Purpose | Status |
|------|---------|--------|
| `validate_generated_ready.py` | Validate for generated | Research |
| `prove_mechanochemical_coupling.py` | Prove coupling | Research |
| `test_core_v2_integration.py` | Test core v2 | Testing |
| `test_reproducibility.py` | Test reproducibility | Testing |
| `test_manual_batch.py` | Manual batch test | Testing |
| `test_metadata_export.py` | Test metadata export | Testing |
| `test_wlc_force.py` | Test WLC force | Testing |
| `test_strain_0p3.py` | Test 0.3 strain | Testing |

---

## 3. Test Files Inventory

### Network Tests (`test/` - 21 files)
| File | Tests For |
|------|-----------|
| `test_phase0_feature_flags.py` | Feature flag system |
| `test_phase0_backward_compat.py` | Backward compatibility |
| `test_phase1_data_models.py` | Data models |
| `test_phase2_plasmin_manager.py` | Plasmin manager |
| `test_phase3_edge_evolution_engine.py` | Edge evolution |
| `test_phase4_deterministic_replay.py` | Deterministic replay |
| `test_phase4_export_consistency.py` | Export consistency |
| `test_phase4_failure_modes.py` | Failure modes |
| `test_phase4_scientific_invariants.py` | Scientific invariants |
| `test_phase4_1_executed_validation.py` | Executed validation |
| `test_phase4_phase5_visualization_validation.py` | Visualization |
| `phase4_phase5_validation_report.py` | Validation report |
| `test_spatial_plasmin_init.py` | Spatial plasmin init |
| `test_spatial_plasmin_units.py` | Spatial plasmin units |
| `test_spatial_plasmin_stiffness.py` | Spatial plasmin stiffness |
| `test_spatial_plasmin_binding.py` | Spatial plasmin binding |
| `test_spatial_plasmin_seeding.py` | Spatial plasmin seeding |
| `test_spatial_plasmin_cleavage.py` | Spatial plasmin cleavage |
| `test_spatial_plasmin_phase2f.py` | Spatial plasmin phase2f |
| `test_binding_integration.py` | Binding integration |
| `test_relaxed_network_physics.py` | Relaxed network physics |

### Force Law Tests (`tests/` - 1 file)
| File | Tests For |
|------|-----------|
| `test_force_laws.py` | Shared force laws (Hookean, WLC) |

### Single Fiber Tests (`projects/single_fiber/tests/` - 24 files)
| File | Tests For |
|------|-----------|
| `test_config_validation.py` | Config validation |
| `test_overdamped_equilibrium_hooke.py` | Hooke equilibrium |
| `test_overdamped_equilibrium_wlc.py` | WLC equilibrium |
| `test_rupture_behavior_wlc.py` | WLC rupture |
| `test_displacement_ramp_reproducible.py` | Displacement ramp |
| `test_export_schema.py` | Export schema |
| `test_chain_state.py` | Chain state |
| `test_chain_model.py` | Chain model |
| `test_chain_integrator.py` | Chain integrator |
| `test_chain_runner.py` | Chain runner |
| `test_frozen_physics_api.py` | Frozen physics API |
| `test_enzyme_hazards.py` | Enzyme hazards |
| `test_enzyme_metrics.py` | Enzyme metrics |
| `test_enzyme_sampler.py` | Enzyme sampler |
| `test_enzyme_registry.py` | Enzyme registry |
| `test_sweep_safety.py` | Sweep safety |
| `test_phase5_features.py` | Phase 5 features |

---

## 4. Source Code Module Count

| Directory | Python Files | Purpose |
|-----------|-------------|---------|
| `src/` | 62 files | Network simulation code |
| `projects/single_fiber/` | 47 files | Single fiber simulation code |
| `test/` | 21 files | Network tests |
| `tests/` | 1 file | Force law tests |
| `utils/` | 4 files | Utility scripts |
| Root level | ~30 files | Mixed (diagnostic, generation, validation) |

---

## 5. Documentation Files

### Root Level (UNTRACKED - Candidates for Cleanup)
| File | Purpose |
|------|---------|
| `COMPLETE_CODEBASE_DOCUMENTATION.md` | Full codebase docs |
| `CONNECTIVITY_AND_TRACKING_FEATURES.md` | Connectivity features |
| `CORE_V2_INTEGRATION_STATUS.md` | Core v2 status |
| `CRITICAL_FIXES_DETERMINISM_AND_BFS.md` | Determinism fixes |
| `CRITICAL_FIXES_SUMMARY.md` | Fix summary |
| `EDGE_CASE_CRASH_ANALYSIS.md` | Crash analysis |
| `FIBRINET_CORE_V2_COMPLETE_DOCUMENTATION.md` | Core v2 docs |
| `FIBRINET_FUNCTIONAL_DOCUMENTATION.md` | Functional docs |
| `FIBRINET_TECHNICAL_DOCUMENTATION.md` | Technical docs |
| `FIXES_VERIFICATION.md` | Fix verification |
| `IMPLEMENTATION_COMPLETE.md` | Implementation status |
| `LAUNCH_CHECKLIST.md` | Launch checklist |
| `MODEL_VALIDATION.md` | Model validation |
| `QUICK_REFERENCE_FORMULA_SHEET.md` | Formula reference |
| `READY_FOR_TRANSPLANT.md` | Transplant readiness |
| `RELAXED_NETWORK_IMPLEMENTATION_SUMMARY.md` | Relaxation summary |
| `REPRODUCIBILITY.md` | Reproducibility notes |
| `RIGOROUS_SANITY_CHECK.md` | Sanity check |
| `ROADMAP_2026.md` | 2026 roadmap |
| `SANITY_CHECK_SUMMARY.md` | Sanity check summary |
| `TESTING_GUIDE.md` | Testing guide |
| `Theoritical Explanation and justification.md` | Theory justification |
| `USER_GUIDE_CORE_V2.md` | User guide |
| `GUI_INTEGRATION_TEMPLATE.py` | GUI template (code, not doc) |

### Project Documentation (TRACKED)
| File | Purpose |
|------|---------|
| `README.md` | Main project README |
| `Guide for CLI/CLI.md` | CLI guide |
| `Guide for CLI/Collapse_Analyzer.md` | Collapse analyzer guide |
| `projects/single_fiber/README.md` | Single fiber README |
| `projects/single_fiber/ROOKIE_GUIDE.md` | Beginner guide |
| `projects/single_fiber/PHASE4_STRAIN_ENZYME_LAB.md` | Phase 4 lab doc |
| `projects/single_fiber/VALIDATION_NOTE.md` | Validation note |

### Historical Implementation Summaries (`readme/`)
| File | Phase |
|------|-------|
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE1.5.md` | Phase 1.5 |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2A.1.md` | Phase 2A.1 |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2A.2.md` | Phase 2A.2 |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2B.md` | Phase 2B |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2C.md` | Phase 2C |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2F.md` | Phase 2F |
| `IMPLEMENTATION_SUMMARY_v5.0_PHASE2G.md` | Phase 2G |

---

## 6. Data Files

### Test Input Data (`test/input_data/`)
| File | Purpose |
|------|---------|
| `TestNetwork.xlsx` | Test network data |
| `Hangman.xlsx` | Hangman network data |
| `Hangman - Copy.xlsx` | Duplicate (candidate for removal) |
| `mega_blaster.xlsx` | Large network |
| `mega_complex_20251006_114244.xlsx` | Complex network |
| `T_Shape.xlsx` | T-shaped network |
| `Realistic_simulation_Network.xlsx` | Realistic network |
| `synthetic_research_network/` | Synthetic CSV data |

### Output Artifacts (Should be gitignored)
| Directory/File | Type |
|----------------|------|
| `exports/` | Simulation exports |
| `output/` | CLI output |
| `test_output/` | Test output |
| `generated_figures/` | Generated figures |
| `validation_results/` | Validation results |
| `*.csv` (root level) | Sweep results |
| `*.png` (root level) | Generated plots |
| `nul` | Windows null device artifact |

---

## 7. Import Dependency Summary

### Files importing `from src.*`
- 76 files reference `src/` modules
- Network code is well-integrated

### Files importing `from projects.single_fiber.*`
- 12 files reference single fiber modules
- Single fiber is self-contained

### Cross-Module Dependencies
- `src/core/force_laws/` is shared between network and single fiber
- Single fiber tests import from both `projects.single_fiber` and `src.core`

---

## 8. Git Status Summary

| Category | Count |
|----------|-------|
| Tracked files | ~198 |
| Untracked files | ~107 |
| Modified files | 6 |
| Deleted files | 26 (old exports) |

### Key Observations
1. Many documentation `.md` files are untracked
2. Many diagnostic/generation scripts are untracked
3. Output directories contain artifacts that should be gitignored
4. `Single_Fibrinet/` is obsolete (only contains README)
5. Duplicate test file: `Hangman - Copy.xlsx`
6. Research simulation page has a `.backup` file
