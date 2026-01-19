# Cleanup Plan - Fibrinet_APP

**Generated:** 2026-01-19
**Purpose:** Define release-ready directory structure and migration strategy

---

## 1. Design Goals

1. **Clear separation** between network code (FibriNet) and single fiber code
2. **Release-ready** structure suitable for GitHub release
3. **Preserve all working code** - no breaking changes
4. **Gitignore output artifacts** to keep repo clean
5. **Archive historical docs** without deleting

---

## 2. Proposed Directory Structure

```
Fibrinet_APP/
│
├── README.md                      # Main README (updated)
├── LICENSE                        # License file (add if missing)
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Modern Python packaging (optional)
├── .gitignore                     # Updated gitignore
│
├── fibrinet_network/              # RENAMED from src/ - Network simulation
│   ├── __init__.py                # Package init
│   ├── config/                    # Feature flags
│   ├── controllers/               # MVC controllers
│   ├── core/                      # Core physics
│   │   └── force_laws/            # Shared force laws
│   ├── managers/                  # Business logic
│   │   ├── export/
│   │   ├── input/
│   │   ├── network/
│   │   └── view/
│   ├── models/                    # Data models
│   └── views/                     # UI (CLI, Tkinter)
│       ├── cli_view/
│       └── tkinter_view/
│
├── single_fiber/                  # MOVED from projects/single_fiber/
│   ├── README.md
│   ├── ROOKIE_GUIDE.md
│   ├── src/                       # Core library
│   │   └── single_fiber/
│   │       ├── enzyme_models/
│   │       └── gui/
│   ├── tests/                     # Unit tests
│   ├── benchmarks/                # Performance tests
│   ├── examples/                  # YAML configs
│   └── protocols/                 # Research protocols
│
├── tests/                         # CONSOLIDATED test directory
│   ├── network/                   # Network tests (from test/)
│   ├── force_laws/                # Force law tests (from tests/)
│   └── conftest.py                # Shared pytest fixtures
│
├── scripts/                       # NEW - Organized scripts
│   ├── diagnostics/               # Development/debugging scripts
│   ├── research/                  # One-off research scripts
│   └── validation/                # Validation scripts
│
├── docs/                          # NEW - Documentation hub
│   ├── guides/                    # User guides
│   │   ├── CLI.md
│   │   ├── Collapse_Analyzer.md
│   │   ├── TESTING_GUIDE.md
│   │   └── USER_GUIDE_CORE_V2.md
│   ├── reference/                 # Technical reference
│   │   ├── QUICK_REFERENCE_FORMULA_SHEET.md
│   │   └── ROADMAP_2026.md
│   └── archive/                   # Historical docs (not deleted)
│       ├── IMPLEMENTATION_SUMMARY_*.md
│       └── (other historical docs)
│
├── examples/                      # NEW - Example code
│   └── GUI_INTEGRATION_TEMPLATE.py
│
├── data/                          # NEW - Test/sample data
│   └── test_networks/             # Moved from test/input_data
│       ├── TestNetwork.xlsx
│       ├── Hangman.xlsx
│       └── synthetic_research_network/
│
├── FibriNet.py                    # Main GUI entry point (update imports)
├── cli_main.py                    # CLI entry point (update imports)
└── analyze_collapse_cli.py        # Collapse CLI (update imports)
```

---

## 3. Key Changes Summary

### Renames
| Current | Proposed | Rationale |
|---------|----------|-----------|
| `src/` | `fibrinet_network/` | Clear naming, distinguishes from single_fiber |
| `test/` | `tests/network/` | Consolidated test structure |
| `tests/` | `tests/force_laws/` | Consolidated test structure |

### Moves
| Current | Proposed | Rationale |
|---------|----------|-----------|
| `projects/single_fiber/` | `single_fiber/` | Promote to top-level |
| `Guide for CLI/` | `docs/guides/` | Organized documentation |
| `readme/` | `docs/archive/` | Historical docs |
| `test/input_data/` | `data/test_networks/` | Clear data organization |
| Root diagnostic scripts | `scripts/diagnostics/` | Organized scripts |
| Root research scripts | `scripts/research/` | Organized scripts |
| Root validation scripts | `scripts/validation/` | Organized scripts |

### Deletions (Safe)
| File | Reason |
|------|--------|
| `Single_Fibrinet/` | Obsolete - only contains outdated README |
| `test/input_data/Hangman - Copy.xlsx` | Duplicate |
| `src/views/tkinter_view/research_simulation_page.py.backup` | Backup file |
| `nul` | Windows artifact |
| `test/input_data/4F83D810` | Temp file |
| `test/input_data/794C1910` | Temp file |

### Gitignore Additions
```gitignore
# Output directories
exports/
output/
test_output/
release_figures/
validation_results/

# Generated files
*.csv
!data/**/*.csv
*.png
!src/views/tkinter_view/images/*.png

# IDE/Editor
.claude/
.vscode/
.idea/

# Python
__pycache__/
*.pyc
.pytest_cache/
.venv/
```

---

## 4. Migration Strategy

### Phase 1: Preparation (No Code Changes)
1. Create new directories: `docs/`, `scripts/`, `data/`, `examples/`
2. Update `.gitignore` with output artifacts
3. Create this cleanup plan for review

### Phase 2: Safe Deletions
1. Delete obsolete/duplicate files (see Deletions table)
2. Verify tests still pass

### Phase 3: Documentation Migration
1. Move `Guide for CLI/*.md` to `docs/guides/`
2. Move useful root docs to `docs/guides/` or `docs/reference/`
3. Move `readme/` contents to `docs/archive/`
4. Move historical root docs to `docs/archive/`

### Phase 4: Script Organization
1. Create `scripts/diagnostics/`, `scripts/research/`, `scripts/validation/`
2. Move diagnostic scripts
3. Move research scripts
4. Move validation scripts
5. Update any hardcoded paths in scripts

### Phase 5: Test Consolidation
1. Create `tests/network/` and `tests/force_laws/`
2. Move `test/*.py` to `tests/network/`
3. Move `tests/test_force_laws.py` to `tests/force_laws/`
4. Move root-level test scripts to appropriate directories
5. Update pytest configuration
6. Verify all tests pass

### Phase 6: Data Organization
1. Create `data/test_networks/`
2. Move `test/input_data/*.xlsx` to `data/test_networks/`
3. Move `test/input_data/synthetic_research_network/` to `data/test_networks/`
4. Update test file paths
5. Verify tests pass

### Phase 7: Single Fiber Promotion
1. Move `projects/single_fiber/` to `single_fiber/`
2. Update import paths in single_fiber code
3. Update entry points
4. Verify single fiber tests pass

### Phase 8: Network Code Rename (OPTIONAL - High Risk)
1. Rename `src/` to `fibrinet_network/`
2. Update ALL imports across codebase
3. Update entry points
4. Verify all tests pass

> **Note:** Phase 8 is optional and carries the highest risk. The current `src/` naming works fine for an internal project. Only do this if publishing as a public package.

---

## 5. Risk Assessment

| Phase | Risk Level | Mitigation |
|-------|------------|------------|
| 1. Preparation | Low | No code changes |
| 2. Safe Deletions | Low | Only removes duplicates/obsolete |
| 3. Documentation | Low | Only moves .md files |
| 4. Script Organization | Low | Scripts are standalone |
| 5. Test Consolidation | Medium | Update paths carefully, run tests |
| 6. Data Organization | Medium | Update paths carefully, run tests |
| 7. Single Fiber Promotion | Medium | Update imports carefully |
| 8. Network Rename | **High** | 76 files with `from src.` imports |

---

## 6. Verification Checklist

After each phase:
- [ ] Run `pytest test/ tests/ projects/single_fiber/tests/ -v`
- [ ] Verify no import errors
- [ ] Test GUI: `python FibriNet.py`
- [ ] Test CLI: `python cli_main.py --help`
- [ ] Test single fiber: `python -m projects.single_fiber.src.single_fiber.cli --help`

---

## 7. Rollback Plan

Before starting migration:
1. Create a git branch: `git checkout -b cleanup/release-structure`
2. Commit after each phase
3. If issues arise: `git checkout main`

---

## 8. Timeline Estimate

| Phase | Effort |
|-------|--------|
| 1. Preparation | Quick |
| 2. Safe Deletions | Quick |
| 3. Documentation | Quick |
| 4. Script Organization | Quick |
| 5. Test Consolidation | Moderate |
| 6. Data Organization | Moderate |
| 7. Single Fiber Promotion | Moderate |
| 8. Network Rename | Significant (optional) |

---

## 9. Recommended Approach

**Minimal Cleanup (Recommended for now):**
- Execute Phases 1-4 only
- Low risk, high impact on organization
- Preserves all existing import paths

**Full Cleanup (For release):**
- Execute all phases including 5-7
- Skip Phase 8 unless publishing as pip package
- Requires careful testing

---

## 10. Files Created by This Audit

| File | Purpose |
|------|---------|
| `REPO_MAP.md` | Current repository structure inventory |
| `UNUSED_FILES_CANDIDATES.md` | Files identified for cleanup |
| `CLEANUP_PLAN.md` | This migration plan |
| `MIGRATION_CHECKLIST.md` | Step-by-step execution guide |
