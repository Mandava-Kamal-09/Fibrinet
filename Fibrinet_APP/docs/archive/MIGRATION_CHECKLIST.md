# Migration Checklist - Fibrinet_APP

**Generated:** 2026-01-19
**Purpose:** Step-by-step execution guide with verification after each step

---

## Pre-Migration Setup

```bash
# Navigate to project root
cd C:/Users/manda/Documents/UCO/Fibrinet-main/Fibrinet_APP

# Create backup branch
git checkout -b cleanup/release-structure

# Verify tests pass before starting
pytest test/ tests/ projects/single_fiber/tests/ -v --tb=short
```

**Checkpoint:** All tests should pass before proceeding.

---

## Phase 1: Preparation (Low Risk)

### 1.1 Create New Directory Structure

```bash
# Create documentation directories
mkdir -p docs/guides
mkdir -p docs/reference
mkdir -p docs/archive

# Create scripts directories
mkdir -p scripts/diagnostics
mkdir -p scripts/research
mkdir -p scripts/validation

# Create data directory
mkdir -p data/test_networks

# Create examples directory
mkdir -p examples
```

### 1.2 Update .gitignore

Add the following to `.gitignore`:

```bash
# Output directories
exports/
output/
test_output/
release_figures/
validation_results/

# Generated data files at root
/*.csv
/*.png

# IDE/Editor specific
.claude/
.vscode/
.idea/

# Python artifacts
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/

# Virtual environment
.venv/
venv/
```

### 1.3 Verification

```bash
# Check directory structure created
ls -la docs/ scripts/ data/ examples/

# Commit Phase 1
git add .
git commit -m "Phase 1: Create new directory structure for cleanup"
```

---

## Phase 2: Safe Deletions (Low Risk)

### 2.1 Delete Obsolete Directory

```bash
# Remove obsolete Single_Fibrinet directory (only has README pointing elsewhere)
rm -rf Single_Fibrinet/
```

### 2.2 Delete Duplicate/Backup Files

```bash
# Delete duplicate test file
rm "test/input_data/Hangman - Copy.xlsx"

# Delete backup file
rm src/views/tkinter_view/research_simulation_page.py.backup

# Delete Windows artifact
rm -f nul

# Delete temp files
rm -f "test/input_data/4F83D810"
rm -f "test/input_data/794C1910"
```

### 2.3 Verification

```bash
# Verify tests still pass
pytest test/ tests/ projects/single_fiber/tests/ -v --tb=short

# Commit Phase 2
git add .
git commit -m "Phase 2: Remove obsolete and duplicate files"
```

---

## Phase 3: Documentation Migration (Low Risk)

### 3.1 Move CLI Guides

```bash
# Move CLI guides to docs/guides
mv "Guide for CLI/CLI.md" docs/guides/
mv "Guide for CLI/Collapse_Analyzer.md" docs/guides/

# Remove empty directory
rmdir "Guide for CLI"
```

### 3.2 Move Useful Root Docs to docs/

```bash
# Keep useful docs in appropriate location
mv TESTING_GUIDE.md docs/guides/ 2>/dev/null || true
mv USER_GUIDE_CORE_V2.md docs/guides/ 2>/dev/null || true
mv QUICK_REFERENCE_FORMULA_SHEET.md docs/reference/ 2>/dev/null || true
mv ROADMAP_2026.md docs/reference/ 2>/dev/null || true
```

### 3.3 Archive Historical Docs

```bash
# Move implementation summaries to archive
mv readme/IMPLEMENTATION_SUMMARY_*.md docs/archive/ 2>/dev/null || true
mv readme/test_binding_simple.py docs/archive/ 2>/dev/null || true
rmdir readme 2>/dev/null || true

# Archive other historical docs
mv COMPLETE_CODEBASE_DOCUMENTATION.md docs/archive/ 2>/dev/null || true
mv CONNECTIVITY_AND_TRACKING_FEATURES.md docs/archive/ 2>/dev/null || true
mv CORE_V2_INTEGRATION_STATUS.md docs/archive/ 2>/dev/null || true
mv CRITICAL_FIXES_DETERMINISM_AND_BFS.md docs/archive/ 2>/dev/null || true
mv CRITICAL_FIXES_SUMMARY.md docs/archive/ 2>/dev/null || true
mv EDGE_CASE_CRASH_ANALYSIS.md docs/archive/ 2>/dev/null || true
mv FIBRINET_CORE_V2_COMPLETE_DOCUMENTATION.md docs/archive/ 2>/dev/null || true
mv FIBRINET_FUNCTIONAL_DOCUMENTATION.md docs/archive/ 2>/dev/null || true
mv FIBRINET_TECHNICAL_DOCUMENTATION.md docs/archive/ 2>/dev/null || true
mv FIXES_VERIFICATION.md docs/archive/ 2>/dev/null || true
mv IMPLEMENTATION_COMPLETE.md docs/archive/ 2>/dev/null || true
mv LAUNCH_CHECKLIST.md docs/archive/ 2>/dev/null || true
mv MODEL_VALIDATION.md docs/archive/ 2>/dev/null || true
mv READY_FOR_TRANSPLANT.md docs/archive/ 2>/dev/null || true
mv RELAXED_NETWORK_IMPLEMENTATION_SUMMARY.md docs/archive/ 2>/dev/null || true
mv REPRODUCIBILITY.md docs/archive/ 2>/dev/null || true
mv RIGOROUS_SANITY_CHECK.md docs/archive/ 2>/dev/null || true
mv SANITY_CHECK_SUMMARY.md docs/archive/ 2>/dev/null || true
mv "Theoritical Explanation and justification.md" docs/archive/ 2>/dev/null || true
```

### 3.4 Verification

```bash
# Check docs structure
ls -la docs/guides/
ls -la docs/reference/
ls -la docs/archive/

# Commit Phase 3
git add .
git commit -m "Phase 3: Organize documentation into docs/"
```

---

## Phase 4: Script Organization (Low Risk)

### 4.1 Move Diagnostic Scripts

```bash
# Move diagnostic scripts
mv diagnostic_fiber_strains.py scripts/diagnostics/ 2>/dev/null || true
mv diagnostic_strain_coupling.py scripts/diagnostics/ 2>/dev/null || true
mv diagnostic_fiber_strain_distribution.py scripts/diagnostics/ 2>/dev/null || true
mv diagnostic_unit_scaling.py scripts/diagnostics/ 2>/dev/null || true
mv diagnostic_strain_sweep_boosted.py scripts/diagnostics/ 2>/dev/null || true
mv diagnostic_node_movement.py scripts/diagnostics/ 2>/dev/null || true
```

### 4.2 Move Research Scripts

```bash
# Move research/generation scripts
mv generate_release_figures.py scripts/research/ 2>/dev/null || true
mv generate_winner_poster_plots.py scripts/research/ 2>/dev/null || true
mv generate_mechanochemical_coupling_figures.py scripts/research/ 2>/dev/null || true
mv generate_clean_golden_curve.py scripts/research/ 2>/dev/null || true
mv run_strain_sweep.py scripts/research/ 2>/dev/null || true
mv run_boosted_strain_sweep.py scripts/research/ 2>/dev/null || true
mv run_gentle_strain_sweep.py scripts/research/ 2>/dev/null || true
mv run_ultra_gentle_strain_sweep.py scripts/research/ 2>/dev/null || true
mv run_production_clearance_study.py scripts/research/ 2>/dev/null || true
mv prove_mechanochemical_coupling.py scripts/research/ 2>/dev/null || true
```

### 4.3 Move Validation Scripts

```bash
# Move validation scripts
mv validate_release_ready.py scripts/validation/ 2>/dev/null || true
mv test_core_v2_integration.py scripts/validation/ 2>/dev/null || true
mv test_reproducibility.py scripts/validation/ 2>/dev/null || true
mv test_manual_batch.py scripts/validation/ 2>/dev/null || true
mv test_metadata_export.py scripts/validation/ 2>/dev/null || true
mv test_wlc_force.py scripts/validation/ 2>/dev/null || true
mv test_strain_0p3.py scripts/validation/ 2>/dev/null || true
```

### 4.4 Move Template/Example Code

```bash
# Move template to examples
mv GUI_INTEGRATION_TEMPLATE.py examples/ 2>/dev/null || true
```

### 4.5 Verification

```bash
# Check scripts structure
ls -la scripts/diagnostics/
ls -la scripts/research/
ls -la scripts/validation/
ls -la examples/

# Verify core tests still pass (script moves shouldn't affect them)
pytest test/ tests/ projects/single_fiber/tests/ -v --tb=short

# Commit Phase 4
git add .
git commit -m "Phase 4: Organize scripts into scripts/ directory"
```

---

## Phase 5: Test Consolidation (Medium Risk)

### 5.1 Create Test Subdirectories

```bash
mkdir -p tests/network
mkdir -p tests/force_laws
```

### 5.2 Move Test Files

```bash
# Move network tests
mv test/*.py tests/network/ 2>/dev/null || true

# Move force law tests
mv tests/test_force_laws.py tests/force_laws/ 2>/dev/null || true
```

### 5.3 Update Test Configuration

Create/update `pytest.ini` or `pyproject.toml` with test paths:

```ini
# pytest.ini
[pytest]
testpaths = tests projects/single_fiber/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### 5.4 Update Test Imports (if needed)

Check if any tests have hardcoded paths:

```bash
grep -r "test/input_data" tests/network/
```

If found, update paths to use relative paths or fixtures.

### 5.5 Verification

```bash
# Run tests with new structure
pytest tests/ projects/single_fiber/tests/ -v --tb=short

# If tests fail, check import paths
# Commit Phase 5
git add .
git commit -m "Phase 5: Consolidate tests into tests/ directory"
```

---

## Phase 6: Data Organization (Medium Risk)

### 6.1 Move Test Data

```bash
# Move test networks
mv test/input_data/*.xlsx data/test_networks/ 2>/dev/null || true
mv test/input_data/synthetic_research_network data/test_networks/ 2>/dev/null || true
mv test/input_data/*.csv data/test_networks/ 2>/dev/null || true

# Clean up old directory
rmdir test/input_data/Output 2>/dev/null || true
rmdir test/input_data 2>/dev/null || true
rmdir test 2>/dev/null || true
```

### 6.2 Update Test File Paths

Search for hardcoded paths and update them:

```bash
# Find files with old paths
grep -r "test/input_data" tests/
grep -r "test\\\\input_data" tests/
```

Update any found references to use `data/test_networks/` instead.

### 6.3 Verification

```bash
# Run tests to verify data paths work
pytest tests/ projects/single_fiber/tests/ -v --tb=short

# Commit Phase 6
git add .
git commit -m "Phase 6: Move test data to data/test_networks/"
```

---

## Phase 7: Single Fiber Promotion (Medium Risk)

### 7.1 Move Single Fiber to Top Level

```bash
# Move single fiber project
mv projects/single_fiber single_fiber

# Remove empty projects directory
rmdir projects 2>/dev/null || true
```

### 7.2 Update Import Paths

Update imports in single_fiber code from:
```python
from projects.single_fiber.src.single_fiber import ...
```
to:
```python
from single_fiber.src.single_fiber import ...
```

Files to check:
- `single_fiber/benchmarks/benchmark_performance.py`
- `single_fiber/tests/*.py`

### 7.3 Update Entry Point Commands

Update documentation to reflect new paths:
```bash
# Old
python -m projects.single_fiber.src.single_fiber.cli

# New
python -m single_fiber.src.single_fiber.cli
```

### 7.4 Verification

```bash
# Run single fiber tests
pytest single_fiber/tests/ -v --tb=short

# Test CLI
python -m single_fiber.src.single_fiber.cli --help

# Test GUI (interactive)
python -m single_fiber.src.single_fiber.gui_cli

# Commit Phase 7
git add .
git commit -m "Phase 7: Promote single_fiber to top-level directory"
```

---

## Phase 8: Network Code Rename (HIGH RISK - OPTIONAL)

> **WARNING:** This phase involves changing imports in 76+ files. Only proceed if publishing as a public package.

### 8.1 Rename src/ to fibrinet_network/

```bash
mv src fibrinet_network
```

### 8.2 Update ALL Import Statements

Find and replace in all files:
- `from src.` → `from fibrinet_network.`
- `import src.` → `import fibrinet_network.`

Files to update (76 files):
```bash
grep -rl "from src\." --include="*.py" | xargs sed -i 's/from src\./from fibrinet_network./g'
grep -rl "import src\." --include="*.py" | xargs sed -i 's/import src\./import fibrinet_network./g'
```

### 8.3 Update Entry Points

Update `FibriNet.py`, `cli_main.py`, `analyze_collapse_cli.py` with new import paths.

### 8.4 Verification

```bash
# Run ALL tests
pytest tests/ single_fiber/tests/ -v --tb=short

# Test main GUI
python FibriNet.py

# Test CLI
python cli_main.py --help

# Test collapse analysis
python analyze_collapse_cli.py --help

# Commit Phase 8
git add .
git commit -m "Phase 8: Rename src/ to fibrinet_network/"
```

---

## Post-Migration Verification

### Full Test Suite

```bash
# Run all tests
pytest tests/ single_fiber/tests/ -v

# Expected: All tests pass
```

### Entry Point Tests

```bash
# Test main GUI (should open window)
python FibriNet.py &
sleep 3
kill %1

# Test CLI help
python cli_main.py --help

# Test collapse CLI help
python analyze_collapse_cli.py --help

# Test single fiber CLI help
python -m single_fiber.src.single_fiber.cli --help
```

### Import Verification

```bash
# Verify no broken imports
python -c "from fibrinet_network.controllers.system_controller import SystemController; print('OK')"
python -c "from single_fiber.src.single_fiber.config import SimulationConfig; print('OK')"
```

---

## Rollback Instructions

If any phase fails:

```bash
# Discard all changes and return to main
git checkout main
git branch -D cleanup/release-structure

# Or reset to specific commit
git log --oneline  # Find commit to reset to
git reset --hard <commit-hash>
```

---

## Final Merge

After all verification passes:

```bash
# Switch to main
git checkout main

# Merge cleanup branch
git merge cleanup/release-structure

# Push to remote
git push origin main

# Delete cleanup branch
git branch -d cleanup/release-structure
```

---

## Summary Checklist

- [ ] Phase 1: Create directory structure
- [ ] Phase 2: Delete obsolete files
- [ ] Phase 3: Organize documentation
- [ ] Phase 4: Organize scripts
- [ ] Phase 5: Consolidate tests
- [ ] Phase 6: Organize data
- [ ] Phase 7: Promote single_fiber
- [ ] Phase 8: Rename src/ (OPTIONAL)
- [ ] Final verification
- [ ] Merge to main
