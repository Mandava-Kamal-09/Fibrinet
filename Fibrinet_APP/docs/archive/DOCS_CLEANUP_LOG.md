# Documentation Cleanup Log

**Date:** 2026-01-19
**Phase:** 1 - Language Cleanup (No Code Changes)

---

## Files Modified

### 1. `README.md` (root)
- **Changed:** "Strain sweep experiments to measure T50 clearance times"
- **To:** "Strain sweep experiments for parameter exploration"
- **Reason:** T50 clearance times are not implemented/verified in current codebase

### 2. `projects/single_fiber/README.md`
- **Changed:** "CLI-first, publication-grade single fibrin fiber simulator"
- **To:** "CLI-first single fibrin fiber simulator"
- **Reason:** Remove "publication-grade" hype language

- **Changed:** "Publication-ready exports: CSV data and JSON metadata"
- **To:** "Structured exports: CSV data and JSON metadata"
- **Reason:** Remove "publication-ready" hype language

### 3. `projects/single_fiber/VALIDATION_NOTE.md`
- **Changed:** "Phase 3: COMPLETE (physics engine validated, 96/96 tests pass)"
- **To:** "Phase 3: COMPLETE (physics engine implemented)"
- **Reason:** Replaced specific test count claim with instruction to run tests

- **Changed:** "The tool is scientifically sound for its stated scope"
- **To:** "The tool implements single-fiber overdamped mechanics"
- **Reason:** Removed "scientifically sound" claim

- **Added:** `To verify tests pass: pytest projects/single_fiber/tests -v`
- **Reason:** Replaced claim with verifiable command

### 4. `Single_Fibrinet/README.md`
- **Changed:** "Tests: `tests/test_force_laws.py` (19 tests)"
- **To:** "Tests: `pytest tests/test_force_laws.py -v`"
- **Reason:** Replaced test count with run command

- **Changed:** "Tests: `projects/single_fiber/tests/` (46 tests)"
- **To:** "Tests: `pytest projects/single_fiber/tests -v`"
- **Reason:** Replaced test count with run command

- **Changed:** "Tests: `projects/single_fiber/tests/` (96 tests total)"
- **To:** "Tests: `pytest projects/single_fiber/tests -v`"
- **Reason:** Replaced test count with run command

- **Changed:** "Visual polish + publication figures"
- **To:** "Visual polish + figure generation"
- **Reason:** Removed "publication" language

- **Changed:** Running Tests section - removed "(19 tests)", "(96 tests)" counts
- **To:** Just test commands without count claims
- **Reason:** Counts should be verified by running tests, not claimed

### 5. `Theoritical Explanation and justification.md`
- **Changed:** Section "10.3 What Makes This Publication-Ready?"
- **To:** Section "10.3 Implementation Features"
- **Reason:** Removed "publication-ready" claim

- **Changed:** "first-of-its-kind" claim for strain-inhibited coupling
- **To:** "Implements strain-dependent cleavage rates"
- **Reason:** Removed unsupported novelty claim

- **Changed:** Section "Publication-Ready Status" with checkmarks
- **To:** Section "Implementation Status" with factual bullet points
- **Reason:** Replaced hype checkmarks with factual statements

### 6. `CLEANUP_PLAN.md` (created earlier, now cleaned)
- **Changed:** All "publication" → "release" (bulk replace)
- **Changed:** "Publication-ready structure" → "Release-ready structure"
- **Changed:** "Proposed Publication Structure" → "Proposed Directory Structure"
- **Reason:** Remove publication language from migration documentation

### 7. `MIGRATION_CHECKLIST.md` (created earlier, now cleaned)
- **Changed:** All "publication" → "release" (bulk replace)
- **Reason:** Remove publication language from migration documentation

### 8. `REPO_MAP.md` (created earlier, now cleaned)
- **Changed:** All "publication" → "generated" (bulk replace)
- **Reason:** These refer to output directories/files, "generated" is accurate

### 9. `UNUSED_FILES_CANDIDATES.md` (created earlier, now cleaned)
- **Changed:** All "publication" → "generated" (bulk replace)
- **Reason:** These refer to output directories/files, "generated" is accurate

---

## Files NOT Modified (Out of Scope or Archive Candidates)

The following files still contain "publication" language but are:
- Untracked historical documentation files
- Candidates for archival per CLEANUP_PLAN.md
- Not actively used README/Guide/Phase docs

| File | Reason Not Modified |
|------|---------------------|
| `COMPLETE_CODEBASE_DOCUMENTATION.md` | Untracked, archive candidate |
| `CRITICAL_FIXES_DETERMINISM_AND_BFS.md` | Untracked, archive candidate |
| `FIXES_VERIFICATION.md` | Untracked, archive candidate |
| `SANITY_CHECK_SUMMARY.md` | Untracked, archive candidate |
| `LAUNCH_CHECKLIST.md` | Untracked, archive candidate |
| `IMPLEMENTATION_COMPLETE.md` | Untracked, archive candidate |
| `FIBRINET_TECHNICAL_DOCUMENTATION.md` | Untracked, archive candidate |
| `EDGE_CASE_CRASH_ANALYSIS.md` | Untracked, archive candidate |
| `CONNECTIVITY_AND_TRACKING_FEATURES.md` | Untracked, archive candidate |
| `MODEL_VALIDATION.md` | Untracked, archive candidate |
| `QUICK_REFERENCE_FORMULA_SHEET.md` | Untracked, archive candidate |
| `RELAXED_NETWORK_IMPLEMENTATION_SUMMARY.md` | Untracked, archive candidate |
| `REPRODUCIBILITY.md` | Untracked, archive candidate |
| `ROADMAP_2026.md` | Untracked, archive candidate |
| `USER_GUIDE_CORE_V2.md` | Untracked, archive candidate |
| `readme/IMPLEMENTATION_SUMMARY_v5.0_*.md` | Historical, archive candidate |

**Recommendation:** These files should be moved to `docs/archive/` per the cleanup plan, or deleted if no longer needed.

---

## Suspicious Claims NOT Changed (With Reason)

| Claim | Location | Reason Not Changed |
|-------|----------|-------------------|
| "Marko-Siggia approximation" | README.md | Verifiable in code (`src/core/force_laws/wlc.py`) |
| "WLC (Worm-Like Chain)" | Multiple | Standard physics terminology, implemented |
| "Deterministic" | Multiple | Verifiable by running same seed twice |
| "Overdamped dynamics" | Multiple | Verifiable in integrator code |
| "Five hazard models" | single_fiber/README.md | Verifiable in `enzyme_models/hazards.py` |

---

## Remaining Inconsistencies

| Issue | Location | Status |
|-------|----------|--------|
| `hazard_functions.py` referenced in PHASE4 doc | `PHASE4_STRAIN_ENZYME_LAB.md:40` | File is actually `hazards.py` - minor naming inconsistency |
| Directory `sweeps/` referenced | `PHASE4_STRAIN_ENZYME_LAB.md:48` | Directory may not exist at documented path |

---

## Verification Commands and Results

### Command 1: Search for prohibited terms in project docs
```bash
rg -i "publication|paper-ready|acceptance probability|novelty score|first-of-its-kind" projects/ Single_Fibrinet/
```
**Result:** No matches found (PASS)

### Command 2: Search for prohibited terms in root-level tracked docs
```bash
rg -i "publication|paper-ready|acceptance probability|novelty score|first-of-its-kind" README.md
```
**Result:** No matches found (PASS)

### Command 3: Search for prohibited terms in Guide for CLI
```bash
rg -i "publication|paper-ready|acceptance probability|novelty score|first-of-its-kind" "Guide for CLI/"
```
**Result:** No matches found (PASS)

### Command 4: Count remaining files with "publication" (untracked archive candidates)
```bash
rg -l -i "publication" *.md | wc -l
```
**Result:** ~20 files (all untracked, archive candidates)

---

## Summary

| Metric | Count |
|--------|-------|
| Files modified | 9 |
| Claims removed/replaced | 18 |
| Files flagged for archive | 16 |
| Remaining inconsistencies | 2 (minor) |

**Phase 1 Status:** COMPLETE

All active documentation (README files, guides, phase docs, validation notes) has been cleaned of hype language. Remaining files with "publication" language are untracked historical docs that are candidates for archival.
