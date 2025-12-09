# 🎯 Action Plan: PR #183 - Update compat and more

**Date**: 2025-12-08  
**PR**: #183 by @ocots | **Branch**: `180-dev-up-compat` → `main`  
**State**: DRAFT | **Linked Issue**: #180

---

## 📋 Overview

**Issue Summary**: Update CTBase/CTModels compat versions, add the `@init` macro for initial guess handling (ported from CTSolvers), and rewrite `docs/make.jl` following CTModels patterns.

**PR Summary**: Currently contains only compat updates (CTBase 0.16→0.17, NLPModels 0.21→0.22). The initial guess macro, tests, and documentation updates are not yet implemented.

**Status**: Draft - Work in progress (~20% complete)

---

## 🎯 Gap Analysis

### ✅ Completed Requirements

- ✓ **CTBase compat updated in `Project.toml`** - Changed from `0.16` to `0.17`
- ✓ **CTBase compat updated in `test/Project.toml`** - Changed from `0.16` to `0.17`
- ✓ **NLPModels compat updated** - Changed from `0.21` to `0.22`

### ❌ Missing Requirements

- ✗ **CTModels compat update** - Still `0.6`, should be `0.7` in `test/Project.toml`
- ✗ **Initial guess macro** - `src/initial_guess.jl` not created
- ✗ **Initial guess tests** - `test/test_initial_guess.jl` not created
- ✗ **Documentation rewrite** - `docs/make.jl` not updated to CTModels pattern
- ✗ **CTBase in docs deps** - Not added to `docs/Project.toml`

### ➕ Additional Work Done

- Minor: `version` and `authors` lines reordered in `Project.toml` (cosmetic)

---

## 🧪 Test Status

**Overall**: ❌ 7/14 checks failing

**Details**:

- CI breakage tests failing with OptimalControl
- Likely due to compat changes propagating through ecosystem
- No new tests added yet (initial guess tests pending)

---

## 📝 Review Feedback

**Reviews**: Requested from @jbcaillau, no reviews yet

**Unresolved comments**: None (only automated breakage test bot comment)

---

## 🔧 Code Quality Assessment

**Current PR changes**:

- ✅ Compat updates are correct format
- ⚠️ CTModels compat not updated (0.6 → 0.7 missing)

**Pending work quality** (to be implemented):

- Initial guess macro: Will need docstrings, exports, type annotations
- Tests: Must replace `CTSolvers.` with `CTModels.` qualifications
- Documentation: Must use `CTBase.automatic_reference_documentation`

---

## 📋 Proposed Action Plan

### 🔴 Critical Priority (blocking merge)

1. **Update CTModels compat to 0.7**
   - Why: Issue requirement, CTModels v0.7.0 is latest
   - Where: `test/Project.toml` line 19
   - Estimated effort: Small (1 line change)
   - Details: Change `CTModels = "0.6"` to `CTModels = "0.7"`

2. **Add initial guess macro**
   - Why: Core feature requested in issue #180
   - Where: Create `src/initial_guess.jl`
   - Estimated effort: Medium
   - Details:
     - Port from [CTSolvers initial_guess.jl](https://github.com/control-toolbox/CTSolvers.jl/blob/main/src/ctparser/initial_guess.jl)
     - Change `__default_init_prefix()` from `:CTSolvers` to `:CTModels`
     - Include file in `src/CTParser.jl`
     - Export `@init` macro and helper functions

3. **Add initial guess tests**
   - Why: Tests required for new functionality
   - Where: Create `test/test_initial_guess.jl`
   - Estimated effort: Medium
   - Details:
     - Port from [CTSolvers test file](https://github.com/control-toolbox/CTSolvers.jl/blob/main/test/ctparser/test_ctparser_initial_guess_macro.jl)
     - Replace all `CTSolvers.` with `CTModels.`
     - Add include to `test/runtests.jl`

### 🟡 High Priority (should do before merge)

4. **Rewrite docs/make.jl**
   - Why: Issue requirement for documentation consistency
   - Where: `docs/make.jl`
   - Estimated effort: Medium
   - Details:
     - Follow [CTModels make.jl](https://github.com/control-toolbox/CTModels.jl/blob/main/docs/make.jl) pattern
     - Use `CTBase.automatic_reference_documentation`
     - Add helper functions for paths
     - Add `EXCLUDE_SYMBOLS` list

5. **Add CTBase to docs dependencies**
   - Why: Required for `CTBase.automatic_reference_documentation`
   - Where: `docs/Project.toml`
   - Estimated effort: Small
   - Details: Add `CTBase = "54762871-cc72-4466-b8e8-f6c8b58076cd"` to deps and compat

### 🟢 Medium Priority (nice to have)

6. **Verify no breaking changes**
   - Why: Issue mentions "this must be checked"
   - Where: Full test suite
   - Estimated effort: Small
   - Details: Run `julia --project=. -e 'using Pkg; Pkg.test()'` after all changes

7. **Add docstrings to initial_guess.jl**
   - Why: Julia best practices
   - Where: `src/initial_guess.jl`
   - Estimated effort: Small
   - Details: Ensure `@init` macro and public functions have docstrings

### 🔵 Low Priority (can defer)

8. **Fix CI breakage tests**
   - Why: Currently 7/14 failing
   - Where: CI configuration or upstream packages
   - Estimated effort: Unknown (may resolve after full implementation)
   - Details: May need coordination with OptimalControl package

---

## 📊 Summary

| Priority | Count | Effort |
|----------|-------|--------|
| 🔴 Critical | 3 | Medium |
| 🟡 High | 2 | Medium |
| 🟢 Medium | 2 | Small |
| 🔵 Low | 1 | Unknown |

**Recommended order of execution**:

1. Update CTModels compat (Critical, 1 min)
2. Add `src/initial_guess.jl` (Critical, 30 min)
3. Add `test/test_initial_guess.jl` (Critical, 20 min)
4. Add CTBase to docs deps (High, 2 min)
5. Rewrite `docs/make.jl` (High, 30 min)
6. Run tests to verify (Medium, 5 min)
7. Add docstrings if missing (Medium, 10 min)

**Total estimated effort**: ~2 hours

---

**Next step**: Start with CTModels compat update, then port initial_guess.jl from CTSolvers.
