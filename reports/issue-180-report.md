# 📊 Status Report: Issue #180 - [Dev] Update compat and more

**Date**: 2025-12-08 | **State**: Open | **Repo**: control-toolbox/CTParser.jl  
**PR**: [#183](https://github.com/control-toolbox/CTParser.jl/pull/183) (Draft) on branch `180-dev-up-compat`

---

## 📋 Summary

This issue covers three main tasks: updating dependency compatibility versions for CTBase/CTModels, adding the `@init` macro for initial guess handling from CTSolvers, and rewriting the documentation `make.jl` following CTModels patterns. Work has started on the compat updates but the initial guess macro and documentation tasks are pending.

**Created**: 2025-12-06 | **Updated**: 2025-12-08 | **Labels**: internal dev  
**Assignee**: jbcaillau

---

## 💬 Discussion

No comments on the issue. PR #183 description states:
- Update compat
- Add initial guess macro (with its tests)
- Update documentation
- Additional tests (beyond initial guess) will be added later

**CI Status**: 7/14 checks failing (breakage tests with OptimalControl)

**Key Decisions**:
- CTBase compat should be `0.17` (latest is v0.17.1)
- CTModels compat should be `0.7` (latest is v0.7.0)
- Initial guess macro must NOT depend on CTSolvers; use CTModels instead
- Documentation should follow CTModels `make.jl` pattern with `CTBase.automatic_reference_documentation`

**References**:
- [CTSolvers initial_guess.jl](https://github.com/control-toolbox/CTSolvers.jl/blob/main/src/ctparser/initial_guess.jl): Source file to port
- [CTSolvers test file](https://github.com/control-toolbox/CTSolvers.jl/blob/main/test/ctparser/test_ctparser_initial_guess_macro.jl): Tests to adapt
- [CTModels make.jl](https://github.com/control-toolbox/CTModels.jl/blob/main/docs/make.jl): Documentation pattern to follow
- [CTModels initial_guess.jl](https://github.com/control-toolbox/CTModels.jl/blob/main/src/init/initial_guess.jl): Methods to qualify with CTModels

---

## ✅ Completed

- ✓ **CTBase compat updated in `Project.toml`** - Changed from `0.16` to `0.17` (commit 33cfa45)
- ✓ **CTBase compat updated in `test/Project.toml`** - Changed from `0.16` to `0.17` (commit 33cfa45)
- ✓ **NLPModels compat updated in `test/Project.toml`** - Changed from `0.21` to `0.22` (commit 33cfa45)
- ✓ **Branch created** - `180-dev-up-compat` with initial compat updates

---

## 📝 Pending Actions

### 🔴 Critical

**Update CTModels compat in `test/Project.toml`**
- Why: Issue specifies CTModels should be updated to `0.7` (currently `0.6`)
- Where: `test/Project.toml` line 19
- Complexity: Simple

### 🟡 High

**Add `initial_guess.jl` from CTSolvers to `src/`**
- Why: Core feature requested in issue
- Where: Create `src/initial_guess.jl`
- Complexity: Moderate
- Notes: 
  - Port from [CTSolvers initial_guess.jl](https://github.com/control-toolbox/CTSolvers.jl/blob/main/src/ctparser/initial_guess.jl)
  - Change `INIT_PREFIX` default from `:CTSolvers` to `:CTModels`
  - Ensure no CTSolvers dependency

**Add initial guess tests**
- Why: Tests required for the new macro (in scope of PR #183)
- Where: Create `test/test_initial_guess.jl`
- Complexity: Moderate
- Notes:
  - Port from [CTSolvers test file](https://github.com/control-toolbox/CTSolvers.jl/blob/main/test/ctparser/test_ctparser_initial_guess_macro.jl)
  - Replace all `CTSolvers.` qualifications with `CTModels.`
  - Add to `runtests.jl`

### 🟢 Medium

**Rewrite `docs/make.jl` following CTModels pattern**
- Why: Documentation improvement requested
- Where: `docs/make.jl`
- Complexity: Moderate
- Notes:
  - Use `CTBase.automatic_reference_documentation` function
  - Add helper functions for paths (`src()`)
  - Add `EXCLUDE_SYMBOLS` list

**Add CTBase to `docs/Project.toml`**
- Why: Required for `CTBase.automatic_reference_documentation`
- Where: `docs/Project.toml`
- Complexity: Simple

### 🔵 Low

**Verify no breaking changes from CTBase 0.17**
- Why: Issue mentions "this must be checked"
- Where: Run full test suite
- Complexity: Simple

---

## 🔧 Technical Analysis

**Code Findings**:
- Current `src/` contains: `CTParser.jl`, `defaults.jl`, `onepass.jl`, `utils.jl`
- No `initial_guess.jl` exists yet
- No initial guess tests exist yet
- `docs/make.jl` uses simple `makedocs` without `automatic_reference_documentation`

**Julia Standards**:
- ✅ Documentation: Documenter.jl configured
- ✅ Testing: Test.jl with Aqua.jl quality checks
- ✅ Structure: Standard Julia package layout
- ⚠️ Docs pattern: Needs update to match CTModels style

**Dependencies to update**:
| Package | Current | Target | Location |
|---------|---------|--------|----------|
| CTBase | 0.17 | 0.17 | Project.toml ✅ |
| CTBase | 0.17 | 0.17 | test/Project.toml ✅ |
| CTModels | 0.6 | 0.7 | test/Project.toml ❌ |
| CTBase | - | add | docs/Project.toml ❌ |

---

## 🚧 Blockers

None identified. CTModels does not export methods (by design), so tests must use qualified calls like `CTModels.build_initial_guess`, `CTModels.validate_initial_guess`, etc. This is consistent with the original CTSolvers pattern.

---

## 💡 Recommendations

**Immediate**:
1. Update CTModels compat to `0.7` in `test/Project.toml`
2. Port `initial_guess.jl` with `:CTModels` as default prefix
3. Port and adapt tests, replacing `CTSolvers.` with `CTModels.`
4. Run test suite to verify no breaking changes

**Long-term**:
- Consider adding the `@init` macro to the main module exports
- Document the initial guess DSL in the package documentation

**Julia Alignment**:
- Use `CTBase.automatic_reference_documentation` for consistent documentation across control-toolbox packages
- Follow the established pattern from CTModels for documentation structure

---

**Status**: Needs attention  
**Effort**: Medium  
**Progress**: ~25% (compat updates partially done, main features pending)
