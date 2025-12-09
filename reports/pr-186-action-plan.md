# 🎯 Action Plan: PR #186 – Add tests

**Date**: 2025-12-09  
**PR**: #186 by @ocots | **Branch**: `185-dev-tests` → `main`  
**State**: OPEN | **Linked Issue**: #185 – “[Dev] Tests”

---

## 📋 Overview

**Issue Summary (#185)**  
Internal dev issue to strengthen the test suite of `CTParser.jl`:

- Make a pass on **unit and integration tests**.
- **Do not replace** existing tests.
- For **each existing test file**, create a new file with `_bis` suffix and put **new tests only** there (no duplication).
- Ensure a **clear separation** between **unit** and **integration** tests.
- When **fake structs** are needed to test logic:
  - Define them at **top level** of the test file.
  - Place them **before** the test function(s).
  - Use **unique struct names** per file to avoid ambiguity.

**PR Summary (#186)**  
- Title: *“Add tests”*  
- Branch: `185-dev-tests`  
- Currently: PR metadata only; `changedFiles = 0` (no test files or source files changed in the remote PR yet).  
- Intention: This PR will host the new `_bis` tests required by issue #185.

**Status**: Needs work – PR is a skeleton; all testing work still to be implemented on `185-dev-tests`.

---

## 🎯 Gap Analysis

### ✅ Completed Requirements (relative to #185)

- ✓ **Branch + PR scaffold**:  
  - Branch `185-dev-tests` exists, PR #186 is open and targets `main`.  
  - This provides a dedicated place to accumulate new tests for issue #185.

- ✓ **Issue-level diagnosis and report** (outside the PR):  
  - A detailed status report exists in `reports/issue-185-report.md`, describing current tests and high-level testing strategy.  
  - This is a good design reference but is not yet wired into PR #186.

### ❌ Missing Requirements (core for this PR)

Everything described below is **not yet implemented in the PR** and forms the basis of the action plan:

- ✗ **Create `_bis` test files for each existing test file**  
  - `test/test_utils_bis.jl`  
  - `test/test_prefix_bis.jl`  
  - `test/test_onepass_fun_bis.jl`  
  - `test/test_onepass_exa_bis.jl`  
  - (Optional) `test/test_aqua_bis.jl` if extra Aqua/meta tests are desired.

- ✗ **Wire `_bis` tests into `test/runtests.jl`**  
  - Ensure `_bis` tests run in CI **in addition** to existing tests.  
  - Respect the requirement of **not modifying / replacing** existing tests, only extending the test list.

- ✗ **Unit tests targeting internal logic via fake structs**  
  - Particularly for parsing helpers in `src/onepass.jl` such as:  
    - `p_time!`, `p_state!`, `p_control!`,  
    - `p_dynamics!`, `p_dynamics_coord!`,  
    - `p_lagrange!`, `p_mayer!`, `p_bolza_fun!`, `p_bolza_exa!`,  
    - `p_constraint!`, `p_constraint_fun!`, `p_constraint_exa!`, etc.  
  - Fake structs must be defined **at top level**, with **unique names** per file.

- ✗ **Additional unit tests and edge cases for utilities and defaults**  
  - `src/utils.jl`: `subs*`, `replace_call`, `has`, `concat`, `constraint_type`, etc.  
  - `src/defaults.jl`: defaults and prefix helpers (already tested, but `_bis` should add corner cases).

- ✗ **Extra integration tests**  
  - New OCPs and scenarios for:
    - Functional backend (`:fun`) via CTModels.
    - ExaModels backend (`:exa`) through `@def_exa` and `discretise_exa`-style logic.
  - More “realistic” and stress-test cases, both CPU and GPU (where available).

### ➕ Additional Work Done

- None within PR #186 yet (no file changes). All test work is ahead.

---

## 🧪 Test Status

**Overall for PR #186**: ⚠️ No new tests in the PR yet.

**Existing Test Structure (from `/test-julia` analysis)**

- `src/`:
  - `src/CTParser.jl`
  - `src/defaults.jl`
  - `src/onepass.jl`
  - `src/utils.jl`
- `test/`:
  - `test/runtests.jl`
  - `test/test_utils.jl`
  - `test/test_prefix.jl`
  - `test/test_onepass_fun.jl`
  - `test/test_onepass_exa.jl`
  - `test/test_aqua.jl`

**Current coverage (before `_bis`)**:

- `test_utils.jl`: unit tests for expression utilities (`subs`, `subs2`–`subs5`, `replace_call`, `has`, `concat`, `constraint_type`, etc.).
- `test_prefix.jl`: tests for defaults and runtime prefix setters.
- `test_onepass_fun.jl`: extensive integration tests of `@def` with CTModels backend.
- `test_onepass_exa.jl`: extensive integration tests with ExaModels backend, including several realistic OCPs.
- `test_aqua.jl`: Aqua.jl meta-tests on the module.

The PR must **add** to this by creating `_bis` tests; none are present yet.

---

## 📝 Review Feedback

- No code reviews yet (no reviewers’ comments).  
- Reviewer `jbcaillau` is requested; checks are pending.

---

## 🔧 Code Quality Assessment

For PR #186 itself:

- **Code diff**: currently empty (no changed files).  
- **Tests**: none added yet.  
- **Documentation**: unchanged by this PR.

The quality assessment and recommendations below are therefore **prospective**: they describe how tests should be written, not how they are currently written in this PR.

---

## 📋 Proposed Action Plan

### 🔴 Critical Priority (blocking merge)

1. **Create `_bis` test files and wire them into `runtests.jl`**

   - **Why**: This is the core requirement of issue #185: extend tests without touching existing ones.
   - **Where**:
     - New files:
       - `test/test_utils_bis.jl`
       - `test/test_prefix_bis.jl`
       - `test/test_onepass_fun_bis.jl`
       - `test/test_onepass_exa_bis.jl`
       - (optional) `test/test_aqua_bis.jl`
     - `test/runtests.jl`: extend the loop over `name` to include new entries (e.g. `:utils_bis`, `:prefix_bis`, `:onepass_fun_bis`, `:onepass_exa_bis`) **without removing** any existing names.
   - **Estimated effort**: Medium (file creation + careful wiring).

2. **Define fake structs / fixtures at top level in `_bis` files**

   - **Why**: The issue explicitly requires testing logic via fake structs:
     - They avoid pulling heavy dependencies into unit tests.
     - They let you exercise internal parsing and backend logic in a controlled way.
   - **Where**:
     - For example:
       - `test/test_onepass_fun_bis.jl`:
         - `struct FunBackendCase1 ... end`
         - `struct ParsingContextFun1 ... end`
       - `test/test_onepass_exa_bis.jl`:
         - `struct ExaBackendCase1 ... end`
         - `struct ExaConstraintFixture1 ... end`
     - Place these structs at the **top of each `_bis` file**, before the `test_*_bis()` function.
   - **Conventions**:
     - Unique struct names per file (e.g. prefix with file or feature name).
     - Only minimal fields / methods needed for the tests (keep them lightweight).
   - **Estimated effort**: Medium (needs careful design per feature).

3. **Add unit tests for core parsing helpers with fake structs**

   - **Why**: Many helpers in `src/onepass.jl` are currently tested indirectly via `@def` integration tests; dedicated unit tests will:
     - make regressions easier to catch,  
     - document invariants and error conditions explicitly.
   - **Targets** (examples, not exhaustive):
     - Time / variable handling: `p_time!`, variable-dependent time intervals.
     - State/control declarations: `p_state!`, `p_control!`.
     - Dynamics and coordinates: `p_dynamics!`, `p_dynamics_coord!`.
     - Costs: `p_lagrange!`, `p_mayer!`, `p_bolza_fun!`, `p_bolza_exa!`.
     - Constraints: `p_constraint!`, `p_constraint_fun!`, `p_constraint_exa!`.
   - **Where**:
     - Primarily `test/test_onepass_fun_bis.jl` and `test/test_onepass_exa_bis.jl`.
   - **Approach**:
     - Use fake structs to represent minimal “OCP-like” objects or parsing contexts.
     - Test both **valid** and **invalid** inputs:
       - Ensure valid inputs produce the expected structures / calls.
       - Ensure invalid inputs raise `ParsingError` / `UnauthorizedCall` where appropriate.
   - **Estimated effort**: Large (requires exploring internals and designing concise fixtures).

4. **Add new integration tests for both backends (`:fun` and `:exa`)**

   - **Why**: The current tests already cover many typical and error scenarios; issue #185 asks to “do the maximum” in terms of unit, edge case, and integration testing.
   - **Where**:
     - `test/test_onepass_fun_bis.jl`: OCP definitions that stress unusual DSL constructs (e.g. combinations of variable/time/state/control with non-trivial aliasing).
     - `test/test_onepass_exa_bis.jl`: additional ExaModels OCPs and bounds scenarios not yet covered, ideally lighter than the heaviest existing cases to limit runtime.
   - **Examples**:
     - OCPs combining:
       - variable start/end times,
       - mixed state/control/variable constraints,
       - non-standard symbols or time aliases.
     - Cross-check that when a problem is expressible for both `:fun` and `:exa`, the semantics match (criterion, constraints, bounds).
   - **Estimated effort**: Large (scenario design + potential performance considerations).

5. **Run full `Pkg.test()` on `185-dev-tests` and ensure CI viability**

   - **Why**: New `_bis` tests will increase runtime; must ensure `Pkg.test()` still completes in reasonable time on CPU and that GPU-dependent tests behave well under CI.
   - **Where**:
     - Local run:
       - `julia --project=. -e 'using Pkg; Pkg.test()'`
     - Ensure:
       - All `_bis` tests are included.
       - Heavy GPU scenarios are still guarded by `CUDA.functional()` checks.
   - **Estimated effort**: Small/Medium (depending on iteration).

---

### 🟡 High Priority (should do before merge)

1. **Extend unit tests for `src/utils.jl` in `test_utils_bis.jl`**

   - **Why**: `test_utils.jl` already covers many cases, but `_bis` is ideal for:
     - new regression tests,
     - rare syntactic patterns (e.g. nested calls, exotic Unicode),
     - edge cases around ranges, empty expressions, or degenerate inputs.
   - **Where**:
     - `test/test_utils_bis.jl`
   - **Focus**:
     - `subs*` with degenerate ranges,
     - `replace_call` with more complex call nesting,
     - `has` for corner cases (e.g. scalars vs symbols),
     - `concat` with empty and single-expression blocks,
     - `constraint_type` for edge expressions near boundaries of classification.

2. **Add more prefix / backend lifecycle tests in `test_prefix_bis.jl`**

   - **Why**: `test_prefix.jl` checks defaults and setters, but `_bis` can:
     - test `activate_backend`, `deactivate_backend`, `is_active_backend`,
     - verify that backend activation interacts correctly with parsing choices and defaults.
   - **Where**:
     - `test/test_prefix_bis.jl`
   - **Examples**:
     - Switching backends multiple times in the same process.
     - Ensuring default prefixes are restored after tests.
     - Ensuring `:fun` and `:exa` activation does not leak across testsets.

3. **Clarify unit vs integration split inside `_bis` files**

   - **Why**: For maintainability, new contributors should immediately see which tests are “small” vs “large”.
   - **Where**:
     - At the top of each `_bis` file.
   - **Approach**:
     - Add a short comment or section header:
       - “This file contains UNIT tests for …”
       - “This section contains INTEGRATION tests for …”
     - Group `@testset`s accordingly.

---

### 🟢 Medium Priority (nice to have before or shortly after merge)

1. **Systematic edge-case coverage for bounds and constraints**

   - **Why**: Many scenarios are already in `test_onepass_exa.jl`; `_bis` files can hold additional rare cases without cluttering the original.
   - **Where**:
     - `test/test_onepass_exa_bis.jl`
   - **Examples**:
     - Mixed scalar/vector bounds with borderline dimensions.
     - Degenerate time intervals (e.g. `t ∈ [t0, t0]` if supported).
     - Very small or very large grid sizes, if relevant.

2. **Additional tests for error messages and exception types**

   - **Why**: Ensure that incorrect DSL usage produces the intended `ParsingError`/`UnauthorizedCall` and clear error patterns (even if you don’t test the exact message strings).
   - **Where**:
     - `test/test_onepass_fun_bis.jl`
     - `test/test_onepass_exa_bis.jl`
   - **Approach**:
     - Focus on the “control flow” (which error type is thrown), not string matching.

---

### 🔵 Low Priority (can be deferred)

1. **Optional: dedicated `_bis` for Aqua/meta tests**

   - **Why**: If you want to prototype additional Aqua or analysis checks (e.g. more restrictive piracy detection), you can use `test_aqua_bis.jl` without touching the original.
   - **Where**:
     - `test/test_aqua_bis.jl` (only if needed).
   - **Notes**:
     - Likely small incremental benefit; can be done later.

2. **Performance-oriented tests**

   - **Why**: Check that new logic or edge handling does not cause excessive allocations or slowdowns.
   - **Where**:
     - Possibly integration tests with `@test` on simple timing/allocation metrics (only if stable enough).
   - **Notes**:
     - More useful as follow-up once functional correctness is well covered.

---

## ✅ Summary

- PR #186 is currently an empty “Add tests” scaffold pointing to branch `185-dev-tests`.  
- Issue #185 clearly defines a testing strategy based on `_bis` files, fake structs, and strong unit + integration coverage.  
- The action plan above provides a **test-centric roadmap**:
  - **Critical**: create and wire `_bis` files, design fake structs, add core unit + integration tests, and run full `Pkg.test()`.
  - **High/Medium**: extend utilities/defaults coverage, add more edge cases and error-path tests, clarify unit vs integration structure.
  - **Low**: optional enhancements (extra Aqua tests, performance checks).

Once confirmed, this plan can be implemented incrementally on branch `185-dev-tests`, starting with the lightest `_bis` files (`test_utils_bis.jl`, `test_prefix_bis.jl`) before tackling the heavier integration suites (`test_onepass_fun_bis.jl`, `test_onepass_exa_bis.jl`).
