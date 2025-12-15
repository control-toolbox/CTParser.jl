# CTParser.jl - Personal Development Context

## Working Preferences
- **Never push to remote without explicit instruction**
- Always use detailed commit messages with context
- Use standard Julia formatting conventions

## Current Work (Issue #181)
- **Branch**: `181-dev-adding-tensors`
- **Focus**: Adding tensor support to CTParser.jl
- **Main file**: `src/onepass.jl`

## Recent Architecture Changes

### p.x vs p.x_m in exa Backend (Completed 2025-12-09)
- **Old**: `p.x` (ExaModels.Variable) - limited operations
- **New**: `p.x_m` (matrix of ExaModels.Var) - full matrix operations

**Why p.x_m?**
Enables standard matrix operations not available with `p.x`:
- Range slicing: `p.x_m[1:2, :]` or `p.x_m[:, 5:10]`
- Direct arithmetic: `p.x_m[i, j+1] - p.x_m[i, j]`
- Broadcasting operations

**Definition** (in `p_state_exa!` line 482):
```julia
p.x_m = [x[i, j] for i ∈ 1:(n), j ∈ 1:grid_size+1]
```
Where:
- `i`: state dimension index (1 to n)
- `j`: time grid index (1 to grid_size+1)

## Codebase Structure

### Backend Architecture
- **`:fun` backend**: Uses `p.x` (unchanged)
- **`:exa` backend**: Uses `p.x_m` (modified)

### Key Functions in src/onepass.jl
- `ParsingInfo` struct: holds parsing state including `x`, `x_m`, `u`, etc.
- `p_state!` / `p_state_exa!`: state variable initialization
- `p_constraint_exa!`: boundary and path constraints
- `p_dynamics_coord_exa!`: dynamics discretization (euler, midpoint, trapeze schemes)
- `p_lagrange_exa!`: Lagrange cost function
- `p_mayer_exa!`: Mayer cost function

### Substitution Functions
- `subs2()`, `subs3()`, `subs5()`: convert symbolic expressions to matrix indexing
- Critical for replacing symbolic state/control variables with actual optimization variables

## Important Notes
- State range constraints are handled via box constraints (`l_x`, `u_x`), not explicit expressions
- Solution getter functions in `def_exa` still use `p.x` for ExaModels API calls (unchanged)
- Numerical schemes: euler, euler_implicit, midpoint, trapeze

## Testing Checklist (for p.x_m changes)
- [ ] Basic optimal control problems with state constraints
- [ ] Boundary conditions (initial and final)
- [ ] All numerical schemes
- [ ] Lagrange and Mayer costs
- [ ] Problems with range constraints on states

## Current Bug: p.x_m Indexing in ExaModels Generators (2025-12-09)

### Problem
The initial p.x → p.x_m migration (commit 2fe6a4e) introduced a critical bug: **`p.x_m` (a regular Julia Matrix) cannot be indexed with ExaModels symbolic expressions inside generators**.

When a variable like `j` is used in an ExaModels generator (`for j in 1:grid_size`), it becomes an `ExaModels.ParSource` or `ExaModels.Node2` symbolic type at runtime, not an integer. This causes:
```
ArgumentError: invalid index: ExaModels.Node2{...} of type ExaModels.Node2{...}
```

**Key insight**: ANY generator variable (even simple ones like `j`, not just arithmetic like `j+1`) becomes symbolic and cannot index a regular Julia array.

### Solution
Use `p.x` instead of `p.x_m` for indexing inside ExaModels generators, because:
- `p.x` is an `ExaModels.Variable` that handles symbolic indexing
- `p.x_m` is a plain `Matrix{ExaModels.Var}` that requires integer indices

### Locations Requiring Fix (9 total)

#### p_constraint_exa! (3 fixes needed)

**Line 754** - `:initial` constraints:
```julia
# WRONG:
e2 = subs3(e2, x0, p.x_m, i, 1)
# CORRECT:
e2 = subs3(e2, x0, p.x, i, 1)
```
Used in generator: `for $i in $rg` (line 757)

**Line 772** - `:final` constraints:
```julia
# WRONG:
e2 = subs3(e2, xf, p.x_m, i, :(grid_size+1))
# CORRECT:
e2 = subs3(e2, xf, p.x, i, :(grid_size+1))
```
Used in generator: `for $i in $rg` (line 775)

**Line 837** - `:state_fun/:control_fun/:mixed` constraints:
```julia
# WRONG:
e2 = subs2(e2, xt, p.x_m, j)
# CORRECT:
e2 = subs2(e2, xt, p.x, j)
```
Used in generator: `for $j in 1:grid_size+1` (line 843)

#### p_dynamics_coord_exa! (4 fixes needed)

**Line 937** - Euler forward scheme:
```julia
# WRONG:
ej1 = subs2(e, xt, p.x_m, j1)
# CORRECT:
ej1 = subs2(e, xt, p.x, j1)
```
Used in generators at lines 949, 956

**Line 940** - Euler backward scheme:
```julia
# WRONG:
ej2 = subs2(e, xt, p.x_m, j2)
# CORRECT:
ej2 = subs2(e, xt, p.x, j2)
```
Used in generators at lines 951, 956

**Line 943** - Midpoint scheme:
```julia
# WRONG:
ej12 = subs5(e, xt, p.x_m, j1)
# CORRECT:
ej12 = subs5(e, xt, p.x, j1)
```
Used in generator at line 953

**Line 946** - Direct difference computation:
```julia
# WRONG:
dxij = :($(p.x_m)[$i, $j2] - $(p.x_m)[$i, $j1])
# CORRECT:
dxij = :($(p.x)[$i, $j2] - $(p.x)[$i, $j1])
```
Used in all generators at lines 949, 951, 953, 956

#### p_lagrange_exa! (2 fixes needed)

**Line 1007** - Lagrange cost (various schemes):
```julia
# WRONG:
ej1 = subs2(e, xt, p.x_m, j1)
# CORRECT:
ej1 = subs2(e, xt, p.x, j1)
```
Used in generators at lines 1015, 1017, 1021, 1022

**Line 1010** - Midpoint scheme for Lagrange:
```julia
# WRONG:
ej12 = subs5(e, xt, p.x_m, j1)
# CORRECT:
ej12 = subs5(e, xt, p.x, j1)
```
Used in generator at line 1019

### Safe Uses (No Changes Needed)

**Lines 738-739** (p_constraint_exa! `:boundary`): Not in a generator, evaluated once
**Lines 1070-1071** (p_mayer_exa!): Not in a generator, evaluated once

These can safely use `p.x_m` because they're not inside ExaModels generators.

### Test Failure
Error appears in: `test_onepass_exa.jl:75` - "max (CPU, euler)" test
Stack trace shows: Line 200 (inside __wrap), originating from line 953 (midpoint scheme in p_dynamics_coord_exa!)
