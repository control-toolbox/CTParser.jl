# 📝 Documentation Preview

**Target**: `src/initial_guess.jl` (CTParser)  
**DOC_MODE**: all  
**Date**: 2025-12-09

## Summary

- **New docstrings**: 5  
  - `INIT_PREFIX` (const)
  - `__default_init_prefix`
  - `init_prefix`
  - `init_prefix!`
  - `_collect_init_specs`
  - `init_fun`
- **Improved / repositioned docstrings**: 1  
  - Macro `@init`

---

## 1. Macro `@init` (current vs proposed)

### Current docstring (attachée à `__default_init_prefix`)

```julia
"""
    @init ocp begin
        ...
    end

Macro to build initialization data (NamedTuple) from a small DSL of the form

    q(t) := sin(t)
    x(T) := X
    u := 0.1
    a = 1.0
    v(t) := a

The macro only transforms this syntax into a `NamedTuple`; all dimensional
validation and detailed handling of OCP aliases is performed by
`build_initial_guess` / `_initial_guess_from_namedtuple` on the backend
module selected by `init_prefix()` (by default `CTModels`).
"""
function __default_init_prefix()
    return :CTModels
end
```

### Proposed docstring (rattachée à la macro `@init`)

```julia
"""
    @init ocp begin
        ...
    end

Build an initial guess object for an optimal control problem from a small
initialisation DSL.

The block following `@init` is interpreted as a collection of assignment
rules for the state, control and variable components of an optimal control
problem, using a compact syntax of the form

```julia
q(t) := sin(t)     # time-dependent function
x(T) := X          # time grid and associated samples
u := 0.1          # constant value
a = 1.0           # ordinary Julia alias (not part of the initial guess)
v(t) := a         # time-dependent function using the alias above
```

The macro itself only rewrites this DSL into a `NamedTuple`-based
representation. All dimensional checks, interpretation of aliases and
construction of the concrete initial guess object are delegated to the
backend selected by [`init_prefix`](@ref) (by défaut `:CTModels`), via
`build_initial_guess` and `validate_initial_guess`.

An optional keyword-like trailing argument controls logging:

```julia
ig = @init ocp begin
    u(t) := t
end log = true
```

When `log = true`, the macro additionally prints a human-readable
`NamedTuple`-like representation of the specification.

# Arguments

- `ocp`: symbolic optimal control problem built with `@def`.
- `begin ... end`: block containing the initialisation DSL.
- `log`: optional Boolean keyword (default `false`) enabling textual
  logging of the parsed specification.

# Returns

- `AbstractOptimalControlInitialGuess`: backend-specific initial guess
  object produced by the current backend (par défaut `CTModels`).

# Example

```julia-repl
julia> using CTParser

julia> ocp = @def begin
           t ∈ [0, 1], time
           x ∈ R, state
           u ∈ R, control
           ẋ(t) == u(t)
           x(0) == 0
           x(1) == 0
           ∫(0.5u(t)^2) → min
       end

julia> ig = @init ocp begin
           u(t) := t
       end

julia> ig isa CTModels.AbstractOptimalControlInitialGuess
true
```
"""
macro init(ocp, e, rest...)
    # corps inchangé
end
```

---

## 2. Constante `INIT_PREFIX`

### Current

Aucune docstring dédiée.

```julia
const INIT_PREFIX = Ref(__default_init_prefix())
```

### Proposed docstring

```julia
"""
Current backend prefix used by `@init`.

This reference stores the symbol of the backend module that provides
`build_initial_guess` and `validate_initial_guess`. It is initialised
by [`__default_init_prefix`](@ref) and can be updated at runtime via
[`init_prefix!`](@ref).
"""
const INIT_PREFIX = Ref(__default_init_prefix())
```

---

## 3. Fonction `__default_init_prefix`

### Current

Aucune docstring dédiée (hormis la doc de la macro `@init` actuellement
placée ici).

```julia
function __default_init_prefix()
    return :CTModels
end
```

### Proposed docstring

```julia
"""
$(TYPEDSIGNATURES)

Return the default backend prefix used by `@init`.

The returned symbol identifies the module that implements
`build_initial_guess` and `validate_initial_guess`. In the current
implementation this is `:CTModels`.

# Returns

- `Symbol`: name of the default backend module.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_init_prefix()
:CTModels
```
"""
function __default_init_prefix()
    return :CTModels
end
```

---

## 4. Fonction `init_prefix`

### Current

Aucune docstring dédiée.

```julia
function init_prefix()
    return INIT_PREFIX[]
end
```

### Proposed docstring

```julia
"""
$(TYPEDSIGNATURES)

Return the current backend prefix used by `@init`.

This is the symbol of the module that will be used to interpret the
`NamedTuple` produced by the initialisation DSL, typically `:CTModels`.

# Returns

- `Symbol`: name of the backend module currently used by `@init`.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.init_prefix()
:CTModels
```
"""
function init_prefix()
    return INIT_PREFIX[]
end
```

---

## 5. Fonction `init_prefix!`

### Current

Aucune docstring dédiée.

```julia
function init_prefix!(p)
    INIT_PREFIX[] = p
    return nothing
end
```

### Proposed docstring

```julia
"""
$(TYPEDSIGNATURES)

Set the backend prefix used by `@init`.

This function updates the global [`INIT_PREFIX`](@ref), thereby changing
which module is used to build and validate initial guesses.

# Arguments

- `p::Symbol`: name of the backend module to use (e.g. `:CTModels`).

# Returns

- `Nothing`.

# Example

```julia-repl
julia> using CTParser

julia> old = CTParser.init_prefix();

julia> CTParser.init_prefix!(:MyBackend)

julia> CTParser.init_prefix()
:MyBackend

julia> CTParser.init_prefix!(old);  # restore
```
"""
function init_prefix!(p)
    INIT_PREFIX[] = p
    return nothing
end
```

---

## 6. Fonction `_collect_init_specs`

### Current

Aucune docstring dédiée.

```julia
function _collect_init_specs(ex)
    alias_stmts = Expr[]
    keys = Symbol[]
    vals = Any[]
    # ...
    return alias_stmts, keys, vals
end
```

### Proposed docstring

```julia
"""
$(TYPEDSIGNATURES)

Internal helper that parses the body of an `@init` block.

The function walks through the expression `ex` and splits it into

- *alias statements*, which are left as ordinary Julia assignments and
  executed verbatim inside the generated block;
- *initialisation specifications* of the form `lhs := rhs` or
  `lhs(t) := rhs` / `lhs(T) := rhs`, which are converted into keys and
  values used to build a `NamedTuple`.

# Arguments

- `ex::Any`: expression or block coming from the body of `@init`.

# Returns

- `alias_stmts::Vector{Expr}`: ordinary statements to execute before
  building the initial guess.
- `keys::Vector{Symbol}`: names of the components being initialised
  (e.g. `:q`, `:v`, `:u`, `:tf`).
- `vals::Vector{Any}`: expressions representing the corresponding
  values, functions or `(T, data)` pairs.
"""
function _collect_init_specs(ex)
    alias_stmts = Expr[]
    keys = Symbol[]
    vals = Any[]
    # corps inchangé
end
```

---

## 7. Fonction `init_fun`

### Current

Aucune docstring dédiée.

```julia
function init_fun(ocp, e)
    alias_stmts, keys, vals = _collect_init_specs(e)
    pref = init_prefix()
    # ...
    return log_str, code_expr
end
```

### Proposed docstring

```julia
"""
$(TYPEDSIGNATURES)

Lowering function used by `@init` to build the expanded code.

Given an optimal control problem `ocp` and the body `e` of an `@init`
block, this function collects the initialisation specifications, builds
an appropriate `NamedTuple` expression and constructs the call to
`build_initial_guess` / `validate_initial_guess` on the current backend
(prefix returned by [`init_prefix`](@ref)).

It also produces a compact string representation of the specification,
used for optional logging when `log = true` is requested at the
macro level.

# Arguments

- `ocp`: symbolic optimal control problem built with `@def`.
- `e`: expression corresponding to the body of the `@init` block.

# Returns

- `log_str::String`: human-readable `NamedTuple`-like description of
  the specification.
- `code_expr::Expr`: block of Julia code that builds and validates the
  initial guess when executed.
"""
function init_fun(ocp, e)
    alias_stmts, keys, vals = _collect_init_specs(e)
    pref = init_prefix()
    # corps inchangé
end
```

---

## Next Steps

1. **Appliquer toutes les docstrings** proposées dans `src/initial_guess.jl` (en replaçant la doc actuelle de `@init` et en insérant les nouvelles docstrings).
2. **Appliquer sélectivement** seulement un sous-ensemble (par ex. uniquement macro `@init` + `init_prefix`/`init_prefix!`).
3. **Demander des ajustements** (formulation, exemples, niveau de détail).
