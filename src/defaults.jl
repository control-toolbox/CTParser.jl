"""
$(TYPEDSIGNATURES)

Return the default parsing backend used by `@def`.

This controls whether the parser targets the functional (`:fun`) or
another backend. In the current implementation the default is `:fun`.

# Returns

- `Symbol`: name of the default parsing backend.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_parsing_backend()
:fun
```
"""
__default_parsing_backend() = :fun

"""
$(TYPEDSIGNATURES)

Return the default collocation scheme used for ExaModels backends.

The symbol identifies the time-discretisation scheme to be used when
building ExaModels-based formulations.

# Returns

- `Symbol`: name of the default ExaModels scheme (currently `:midpoint`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_scheme_exa()
:midpoint
```
"""
__default_scheme_exa() = :midpoint

"""
$(TYPEDSIGNATURES)

Return the default grid size used for ExaModels discretisations.

This is the number of time intervals used when constructing the
discretised optimal control problem.

# Returns

- `Int`: default grid size (currently `250`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_grid_size_exa()
250
```
"""
__default_grid_size_exa() = 250

"""
$(TYPEDSIGNATURES)

Return the default ExaModels backend, if any.

When this is `nothing`, no specific ExaModels backend is selected by
default and the library's own defaults are used instead.

# Returns

- `Union{Nothing,Symbol}`: name of the default ExaModels backend, or
  `nothing` if the generic backend should be used.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_backend_exa()
nothing
```
"""
__default_backend_exa() = nothing

"""
$(TYPEDSIGNATURES)

Return the default initial values used for ExaModels problems.

The tuple typically corresponds to initial guesses for the state,
control and additional variables when no user-specified initialisation
is provided.

# Returns

- `Tuple`: default initial values (currently `(0.1, 0.1, 0.1)`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_init_exa()
(0.1, 0.1, 0.1)
```
"""
__default_init_exa() = (0.1, 0.1, 0.1) # default init for v, x, u

"""
$(TYPEDSIGNATURES)

Return the default base scalar type used for ExaModels discretisations.

This type is used for the floating-point data stored in the underlying
problems.

# Returns

- `DataType`: default scalar type (currently `Float64`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_base_type_exa()
Float64
```
"""
__default_base_type_exa() = Float64

# Modules the generated code calls. They are interpolated into the expressions
# `@def` and `@init` return, so they are resolved where the macros are defined —
# a `GlobalRef` to the module itself — and never looked up by name in the
# caller's module (#356). Being module objects, not `Symbol`s, they are not
# configurable: there is exactly one provider per concern.
const FUN = CTModels # functional backend: PreModel, build!, definition!, ...
const EXA = ExaModels # :exa backend: ExaCore, add_var, add_con, ExaModel, ...
const ERR = CTBase # exceptions in generated code: ParsingError, ...
