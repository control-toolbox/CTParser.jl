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

"""
$(TYPEDSIGNATURES)

Return the default prefix for functional backends.

The prefix identifies the module that provides the `PreModel` and
associated routines used by the `@def` macro in functional mode.

# Returns

- `Symbol`: name of the default functional backend module
  (currently `:CTModels`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_prefix_fun()
:CTModels
```
"""
__default_prefix_fun() = :CTModels

"""
$(TYPEDSIGNATURES)

Return the default prefix for ExaModels backends.

The prefix identifies the module that provides the ExaModels-based
discretisation support used when the `:exa` backend is active.

# Returns

- `Symbol`: name of the default ExaModels backend module
  (currently `:ExaModels`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_prefix_exa()
:ExaModels
```
"""
__default_prefix_exa() = :ExaModels

"""
$(TYPEDSIGNATURES)

Return the default prefix for error-handling utilities.

The prefix identifies the module that defines error types and
exceptions used when reporting parsing and modelling issues.

# Returns

- `Symbol`: name of the default error-handling module
  (currently `:CTBase`).

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_e_prefix()
:CTBase
```
"""
__default_e_prefix() = :CTBase

