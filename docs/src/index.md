# CTParser.jl

The `CTParser.jl` package is part of the [control-toolbox ecosystem](https://github.com/control-toolbox).

!!! note

    The root package is [OptimalControl.jl](https://github.com/control-toolbox/OptimalControl.jl) which aims
    to provide tools to model and solve optimal control problems with ordinary differential equations
    by direct and indirect methods, both on CPU and GPU.

!!! warning

    In some examples in the documentation, private methods are shown without the module prefix.
    This is done for the sake of clarity and readability.

    ```julia-repl
    julia> using CTParser
    julia> x = 1
    julia> private_fun(x) # throws an error
    ```

    This should instead be written as:

    ```julia-repl
    julia> using CTParser
    julia> x = 1
    julia> CTParser.private_fun(x)
    ```

    If the method is re-exported by another package,

    ```julia
    module OptimalControl
        import CTParser: private_fun
        export private_fun
    end
    ```

    then there is no need to prefix it with the original module name:

    ```julia-repl
    julia> using OptimalControl
    julia> x = 1
    julia> private_fun(x)
    ```

## What CTParser provides

At a high level, `CTParser.jl` is responsible for turning a compact,
mathematical DSL into executable Julia code for the rest of the
ecosystem (in particular `CTModels.jl` and `ExaModels.jl`). It does not
solve optimal control problems by itself; instead, it focuses on
parsing and code generation.

The two main entry points are:

- **`@def` macro** – define an optimal control problem from a
  human-readable specification.
- **`@init` macro** – define an initial guess for state, control and
  variables using a small initialisation DSL.

### The `@def` macro and its backends

The macro

```julia
ocp = @def begin
    # symbolic definition of an OCP
end
```

parses the block and builds an intermediate representation of the
optimal control problem. Internally, `@def` can target different
*backends*:

- the **functional backend** `:fun` (default), where the OCP is
  represented by a `CTModels.Model` and evaluated through Julia
  functions;
- the **ExaModels backend** `:exa`, where the same symbolic description
  is lowered to an `ExaModels.ExaModel` suitable for large-scale NLP
  solvers.

The active backends and their prefixes are controlled by
`prefix_fun()`, `prefix_exa()` and the corresponding setters. This
allows other packages (such as `OptimalControl.jl`) to plug in custom
model types while reusing the same parsing layer.

### The `@init` macro for initial guesses

The macro

```julia
ig = @init ocp begin
    # initialisation DSL
end
```

provides a compact way of building initial guesses for an OCP. The
block can mix:

- **time-dependent functions**, e.g. `u(t) := t` or `x(t) := [sin(t), 1]`;
- **time grids**, e.g. `x(T) := X` where `T` is a vector of times and
  `X` samples along the trajectory;
- **constants and aliases**, e.g. `a = 1.0; v(t) := a` or `tf := 1.0`.

`@init` itself only rewrites this DSL into a `NamedTuple` of symbolic
specifications. The actual construction and validation of a
backend-specific initial guess object is delegated to the module
selected by `init_prefix()` (by défaut `CTModels`), via
`build_initial_guess` et `validate_initial_guess`.
