<!-- markdownlint-disable MD024 -->
# Breaking Changes

Breaking changes in CTParser releases, and how to migrate. Tracked from the 0.8.15
baseline onward; see [CHANGELOG.md](CHANGELOG.md) for the full record.

## [0.9.0] - 2026-08-25

### The `:exa` emission requires ExaModels ≥ 0.12

ExaModels 0.12 deleted `src/deprecated.jl` and with it the mutable builder API —
`variable`, `parameter`, `subexpr`, `constraint!`, `LegacyExaCore`. The code `@def`
generates for `backend = :exa` now uses the functional API instead, where `add_var` /
`add_con` / `add_obj` return `(new_core, result)` and the core is threaded through the
build.

**Who is affected**: anyone pinning ExaModels below 0.12 while using the `:exa` backend.
Nothing else — `@def` problem definitions, the `:fun` backend, and the shape of the
`(model, getter)` pair `build_examodel` returns are all unchanged.

There is no compatibility shim, and adding one is not practical: CTParser does not
depend on ExaModels (the module is reached through `prefix_exa()`), so it cannot detect
the version at macro-expansion time and pick an emission.

**Migration**: raise your ExaModels bound.

```toml
# Before
ExaModels = "0.9"

# After
ExaModels = "0.12"
```

For reference, what changed in the generated code:

```julia
# Before — emitted against ExaModels 0.9–0.11
c = ExaModels.ExaCore(base_type; backend, minimize)
x = ExaModels.variable(c, n, 0:grid_size; lvar, uvar, start)
ExaModels.constraint(c, expr for j in 0:grid_size-1; lcon, ucon)
ExaModels.objective(c, expr for j in 0:grid_size-1)

# After — emitted against ExaModels 0.12
c = ExaModels.ExaCore(base_type; backend, minimize)   # unchanged
c, x = ExaModels.add_var(c, n, 0:grid_size; lvar, uvar, start)
c, _ = ExaModels.add_con(c, expr for j in 0:grid_size-1; lcon, ucon)
c, _ = ExaModels.add_obj(c, expr for j in 0:grid_size-1)
```

### Untyped `String` errors are now `CTException` subtypes

Seventeen sites in `src/onepass.jl` threw a bare `String`. Callers could not dispatch on
them, and `showerror` fell back to `show`, rendering the message wrapped in quotes rather
than the formatted block every other error in the ecosystem produces.

**Who is affected**: code that catches these errors and compares the caught value to a
string, or that matches on `String`.

| what | type now |
| --- | --- |
| unknown numerical scheme | `CTBase.IncorrectArgument` |
| lower/upper bound length mismatch | `CTBase.PreconditionError` |
| bound lengths vs. the constrained range | `CTBase.PreconditionError` |
| unknown value for the getter's `val` keyword | `CTBase.IncorrectArgument` |
| unknown parsing backend | `CTBase.IncorrectArgument` |
| `:fun` cannot be activated or deactivated | `CTBase.PreconditionError` |

The split follows the Handbook's rule: a single argument's value out of domain is
`IncorrectArgument`; a relational, state or timing contract is `PreconditionError`. A
bound-length mismatch relates two things — the bounds to each other, or to the
constrained range — hence `PreconditionError`. `:fun` is a perfectly valid backend name;
what is forbidden is toggling it, which is a state contract, not a bad value.

**Migration**:

```julia
# Before
try
    solve(ocp, :exa; scheme=:gauss_legendre_2)
catch e
    e == "unknown numerical scheme: gauss_legendre_2 (possible choices are ...)"
end

# After
try
    solve(ocp, :exa; scheme=:gauss_legendre_2)
catch e
    e isa CTBase.IncorrectArgument
end
```

`@test_throws String` in a downstream test suite becomes `@test_throws
CTBase.IncorrectArgument` or `@test_throws CTBase.PreconditionError`, per the table.

## Non-breaking note (0.9.0)

- **New `CTParserExaModels` extension.** Additive, and triggered only when both
  `ExaModels` and `LinearAlgebra` are loaded — the main module gains no dependency. It
  restores the linear algebra on ExaModels expression nodes that a dynamics like
  `∂(x)(t) == A * x(t) + B * u(t)` needs, which ExaModels 0.12 ships but never registers
  in its own `[extensions]`. **No migration required.** It is temporary: when upstream
  wires its extension up ([madsuite-org/ExaModels.jl#323](https://github.com/madsuite-org/ExaModels.jl/issues/323)),
  the file and the two weak dependencies go away, with no user-visible change either way.

- **Compat bounds** raised for CTBase (`0.29`), CTModels (`0.18`), OrderedCollections
  (`1, 2`) and Parameters (`0.13`), and widened for CUDA (`5, 6`), MadNLP (`0.9, 0.10`)
  and MadNLPGPU (`0.8, 0.10`). **No breaking change**: none of these required a source
  change, and the `:fun` test groups stayed green at 1054/1054 across the bump.

## [0.8.15] - 2026-04-21 — baseline

Reference version. Breaking changes are tracked from here onward; use `git log` for
earlier history.
