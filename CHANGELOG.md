# Changelog

All notable changes to CTParser will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🐛 Bug Fixes

- **The error for a constraint bound that depends on `v`, the state, the control or the time suggests a rewriting built from the constraint** ([#352](https://github.com/control-toolbox/CTParser.jl/issues/352)). It always quoted `x₂(0) - v == 0`. It now names the bound as it was written and gives the constraint rewritten: `1 ≤ x₂(t)` → `x₂(t) ≥ 1` (sides swapped), `x₂(0) == v` → `x₂(0) - v == 0`, `x₁(t) ≤ u(t)` → `x₁(t) - u(t) ≤ 0`. The message starts with a capital letter.
- **A declaration with named components and a domain `R`, such as `u = (z,) ∈ R, control`, is rejected with a `ParsingError`** ([#4](https://github.com/control-toolbox/CTParser.jl/issues/4)). It was read as an alias: nothing was declared, `control_dimension` was 0 and no error was raised. The same holds for the state and the variable (`x = (q,) ∈ R, state`, `w = (a,) ∈ R, variable`). `u = (z,) ∈ R^1, control`, which was accepted (a control of dimension one whose component is named `z`), is rejected as well, so that `R` and `R^1` are treated alike. The error says to write `u ∈ R, control`, or to name the components of an `n`-dimensional element, `n ≥ 2`, with `R^n`. Code that relied on the silent alias, or on a name for a one-dimensional component, now fails at parse time.
- **`ẋ(t) = …` (a single `=`) gets a clear error** ([#353](https://github.com/control-toolbox/CTParser.jl/issues/353)). Any statement `f(args) = body` is rejected with a `ParsingError` that says an equation is written with `==`, shows the line as it was written (no `#= file:line =#` comment, no `begin … end` wrapper) and suggests it with `==`. Before, the message was `forbidden alias name: (∂(x))(t)` followed by the lowered block.
- **Linear algebra with a `Diagonal`, sparse or static constant works in a `:exa` `@def`** ([#365](https://github.com/control-toolbox/CTParser.jl/issues/365)). `Diagonal(d) * x(t)`, `SMatrix(A) * x(t)`, `x(t) + SVector(q)`, `dot(sparsevec(q), x(t))` and `x(t)' * sparse(A) * x(t)` raised `MethodError: … is ambiguous`: the extension's methods on any array of nodes collided with the methods LinearAlgebra, SparseArrays and StaticArrays define on their own arrays. The array methods now take strided arrays of nodes (what `@def` builds, and their views and reshapes), and the collisions left on the constants have methods on the exact intersection.
- **An `:exa` expression mixing `ExaModels.Constant` with a folded zero or one no longer raises an ambiguity error** ([#364](https://github.com/control-toolbox/CTParser.jl/issues/364)). The extension folds structural zeros and ones on `ExaModels.Null`, ExaModels folds them on `ExaModels.Constant`, and a call such as `Null(2) + Constant(0)` matched one rule of each side (`MethodError: … is ambiguous`), for instance from `0 * x₂(t) + ExaModels.Constant(0)` in a `@def`. The 32 intersection methods now fold such a call to a `Null`. Ordinary `@def` code, which never builds a `Constant`, was not affected.
- **A bound-length error raised while building an `:exa` model names its `@def` line again.** Since 0.9.1-beta ([#338](https://github.com/control-toolbox/CTParser.jl/issues/338)) `__wrap` no longer prints `Line n: …` for structured exceptions, and the bound-length `PreconditionError`s carried no `context`, so nothing pointed at the offending line. They now set `context = "line n: <source>"`.

### 🛠 Enhancements

- **Parse-time errors can carry a `suggestion`.** The internal helper `__throw` takes an optional `suggestion` keyword; the two errors above use it.
- **Two companion extensions, `CTParserExaModelsSparseArrays` and `CTParserExaModelsStaticArrays`**, loaded with SparseArrays or StaticArrays: sparse constants with nodes sum over the stored nonzeros only, so the expression graph has one term per nonzero; static constants give plain `Vector` / `Matrix` results.
- **`tr` of a matrix of nodes is LinearAlgebra's own**, which only needs `zero` and `+` on nodes: the extension no longer defines it. A non-square matrix raises its `DimensionMismatch` instead of `CTBase.IncorrectArgument`.
- **Errors raised from generated `:exa` code and the backend-selection errors set `suggestion` and `context`.**
- **The ExaModels linear-algebra extension throws typed errors.** A dimension mismatch between two operands is a `CTBase.PreconditionError`; a non-square matrix for `det`, `norm(v, Inf)` and non-3-vectors for `cross` are a `CTBase.IncorrectArgument`. They were an `AssertionError` or an `ErrorException`.
- **`Null(0) + x` folds to `x` in the extension**, as `x + Null(0)` already did.

### 🔄 Refactoring

- **`ext/CTParserExaModels.jl` is split into `ext/CTParserExaModels/`**: the module file and `types.jl`, `scalar.jl`, `checks.jl`, `arrays.jl`, `expansions.jl`. The array aliases name their type variables, so that a method prints the same signature on every Julia version.
- **One helper per generated error in `src/onepass.jl`.** The bound-length check, the unknown-scheme error and the backend checks were copied 7, 3 and 6 times; the list of `:exa` schemes is now a single constant, `EXA_SCHEMES`.

### 📚 Documentation

- **A page on the constraints of `@def` and the rule on their bounds** (a bound cannot depend on the state, the control, the variable or the time, and what to write instead).- **Two pages on the syntax of `@def`: declarations (dimension, names of the components, what is refused and why) and aliases versus equations (`=` against `==`).**

### 🧪 Testing

- **The suggestions of the dependent-bound error are covered** in `test/test_onepass_fun_bis.jl` ([#352](https://github.com/control-toolbox/CTParser.jl/issues/352)): the three shapes, a labelled constraint, and the roles of the operands of every constraint form.
- **The errors of [#4](https://github.com/control-toolbox/CTParser.jl/issues/4) and [#353](https://github.com/control-toolbox/CTParser.jl/issues/353) are covered** in `test/test_onepass_fun_bis.jl`: the three kinds of declaration, the forms that must stay valid, and the message and the suggestion.
- **`test/test_exa_linalg_inputs.jl`**: the linear algebra a user may write with every usual kind of constant (dense, `Diagonal`, `Symmetric`, triangular, `Tridiagonal`, sparse, static, `transpose`, `adjoint`, views, ranges), unit (evaluated against the numbers) and integration (`@def` solved against its dense version).
- **`test/test_exa_extensions.jl`**: the two companion extensions (loading, shared aliases, values, one term per stored nonzero, typed errors).
- **The ct-conventions TST8 check runs on the three extensions** (`test/test_aqua.jl`): no ambiguity, and every pirated method listed with its reason in `test/extension_piracies.jl`.
- **Use case no. 5 of `test/test_onepass_exa.jl` compares two different models, and the GPU comparisons of use cases 4, 5 and 9 use a tolerance the solver can meet** ([#383](https://github.com/control-toolbox/CTParser.jl/issues/383)): `m2` was built from `o1` instead of `o2` and `sol1` was solved twice; on GPU the objectives are now compared to `1e-5` (as for use case no. 8), since GPU solves are not reproducible at a fixed `max_iter` ([CTDirect.jl#531](https://github.com/control-toolbox/CTDirect.jl/issues/531)).
- **`import` is gone from the tests and the docs, and a guard keeps it out** (`MOD3`, `MOD4`, [#303](https://github.com/control-toolbox/CTParser.jl/issues/303)): `test/runtests.jl`, `test/test_control_zero.jl` and a docs snippet now use `using Pkg: sym`; the new `test/test_no_import.jl` fails on any tracked `.jl` or `.md` file with an `import` or a dotted `using Pkg.Sub` (`CHANGELOG.md` and `BREAKING.md` quote past text and are exempt from the dotted check).
- **`test/test_exa_constant_null.jl` pins the `Null`/`Constant` intersections**: each formerly ambiguous call (unit), the scope of the fix with a fake node type (contract), no ambiguity left on the scalar methods of the extension (non-regression), and the `@def` path that reached it, solved against a reference problem on the four schemes (integration).
- **[#343](https://github.com/control-toolbox/CTParser.jl/issues/343) is covered on both backends** for the state, the control and the time, on the upper-bound side and with its message; the documented work-around is also tested on `:exa`.
- **A sentinel in `test/test_exa_linalg.jl` fails as soon as ExaModels declares its own `ExaModelsOptimalControl` extension**, the signal to delete `ext/CTParserExaModels.jl` ([#325](https://github.com/control-toolbox/CTParser.jl/issues/325)).
- **Standard-output capture is one helper, `capture_stdout` in `test/utils.jl`.**

### ✅ Compatibility

- **No breaking changes**: the public API is unchanged. The error types thrown by the ExaModels extension change, see the non-breaking note in [BREAKING.md](BREAKING.md).

## [0.9.5-beta] - 2026-08-31

### 🐛 Bug Fixes

- **A constraint bound that depends on `v`, the state, the control or the time is now rejected with a clear `ParsingError`** ([#343](https://github.com/control-toolbox/CTParser.jl/issues/343)). Previously `x₂(0) == v` (with `v` the optimization variable) failed with an internal `UndefVarError: v##NNNN` leaked from generated code. Such a relation must be written as a functional constraint by moving the term to the constrained side, e.g. `x₂(0) - v == 0`.
- **Trace mode (`@def name … end true`) no longer prints the parsed model twice** ([#344](https://github.com/control-toolbox/CTParser.jl/issues/344)). When the `:exa` backend is active, `def_fun` parses the definition a second time to build the ExaModels artifact; that second pass was inheriting the `log` flag and re-emitting the whole trace. It now runs with `log=false`.

### ✅ Compatibility

- **No breaking changes**: both fixes only affect inputs that already errored (`#343`) or trace-only output (`#344`). See [BREAKING.md](BREAKING.md).

## [0.9.4-beta] - 2026-08-30

### 📦 Dependencies

- **CTBase 0.30 is now supported** alongside CTBase 0.29 ([#342](https://github.com/control-toolbox/CTParser.jl/issues/342)).

### ✅ Compatibility

- **No breaking changes**: `[compat]` is widened, no source change. See [BREAKING.md](BREAKING.md).

## [0.9.3-beta] - 2026-08-30

### 📚 Documentation

- **Docstring examples no longer trigger Documenter warnings** ([#341](https://github.com/control-toolbox/CTParser.jl/issues/341)). Examples embedded in docstrings now use static `julia` fences instead of `@example` fences, because Documenter does not execute `@example` blocks when those docstrings are transcluded into a consumer's `@docs` block.

### ✅ Compatibility

- **No breaking changes**: this release only adjusts docstring rendering and release metadata. See [BREAKING.md](BREAKING.md).

## [0.9.2-beta] - 2026-08-28

### 🧪 Testing

- **GPU runner capability detection recognises both `kkt` and `occidata`**
  ([#339](https://github.com/control-toolbox/CTParser.jl/issues/339)). The suite had no
  notion of which runner it was executing on. `test/runtests.jl` now defines a single
  `TestCapabilities` module holding `CUDA_FUNCTIONAL`, `ON_GPU_RUNNER` and
  `GPU_SOLVER_ARMED`. `ON_GPU_RUNNER` matches the `kkt` / `occidata` substring of
  `RUNNER_NAME` — the self-hosted runners are registered as `kkt-runner` /
  `occidata-runner`, whereas the `CI.yml` `runs_on` label is the bare `kkt` / `occidata`
  — so a missing or broken CUDA device now fails loudly on either GPU runner instead of
  being silently skipped. `RUNNER_NAME` is set by the GitHub Actions runner agent itself,
  so no `.github/workflows/CI.yml` or CTActions change was needed.

- **The two GPU test tiers now skip visibly.** `test_dynamics_exa.jl` and
  `test_onepass_exa.jl` gated their GPU runs behind a bare short-circuit on the raw device
  predicate, which made a correctly-skipped run (no device, as expected on a developer
  machine) and a silently-broken one (device missing on a GPU runner) produce the same
  output: a green run with the GPU tier simply absent. Both now branch to
  `Test.@test_skip`, so each scheme's GPU tier shows as `Broken` in the summary — eight
  entries on a CPU runner.

- **New meta-test `test/test_environment_contract.jl`**, mirroring CTSolvers'
  `test/suite/environment/test_environment_contract.jl`. It asserts that the
  MadNLPGPU/CUDSS GPU solver extension is armed (on every runner, CPU laptops included —
  this is the assertion that catches the CUDSS wiring regression), that a CUDA device is
  present when running on `kkt` or `occidata`, and that the silent-guard anti-pattern has
  not reappeared anywhere under `test/`.

### ✅ Compatibility

- **No breaking changes**: test-suite and release metadata only — `src/` and `ext/` are
  untouched, and `.github/workflows/CI.yml` is unchanged (it already targets `occidata`).
  See [BREAKING.md](BREAKING.md).

## [0.9.1-beta] - 2026-08-26

### 🐛 Bug Fixes

- **Structured errors no longer print a misleading source line** ([#338](https://github.com/control-toolbox/CTParser.jl/issues/338)). Errors such as an unsupported ExaModels `scheme` are propagated with their structured diagnostic without an unrelated `Line n: ...` message on `stdout`.

### ✅ Compatibility

- **No breaking changes**: exception types are unchanged, only the extra `stdout` line is gone. See [BREAKING.md](BREAKING.md).

## [0.9.0] - 2026-08-25

Aligns CTParser with the released ecosystem — CTBase 0.29.3, CTModels 0.18.0,
CTSolvers 0.5.3, CTFlows 0.17.2 — and leaves the beta series behind.

CTSolvers 0.5.3 already declared `ExaModels = "0.12"`, but it only reads backend
metadata from ExaModels. The package that *emits* the builder calls is CTParser, from
`def_exa` in `src/onepass.jl`, and ExaModels 0.12 deleted the mutable builder API those
calls used. Until this release, that declared 0.12 support was nominal: any `:exa` solve
failed at run time with `UndefVarError: variable not defined in ExaModels`.

### 💥 Breaking Changes

#### The `:exa` emission requires ExaModels ≥ 0.12

ExaModels 0.12 removed `variable`, `parameter`, `subexpr`, `constraint!` and
`LegacyExaCore` along with `src/deprecated.jl`. The replacement is functional:
`add_var` / `add_con` / `add_obj` each return `(new_core, result)`, so the code `@def`
generates now threads and rebinds the core.

Supporting both APIs is not practical. CTParser does not depend on ExaModels — the
module is reached through `prefix_exa()` — so it cannot branch on the version at
macro-expansion time.

**Migration**: none for `@def` users, whose problem definitions are unchanged. Callers
pinning ExaModels must move to 0.12.

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

`ExaCore` itself is called exactly as before. Under 0.12 it no longer warns — the
deprecation shim is gone — so no `concrete` keyword is passed. That is deliberate: with
the 0.12 default (`Vector{Any}` block storage) `typeof(core)` is invariant across every
`add_*`, so rebinding the core costs nothing, whereas `concrete = Val(true)` changes the
core's type on each `add_*` and would recompile the builder once per block.

See [BREAKING.md](BREAKING.md).

#### Untyped `String` errors are now `CTException` subtypes

Seventeen sites in `src/onepass.jl` threw a bare `String`, so `typeof(e) === String` and
callers had nothing typed to catch. They now throw the type the Handbook's choice rule
selects — a single argument's value out of domain is `IncorrectArgument`; a relational,
state or timing contract is `PreconditionError`:

| what | type |
| --- | --- |
| unknown numerical scheme | `IncorrectArgument` |
| lower/upper bound length mismatch | `PreconditionError` |
| bound lengths vs. the constrained range | `PreconditionError` |
| unknown value for the getter's `val` keyword | `IncorrectArgument` |
| unknown parsing backend | `IncorrectArgument` |
| `:fun` cannot be activated or deactivated | `PreconditionError` |

Each carries `got`/`expected` (or `reason`) and a `suggestion`, so `showerror` renders
the `Reason / Context / Hint` block every other error in the ecosystem produces, instead
of the message wrapped in quotes.

**Migration**:

```julia
# Before
catch e
    e == "unknown numerical scheme: gauss_legendre_2 (possible choices are ...)"
end

# After
catch e
    e isa CTBase.IncorrectArgument
end
```

See [BREAKING.md](BREAKING.md).

### ✨ New Features

- **`CTParserExaModels` extension** — linear algebra on ExaModels expression nodes
  (`dot`, `*` on node vectors and matrices, `det`, `norm`, `tr`, `diag`, `cross`,
  `convert`/`promote_rule`/`zero`/`one` for `AbstractNode`, and `Null` zero/one
  elimination), triggered by `ExaModels` + `LinearAlgebra` weak dependencies. It makes a
  dynamics written as `∂(x)(t) == A * x(t) + B * u(t)`, or an objective using
  `dot(q, x(t))`, work again under ExaModels 0.12.

  This is a **temporary** port of ExaModels' own `ext/ExaModelsOptimalControl.jl`, which
  ships in the package but is never declared in its `[extensions]`, so Julia never loads
  it. Upstream question: [madsuite-org/ExaModels.jl#323](https://github.com/madsuite-org/ExaModels.jl/issues/323).
  Delete the file and the two weak dependencies once upstream wires its own up.

### 🐛 Bug Fixes

- **Unknown discretisation schemes are catchable** ([#322](https://github.com/control-toolbox/CTParser.jl/issues/322)).
- **No deprecation warning from `ExaCore`** ([#323](https://github.com/control-toolbox/CTParser.jl/issues/323)).
  Fixed by the ExaModels upgrade itself, not by the `concrete = Val(true)` the issue
  suggested — under 0.12 the current call emits nothing. The `with_logger(NullLogger())`
  workaround can come out of the OptimalControl documentation
  ([OptimalControl.jl#877](https://github.com/control-toolbox/OptimalControl.jl/issues/877)).
- **No name clash on `constraint` in the test runner** ([#230](https://github.com/control-toolbox/CTParser.jl/issues/230)).
  `test/runtests.jl` used a bare `using ExaModels` while `constraint` was imported from
  CTModels, so every run opened with `WARNING: using ExaModels.constraint in module Main
  conflicts with an existing identifier`. Now a qualified `using ExaModels: ExaModels`.

### 📦 Dependencies

| | from | to |
| --- | --- | --- |
| `CTBase` | `0.18, 0.27, 0.28` | `0.29` |
| `CTModels` (test) | `0.10, 0.14, 0.15` | `0.18` |
| `ExaModels` (test) | `0.9` | `0.12` |
| `CUDA` (test) | `5` | `5, 6` |
| `MadNLP` (test) | `0.9` | `0.9, 0.10` |
| `MadNLPGPU` (test) | `0.8` | `0.8, 0.10` |
| `OrderedCollections` | `1` | `1, 2` |
| `Parameters` | `0.12, 0.13` | `0.13` |

CUDA, MadNLP and MadNLPGPU keep their lower bound, mirroring CTSolvers 0.5.3, so the
GitHub-hosted runners are not forced onto a CUDA 6 resolve. The CTBase, CTModels and
OrderedCollections bumps needed no source change.

### 🔄 Refactoring

- **CI**: the retired self-hosted `kkt` runner replaced by `occidata`, and the trigger
  labels renamed to the ecosystem's `run ci <target>` form.

### 🧪 Testing

- The 19 `@test_throws String` assertions now assert the concrete exception type.
- `test/test_exa_linalg.jl` migrated off the removed `ExaModels.variable`.

### ✅ Compatibility

- **Breaking**: see the two entries above and [BREAKING.md](BREAKING.md). Problem
  definitions written with `@def` are unaffected; the breaks are the ExaModels floor and
  the exception types.

## [0.8.15] - 2026-04-21 — baseline

This is the reference version. No changelog was maintained before this point; use
`git log` for earlier history. Breaking changes from this version onward are tracked in
[BREAKING.md](BREAKING.md).
