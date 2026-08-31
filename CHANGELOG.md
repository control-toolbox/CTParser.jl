<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to CTParser will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🐛 Bug Fixes

- **Trace mode (`@def name … end true`) no longer prints the parsed model twice** ([#344](https://github.com/control-toolbox/CTParser.jl/issues/344)). When the `:exa` backend is active, `def_fun` parses the definition a second time to build the ExaModels artifact; that second pass was inheriting the `log` flag and re-emitting the whole trace. It now runs with `log=false`.

## [0.9.4-beta] - 2026-08-30

### ✅ Compatibility

- **CTBase 0.30 is now supported** alongside CTBase 0.29 ([#342](https://github.com/control-toolbox/CTParser.jl/issues/342)).

### 🐛 Bug Fixes

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

### 🧪 Testing

- The 19 `@test_throws String` assertions now assert the concrete exception type.
- `test/test_exa_linalg.jl` migrated off the removed `ExaModels.variable`.

### 🔄 Refactoring

- **CI**: the retired self-hosted `kkt` runner replaced by `occidata`, and the trigger
  labels renamed to the ecosystem's `run ci <target>` form.

### ✅ Compatibility

- **Breaking**: see the two entries above and [BREAKING.md](BREAKING.md). Problem
  definitions written with `@def` are unaffected; the breaks are the ExaModels floor and
  the exception types.

## [0.8.15] - 2026-04-21 — baseline

This is the reference version. No changelog was maintained before this point; use
`git log` for earlier history. Breaking changes from this version onward are tracked in
[BREAKING.md](BREAKING.md).
