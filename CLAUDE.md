# CTParser.jl — Agent Navigation Guide

Quick-reference for any agent working on this repository.

---

## Repository Layout

CTParser is a single flat module — no submodule split.

```text
src/        # onepass.jl (the `@def` parser), defaults.jl, initial_guess.jl, utils.jl
ext/        # CTParserExaModels.jl only — see below
test/       # Test suite: flat files (not test/suite/)
docs/       # Documentation site (DocumenterVitepress)
```

`ext/` holds exactly one extension, and it is **temporary**. ExaModels ships the
linear-algebra glue its expression nodes need (`dot`, `*` on node vectors/matrices,
`det`, `norm`, `Null` zero elimination) but never declares that extension in its
`[extensions]`, so Julia never loads it. Until upstream wires it up, CTParser carries
a port of it, triggered by `ExaModels` + `LinearAlgebra` weak dependencies. Delete
the file and the two weakdeps when upstream fixes it — see
[#325](https://github.com/control-toolbox/CTParser.jl/issues/325).

---

## Developer Resources

Design philosophy, operational rules, plan templates, and CI/CD conventions live in the
[control-toolbox Handbook](https://github.com/control-toolbox/Handbook):

| Topic | Link |
| --- | --- |
| Code philosophy (modules, types/traits, exceptions, docstrings, testing, docs) | [`PHILOSOPHY.md`](https://github.com/control-toolbox/Handbook/blob/main/PHILOSOPHY.md) |
| Operational rules (tests, coverage, docs, git) | [`RULES.md`](https://github.com/control-toolbox/Handbook/blob/main/RULES.md) |
| Plan template | [`PLAN.md`](https://github.com/control-toolbox/Handbook/blob/main/PLAN.md) |
| CI/CD workflows (centralized reusable workflows, label-gated triggers) | [`WORKFLOWS.md`](https://github.com/control-toolbox/Handbook/blob/main/WORKFLOWS.md) |

---

## Key Conventions

- **No top-level exports** — use `Package.symbol` everywhere.
- **Qualified imports** — `using Pkg: Pkg`, never bare `using Pkg`; `import` is never
  used.
- **Fake types at module top-level** — never inside test functions.
- **Structured errors** — seven typed exceptions under `CTException`; pick by the
  IncorrectArgument / PreconditionError / NotImplemented rule.
- **Type stability enforced** — hot paths must be `@inferred`-clean, verified with JET;
  setup-path dispatch is fine.
- **1-D is a scalar** — a one-dimensional state/control/variable is a `Number`, never a
  length-1 vector.
- **Plans before code** — write a plan and confirm with the user before touching files.
- **Docstrings last** — written only after all implementation steps are stable.
- **Never commit or push without explicit user approval.**
