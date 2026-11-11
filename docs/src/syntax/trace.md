# Trace mode

Adding `true` after a `@def` block prints the model as the parser reads it, one line per statement of
the block. It is a way to check what a block means, in particular after the aliases are expanded.

```julia
c(t) = 1.0
@def damped_integrator begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x = (q, v) ∈ R², state
    u ∈ R, control
    q̇ = v(t)
    v̇ = u(t) - c(t) * v(t)
    ẋ(t) == [q̇, v̇]
    ∫(u(t)^2) → min
end true;
```

```text
variable: tf, dim: 1
time: t, initial time: 0, final time: tf
state: x, dim: 2
control: u, dim: 1
alias: q̇ = (x[2])(t)
alias: v̇ = u(t) - c(t) * (x[2])(t)
dynamics: ∂(x)(t) == [(x[2])(t), u(t) - c(t) * (x[2])(t)]
objective (Lagrange): ∫(u(t) ^ 2) → min
```

The trailing `;` keeps the REPL from printing the model a second time.

## What the trace shows

- **The aliases are expanded.** `q̇ = v(t)` is shown as `(x[2])(t)`: `v` is the second component of
  the state, which the parser writes `x[2]`. The components of a state, a control or a variable of
  dimension two or more are always shown as `x[i]`.
- **A scalar is shown under its own name.** A one-dimensional variable, state or control (`tf`, `u`
  above) is a scalar: the trace shows `tf` and `u(t)`, not an index into a vector of length one.
  Internally the parser renames such an element and reads it as `name##123[1]`; this internal name is
  never shown.
- **The constraints show their type**, for instance `constraint ((:control_range, 1)): -1 ≤ u(t) ≤ 1`.
  A constraint without a label is shown without one; a label given by the user is shown at the end
  of the line, `... ≤ 2,    (bound)`. A missing bound is shown as `nothing`.

## Errors

The trace is printed while the block is parsed, so it stops at the statement that fails. The error
then names the line, `Line n: …`, as it was written; see [Aliases and equations](equations.md).
