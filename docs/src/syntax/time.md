# The state and the control are functions of time

In a `@def` block the state `x` and the control `u` are functions of the time `t`. They are used
through a call, and the argument says when:

| Where | Allowed |
| --- | --- |
| dynamics, costs, constraints along the trajectory | `x(t)`, `u(t)`, and their components `x₁(t)`, `x[2](t)`, … |
| boundary conditions, Mayer cost | `x(t0)` and `x(tf)`, with `t0` and `tf` the bounds of the time interval as they were declared, and the components `x₁(t0)`, … |
| anywhere | the control only at the time `t` |

```julia
@def begin
    t ∈ [0, 1], time
    x ∈ R², state
    u ∈ R, control
    ẋ(t) == [x₂(t), u(t)]            # x(t), u(t)
    x(0) == [0, 0]                   # x at the initial time
    x₁(t) + u(t) ≤ 5                 # along the trajectory
    ∫(x₁(t)^2 + u(t)^2) + x₁(1) → min   # integral along the trajectory, state at the final time
end
```

If the interval is declared with names, `t ∈ [t0, tf], time`, the boundary conditions use these
names: `x(t0)`, `x(tf)`.

## Why the control is only evaluated at `t`

The state is a trajectory: it has a value at every time, and the problem can fix or bound it at the
initial or at the final time (`x(0) == [0, 0]`, `x₁(1) + x₂(0) ≤ 3`). The control is not constrained
at an instant: what the model holds for it is an expression that is valid for every time `t`
(`-1 ≤ u(t) ≤ 1`, a term of the cost or of the dynamics). `u(0)` or `u(tf)` has no place in this
description, and a statement that contains it is refused, in the dynamics, in the costs and in the
constraints.

## What is refused

A state or a control that is not evaluated at one of these times is a mistake, and the parser
refuses it with a `ParsingError`. The error names the line as it was written.

**Without a time.** A bare `x₁`, `x` or `u` where `x₁(t)`, `x(t)` or `u(t)` was meant:

```julia
ẋ(t) == [x₂(t), u]                 # The control `u` is used here without being evaluated at a time
x₁ → min                           # The state `x` is used here without being evaluated at a time
x₁(t) + u ≤ 1                      # same, in a constraint
```

**At another time.** A call whose argument is none of the allowed times:

```julia
x(0.5) == [0, 0]                   # The state `x` is evaluated at `0.5`, which is neither the time `t`, ...
x(s) == [0, 0]                     # same, `s` is not a time of the problem
x₁(0) + u(0) ≤ 1                   # The control `u` is evaluated at `0`, which is not the time `t`
```

The suggestion of the error gives the times that are allowed (`t`, `0` and `1` for a problem on
`[0, 1]`).

## Why

Before this check, the parser let such a statement through. A bare `x₁` is rewritten to the
component `x[1]`, a name that exists in the generated code; the model was built, and the statement
meant nothing, or took the value of a variable of the same name that happened to exist where the
model is evaluated. A call at an unexpected time was either refused with the generic message
`bad constraint declaration`, or accepted in a cost.

## Not checked

A state evaluated at `t0` or `tf` inside the dynamics (`ẋ(t) == u(t) + x₁(0)`) is not refused: it is
an evaluation at an allowed time, in a place where it has no meaning for the model. The parser does
not check where the boundary times are used.
