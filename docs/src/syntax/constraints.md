# Constraints and their bounds

A constraint is written `lower ≤ expression ≤ upper`, `expression ≤ upper`, `expression ≥ lower` or
`expression == value`. The **expression** is the constrained one: it is evaluated along the
trajectory, so it can depend on the state, the control, the variable and the time. The **bounds**
(`lower`, `upper`, `value`) are evaluated once, when the model is built, so they are numbers (or
values defined outside the block): they cannot depend on the state, the control, the variable or the
time.

```julia
@def begin
    t ∈ [0, 2], time
    x ∈ R², state
    u ∈ R, control
    x₂(t) ≥ 1                  # the expression x₂(t) is bounded below by 1
    -1 ≤ u(t) ≤ 1
    ...
end
```

## A bound that depends on the model

When a bound depends on the state, the control, the variable or the time, the parser raises a
`ParsingError` that names the bound, the cause, and a way to write the constraint. The suggestion is
built from the constraint itself and takes one of three shapes.

**The sides are in the wrong order.** `e ≤ b` bounds `e` above by `b`, so in `1 ≤ x₂(t)` the
constrained expression is `1`. Put the expression that depends on the model first:

```text
ParsingError
│  Line 7: 1 ≤ x₂(t)
│  The upper bound `x₂(t)` of this constraint depends on the state; a bound is evaluated once, when the model is built, so it cannot depend on the variable, the state, the control or the time
│
│  Hint  put the expression that depends on the state first: write `x₂(t) ≥ 1`
└─
```

**An equality with a term that depends on the model.** `x₂(0) == v` constrains `x₂(0)` to the value
`v`, which is a variable of the problem. Move the term to the constrained side:

```julia
x₂(0) - v == 0
```

**Both sides depend on the model.** Move the term to the constrained side as well, and compare with a
constant:

```julia
x₁(t) - u(t) ≤ 0           # rather than x₁(t) ≤ u(t)
```

These are constraints of a different kind in the model (a function of the state and of the control,
not a box constraint), which is the reason why they cannot be written as a bound.
