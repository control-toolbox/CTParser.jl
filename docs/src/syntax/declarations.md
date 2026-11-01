# Declaring the state, the control and the variable

A `@def` block declares the time, the state, the control and, optionally, a variable. This page
describes how the dimension and the names of the components are written, and what the parser
refuses.

```julia
@def begin
    t ∈ [0, 1], time
    x ∈ R², state        # dimension 2, components named x₁ and x₂ by default
    u ∈ R, control       # dimension 1
    ...
end
```

## Dimension

| Declaration | Dimension | Name | Components |
| --- | --- | --- | --- |
| `u ∈ R, control` | 1 | `u` | `u` |
| `u ∈ R^m, control` (or `R²`, `R³`, …) | `m` | `u` | `u₁`, …, `uₘ` |

A one-dimensional state, control or variable is a scalar: it is declared with `∈ R`, and read as a
number in the model, not as a vector of length one.

## Naming the components

The components of an `n`-dimensional element are named with a tuple, and the dimension is given as
`R^n`:

```julia
x = (q, v) ∈ R², state
u = (a, b, c) ∈ R^3, control
w = [λ₁, tf] ∈ R^2, variable
```

The name of the element (`x`, `u`, `w`) is the left-hand side; the names of the components are the
ones in the tuple. In the model, they are what `CTModels.state_name`, `state_components`, and their
control and variable counterparts return, and what the plots use to label the curves.

## What is refused

A declaration of the form `name = (…) ∈ domain, kind` whose domain is not `R^n` is refused with a
`ParsingError`:

```julia
u = (z,) ∈ R, control      # ParsingError
u = (z) ∈ R, control       # ParsingError
x = (q,) ∈ R, state        # ParsingError
w = (a, b) ∈ R, variable   # ParsingError
```

Before this check, such a line was read as an alias (`u` standing for the tuple), nothing was
declared, and the problem was built without any control. The error now names the line and says what
to write:

- for a one-dimensional element, `u ∈ R, control`; its only component is named `u`;
- for an `n`-dimensional element, `u = (name₁, …, nameₙ) ∈ R^n, control`.

### Why not name a single component?

A name for the element and a second name for its only component would be two names for one scalar.
In the model and in the plots the curve of a scalar control is labelled with the name of the
component, so `u = (z,) ∈ R, control` would label it `z` and group it under `u`, with no gain over
`z ∈ R, control`. Until a use appears, the parser does not accept this notation; the issue
[#4](https://github.com/control-toolbox/CTParser.jl/issues/4) records the decision and can be
reopened.

!!! note

    `u = (z,) ∈ R^1, control` is accepted: it declares a one-dimensional control named `u` whose
    component is named `z`. It is equivalent to a scalar declaration with an extra name, and is not
    the recommended form.
