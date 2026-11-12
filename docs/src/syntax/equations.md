# Aliases and equations

## `=` defines an alias, `==` writes an equation

In a `@def` block, a line `name = expression` defines an **alias**: later lines can use `name`
in place of the expression.

```julia
@def begin
    t ∈ [0, 1], time
    x = (q, v) ∈ R², state
    u ∈ R, control
    q̇ = v(t)             # alias: q̇ stands for v(t)
    v̇ = u(t)
    ẋ(t) == [q̇, v̇]       # the dynamics is an equation: ==
    ...
end
```

Dynamics, constraints and boundary conditions are written with `==` (or `≤`, `≥`). An alias has a
name and no arguments.

## The common slip

Writing the dynamics with `=` instead of `==` is easy to do:

```julia
ẋ(t) = [q̇, v̇]       # ParsingError
```

Julia reads `ẋ(t) = …` as the short form of a function definition, which has no meaning in the
DSL. The parser refuses any statement of the form `f(args) = body` with a `ParsingError` that shows
the line as it was written and suggests the same line with `==`:

```text
ParsingError
│  Line 7: ẋ(t) = [q̇, v̇]
│  `=` defines an alias, which is a name without arguments; an equation or a constraint is written with `==`
│
│  Hint  write `ẋ(t) == [q̇, v̇]`
└─
```

This holds for the derivative of the whole state (`ẋ(t)`, `∂(x)(t)`, `derivative(x)(t)`), for a
component (`ẋ₁(t)`), and for any other equation such as a boundary condition `x(0) = 0`.

## An alias is defined once

An alias cannot be redefined: a second `c = …` after `c = …` is refused with a `ParsingError` that
names the alias, even when the value is the same.

```text
ParsingError
│  Line 5: c = 7
│  the alias `c` is already defined
│
│  Hint  rename the second alias, or remove the first one
└─
```

## Reading the line in an error

An error names the line of the block it comes from, `Line n: …`, printed as the statement was
written, without the internal comments Julia adds when it lowers a function definition. The aliases
that `@def` itself provides (`derivative` for `∂`, `in` for `∈`, `<=` for `≤`, …) are shown in their
substituted form.
