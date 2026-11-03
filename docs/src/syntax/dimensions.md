# Dimensions and automatic names

The dimension of the state, the control and the variable is given by the domain of the declaration:
`R` for a scalar, `R^n` for a vector of dimension `n`.

```julia
@def begin
    t ∈ [0, 1], time
    x ∈ R^3, state       # dimension 3
    u ∈ R², control      # dimension 2, same as R^2
    v ∈ R, variable      # dimension 1: a scalar
    ...
end
```

## Digit exponents

`R²`, `R³`, … are other spellings of `R^2`, `R^3`, …. They exist up to `R²⁰`; for a larger dimension
write `R^21`.

## Symbolic dimension: `R^n`

The exponent can be any expression whose value is known when the block is evaluated: a global, an
argument of the enclosing function, or an expression such as `n + 1`.

```julia
function problem(n)
    @def begin
        t ∈ [0, 1], time
        x ∈ R^n, state
        u ∈ R^(n + 1), control
        ...
    end
end
```

If `n` is not defined, Julia reports `UndefVarError: n not defined`.

### `Rⁿ` is not `R^n`

`Rⁿ` is a single identifier whose last letter is a superscript `n`. Only the digit exponents are
recognised, so a symbolic dimension cannot be read from it, and

```julia
x ∈ Rⁿ, state          # ParsingError: unknown syntax
```

is rejected on purpose. Write `x ∈ R^n, state`.

## Names of the components

By default the components of `x` are named `x₁`, `x₂`, …, and can be used in the block under these
names or as `x1`, `x2`, …; they are also accessible as `x[1]`, `x[2]`, …. A name given by the user,
`x = (q, v) ∈ R², state`, adds `q` and `v` for the same components (see
[Declarations](declarations.md)).

!!! warning "Limit of the automatic names"

    When the dimension is **a literal number** (`R^12`, `R¹²`), every component has its automatic
    names. When it is **symbolic** (`R^n`), the dimension is not known when the block is parsed, and
    only the components `x₁` to `x₉` (and `x1` to `x9`) are named. A component beyond the ninth is
    reached by its index, `x[10]`, or by a name given in the declaration. A name such as `x₁₀` is
    not an error at parse time: it is an undefined variable, and Julia reports it when the model is
    evaluated.
