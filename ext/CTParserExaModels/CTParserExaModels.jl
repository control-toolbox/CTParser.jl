"""
    CTParserExaModels

Linear-algebra glue for ExaModels expression nodes, needed by the code `@def` emits
for the `:exa` backend.

# Why this exists

A user writing an optimal control problem for the `:exa` backend may use ordinary
linear algebra on the state and control:

```julia
@def begin
    ∂(x)(t) == A * x(t) + B * u(t)      # matrix × vector
    ∫(dot(q, x(t))) → min               # dot on a vector of nodes
end
```

`@def` substitutes ExaModels expression nodes for `x(t)` and `u(t)`, so those
operators are applied to `ExaModels.AbstractNode` values rather than numbers.
ExaModels does not define them in its core module.

Up to ExaModels 0.9 they came from `ExaModelsLinearAlgebra`, an extension triggered
by `LinearAlgebra`. From 0.11 that extension was renamed `ExaModelsOptimalControl`
and re-triggered on `OptimalControl` — but it is **not declared in `[extensions]`**,
and `OptimalControl` is not in `[weakdeps]` either, so Julia never loads it. The
file still ships in the tarball, and still works verbatim against 0.12; it is only
unwired. `LinearAlgebra` remains in ExaModels' `[weakdeps]` with nothing referencing
it, which is the fingerprint of the same slip. This is true of ExaModels `main` as
well as the 0.12.0 release.

# Provenance and scope

The method definitions below are ported from that shipped-but-unregistered
`ext/ExaModelsOptimalControl.jl`, minus one section:

- Upstream's Section F (`ExaModels.add_con(core, ::AbstractVector)`) is not ported:
  it is broken as written (it starts from `c1 = nothing` and calls the removed
  `ExaModels.constraint` on it), and `p_constraint_exa!` never emits the vector
  form — it loops over components instead.

Deliberate departures from the upstream file:

- dimension and shape checks throw typed `CTBase` errors instead of `@assert` / `error`;
- `Null(0) + x` folds to `x`, like `x + Null(0)` already did;
- the square-matrix check of `det` is done once, not at every level of the recursion;
- `Null op Constant` has its own methods where the `Null` rules met ExaModels' `Constant`
  folding rules (control-toolbox/CTParser.jl#364);
- the array methods take *strided* arrays of nodes (`StridedVector{T} where T<:ExaNode`),
  not any `AbstractVector` of nodes: an `Abstract` signature is ambiguous with the methods
  other packages define on their own arrays, which are the constants users write
  (`Diagonal`, sparse, `SMatrix`). The constants keep `AbstractVector{<:Real}` /
  `AbstractMatrix{<:Real}`, and the collisions left there have methods on the exact
  intersection: here for LinearAlgebra, in `CTParserExaModelsSparseArrays` and
  `CTParserExaModelsStaticArrays` for the other two (control-toolbox/CTParser.jl#365);
- `tr` is left to LinearAlgebra, whose own method only needs `zero` and `+` on nodes.

Nothing here overwrites an ExaModels method. ExaModels' core defines arithmetic on
`AbstractNode` generically; every `Null`-specific method below is strictly more
specific, and adds the zero/one elimination that keeps a structural zero from
becoming a `Node2(*, Null(0), x)` left in the expression graph. Without them a
model still evaluates correctly — the redundant nodes just make the graph, and so
the derivative kernels, larger than they need to be.

# Files

`types.jl` (what an array of nodes is, promotion, `zero` / `one`, adjoint), `scalar.jl`
(arithmetic on single nodes), `checks.jl` (typed errors), `arrays.jl` (linear algebra on
arrays of nodes) and `expansions.jl` (`det`, `norm`, `cross` as one scalar graph). Two
companion extensions complete it for sparse and static constants:
`CTParserExaModelsSparseArrays` and `CTParserExaModelsStaticArrays`.

# Lifetime and piracy

Meant to be temporary: when ExaModels declares its own extension, delete this directory,
the two companion extensions and their weak dependencies (control-toolbox/CTParser.jl#325).
Upstream has not said when (madsuite-org/ExaModels.jl#323), so the code is kept to the
conventions in the meantime, and its type piracy is accepted as long as needed
(control-toolbox/CTParser.jl#365): CTParser owns neither `Base`, `LinearAlgebra` nor the
ExaModels node types. Every pirated method is listed with its reason in
`test/extension_piracies.jl`, and the ct-conventions TST8 check (`test/test_aqua.jl`) fails
on an unlisted one and on any ambiguity. If ExaModels ever declares its own
`ExaModelsOptimalControl` extension, both would define the same methods: the sentinel in
`test/test_exa_linalg.jl` fails as soon as it does.
"""
module CTParserExaModels

using CTBase: CTBase
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra

include("scalar.jl")
include("types.jl")
include("checks.jl")
include("arrays.jl")
include("expansions.jl")

end # module CTParserExaModels
