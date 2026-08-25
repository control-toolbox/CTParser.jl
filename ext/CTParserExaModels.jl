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

Nothing here overwrites an ExaModels method. ExaModels' core defines arithmetic on
`AbstractNode` generically; every `Null`-specific method below is strictly more
specific, and adds the zero/one elimination that keeps a structural zero from
becoming a `Node2(*, Null(0), x)` left in the expression graph. Without them a
model still evaluates correctly — the redundant nodes just make the graph, and so
the derivative kernels, larger than they need to be.

# Lifetime

Temporary. When ExaModels declares its own extension, delete this file and drop the
two weak dependencies from `Project.toml`; nothing else references them. Aqua needs
no exemption — `Aqua.test_all(CTParser)` inspects the package module, not its
extensions, so `piracies=true` stays on as it is. See control-toolbox/CTParser.jl#325.
"""
module CTParserExaModels

using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra

# ============================================================================
# Null node pass-throughs for LinearAlgebra functions
#
# A Null wraps a constant, so a unary function of it is again a constant and can
# be folded now rather than left as a Node1 for the AD kernels to walk.
# ============================================================================

for f in [:inv, :abs, :sqrt, :cbrt, :abs2, :exp, :log, :sin, :cos, :tan]
    @eval @inline function Base.$f(x::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$f(x.value))
    end
end

# ============================================================================
# Scalar Null arithmetic with zero/one elimination
# ============================================================================

# Null op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(
        a::ExaModels.Null{T}, b::ExaModels.Null{S}
    ) where {T<:Real,S<:Real}
        return ExaModels.Null(Base.$op(a.value, b.value))
    end
end

# Null + AbstractNode / AbstractNode + Null (zero elimination)
@inline function Base.:+(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return b.value == zero(T) ? a : ExaModels.Node2(+, a, b)
end

# Null * AbstractNode / AbstractNode * Null (zero/one elimination)
@inline function Base.:*(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return if a.value == zero(T)
        ExaModels.Null(zero(T))
    elseif a.value == one(T)
        b
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return if b.value == zero(T)
        ExaModels.Null(zero(T))
    elseif b.value == one(T)
        a
    else
        ExaModels.Node2(*, a, b)
    end
end

# Null - AbstractNode / AbstractNode - Null
@inline function Base.:-(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return a.value == zero(T) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return b.value == zero(T) ? a : ExaModels.Node2(-, a, b)
end

# Null op Real / Real op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Real) where {T<:Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Real, b::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Null op Integer / Integer op Null — disambiguate Null{T} op Real vs AbstractNode op Integer
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Integer) where {T<:Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Integer, b::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Integer × AbstractNode zero/one elimination (more specific than core's Real × AbstractNode)
# Fixes: 0 * x → Null(0), 1 * x → x, 0 + x → x, etc.
@inline function Base.:*(a::Integer, b::ExaModels.AbstractNode)
    return if iszero(a)
        ExaModels.Null(zero(a))
    elseif isone(a)
        b
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::Integer)
    return if iszero(b)
        ExaModels.Null(zero(b))
    elseif isone(b)
        a
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:+(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? b : ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(+, a, b)
end
@inline function Base.:-(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(-, a, b)
end

# ----------------------------------------------------------------------------
# Upstream ambiguity, surfaced by the zero elimination above
#
# ExaModels declares both
#     _hdrpass_val(::Type{<:SecondAdjointNull}, ::Type)
#     _hdrpass_val(::Type, ::Type{<:SecondAdjointNull})
# and no method for the case where *both* arguments are `SecondAdjointNull`, so
# that call is ambiguous (simdfunction.jl:142-143). Folding a structural zero to
# `Null` is what makes both operands Null at once, so the ambiguity only shows up
# once the eliminations above are in play — it hits second-order derivatives of a
# dynamics written with `dot`, under the trapeze scheme.
#
# The value is forced, not a judgement call: both declared methods return `Val(0)`,
# so the intersection must too. This is the "Possible fix" Julia itself prints.
# Delete along with the rest of this file once upstream fixes it.
# ----------------------------------------------------------------------------

function ExaModels._hdrpass_val(
    ::Type{<:ExaModels.SecondAdjointNull}, ::Type{<:ExaModels.SecondAdjointNull}
)
    return Val(0)
end

# ============================================================================
# Type aliases, promotion, and adjoint for nodes
# ============================================================================

const ExaNode = Union{
    ExaModels.AbstractNode,ExaModels.AbstractAdjointNode,ExaModels.AbstractSecondAdjointNode
}
const VecExaNode = AbstractVector{<:ExaNode}
const MatExaNode = AbstractMatrix{<:ExaNode}

# Type promotion: [x, 0] should give Vector{AbstractNode} with Null(0), not Vector{Any}
Base.promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Real}) = ExaModels.AbstractNode
function Base.convert(::Type{ExaModels.AbstractNode}, x::Real)
    return iszero(x) ? zero(ExaModels.AbstractNode) : ExaModels.Null(x)
end

# zero/one for ExaNode types — needed by stdlib (e.g. tr) and general array ops
Base.zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(0)
Base.zero(::ExaNode) = ExaModels.Null(0)
Base.one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)
Base.one(::ExaNode) = ExaModels.Null(1)

# adjoint/transpose for scalar ExaNode — nodes are real-valued, so both are identity
Base.adjoint(x::ExaModels.AbstractNode) = x
Base.adjoint(x::ExaModels.AbstractAdjointNode) = x
Base.adjoint(x::ExaModels.AbstractSecondAdjointNode) = x
Base.transpose(x::ExaModels.AbstractNode) = x
Base.transpose(x::ExaModels.AbstractAdjointNode) = x
Base.transpose(x::ExaModels.AbstractSecondAdjointNode) = x

# adjoint/transpose for matrices of ExaNode — materialize to plain Matrix
Base.adjoint(A::MatExaNode) = [A[j, i] for i in axes(A, 2), j in axes(A, 1)]
Base.transpose(A::MatExaNode) = [A[j, i] for i in axes(A, 2), j in axes(A, 1)]

# Dispatch pair constants for 3-way type combos
const _VEC_PAIRS = [
    (VecExaNode, VecExaNode),
    (AbstractVector{<:Real}, VecExaNode),
    (VecExaNode, AbstractVector{<:Real}),
]
const _MAT_PAIRS = [
    (MatExaNode, MatExaNode),
    (AbstractMatrix{<:Real}, MatExaNode),
    (MatExaNode, AbstractMatrix{<:Real}),
]

# ============================================================================
# Plain Julia scalar decompositions
#
# Every operation below rebuilds the result element by element from scalar node
# arithmetic. Nodes are expression graph vertices, not numbers, so the stdlib's
# BLAS-backed paths do not apply.
# ============================================================================

# --- sum ---

function Base.sum(v::VecExaNode)
    s = v[1]
    for i in 2:length(v)
        s = s + v[i]
    end
    return s
end

# --- dot ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.dot(a::$T1, b::$T2)
        @assert length(a) == length(b)
        s = a[1] * b[1]
        for i in 2:length(a)
            s = s + a[i] * b[i]
        end
        return s
    end
end

# --- scalar * vector ---

for (T1, T2) in
    [(Real, VecExaNode), (ExaNode, AbstractVector{<:Real}), (ExaNode, VecExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i] for i in eachindex(b)]
end
for (T1, T2) in
    [(VecExaNode, Real), (AbstractVector{<:Real}, ExaNode), (VecExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i] * b for i in eachindex(a)]
end

# --- scalar * matrix ---

for (T1, T2) in
    [(Real, MatExaNode), (ExaNode, AbstractMatrix{<:Real}), (ExaNode, MatExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i, j] for i in axes(b, 1), j in axes(b, 2)]
end
for (T1, T2) in
    [(MatExaNode, Real), (AbstractMatrix{<:Real}, ExaNode), (MatExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i, j] * b for i in axes(a, 1), j in axes(a, 2)]
end

# --- matrix * vector (inline dot to avoid dispatch issues with view types) ---

function _dot_row(A, i, x)
    n = size(A, 2)
    s = A[i, 1] * x[1]
    for j in 2:n
        s = s + A[i, j] * x[j]
    end
    return s
end

for (T1, T2) in [
    (MatExaNode, VecExaNode),
    (AbstractMatrix{<:Real}, VecExaNode),
    (MatExaNode, AbstractVector{<:Real}),
]
    @eval function Base.:*(A::$T1, x::$T2)
        m = size(A, 1)
        @assert size(A, 2) == length(x)
        return [_dot_row(A, i, x) for i in 1:m]
    end
end

# --- matrix * matrix (inline dot to avoid dispatch issues) ---

function _dot_col(A, i, B, j)
    n = size(A, 2)
    s = A[i, 1] * B[1, j]
    for k in 2:n
        s = s + A[i, k] * B[k, j]
    end
    return s
end

for (T1, T2) in _MAT_PAIRS
    @eval function Base.:*(A::$T1, B::$T2)
        @assert size(A, 2) == size(B, 1)
        m, n = size(A, 1), size(B, 2)
        return [_dot_col(A, i, B, j) for i in 1:m, j in 1:n]
    end
end

# --- vector +/- ---

for op in (:+, :-)
    for (T1, T2) in _VEC_PAIRS
        @eval function Base.$op(a::$T1, b::$T2)
            @assert length(a) == length(b)
            return [$op(a[i], b[i]) for i in eachindex(a)]
        end
    end
end

# Win dispatch over Base's +(::Array, ::Array...) from arraymath.jl
for (T1, T2) in [(ExaNode, ExaNode), (Real, ExaNode), (ExaNode, Real)]
    @eval function Base.:+(a::Array{<:$T1,1}, b::Array{<:$T2,1})
        @assert length(a) == length(b)
        return [a[i] + b[i] for i in eachindex(a)]
    end
    @eval function Base.:+(A::Array{<:$T1,2}, B::Array{<:$T2,2})
        @assert size(A) == size(B)
        return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
    end
end

# Unary minus for vector/matrix of nodes
Base.:-(a::VecExaNode) = [-a[i] for i in eachindex(a)]
Base.:-(A::MatExaNode) = [-A[i, j] for i in axes(A, 1), j in axes(A, 2)]

# --- matrix +/- ---

for op in (:+, :-)
    for (T1, T2) in _MAT_PAIRS
        @eval function Base.$op(A::$T1, B::$T2)
            @assert size(A) == size(B)
            return [$op(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
        end
    end
end

# --- tr ---

function _tr_impl(A)
    @assert size(A, 1) == size(A, 2) "Matrix must be square for tr"
    n = size(A, 1)
    s = A[1, 1]
    for i in 2:n
        s = s + A[i, i]
    end
    return s
end

LinearAlgebra.tr(A::MatExaNode) = _tr_impl(A)
# More specific methods to win dispatch over stdlib's tr(::Matrix{T}) (Julia 1.10)
# and tr(::StridedMatrix{T}) (Julia 1.12+)
LinearAlgebra.tr(A::Matrix{<:ExaNode}) = _tr_impl(A)
LinearAlgebra.tr(A::StridedMatrix{<:ExaNode}) = _tr_impl(A)

# --- diag ---

function LinearAlgebra.diag(A::MatExaNode)
    n = minimum(size(A))
    return [A[i, i] for i in 1:n]
end

# --- diagm ---

function LinearAlgebra.diagm(v::VecExaNode)
    n = length(v)
    T = typeof(v[1])
    M = Matrix{Union{T,ExaModels.Null{Int}}}(undef, n, n)
    for i in 1:n, j in 1:n
        if i == j
            M[i, j] = v[i]
        else
            M[i, j] = ExaModels.Null(0)
        end
    end
    return M
end

# --- adjoint/transpose operations ---

# v' * w = dot(v, w)
for (TA, TV, TB) in [
    (ExaNode, VecExaNode, VecExaNode),
    (ExaNode, VecExaNode, AbstractVector{<:Real}),
    (Real, AbstractVector{<:Real}, VecExaNode),
]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA,<:$TV}, b::$TB)
        return LinearAlgebra.dot(parent(a), b)
    end
end

# v' * A
for (TA, TV, TB) in
    [(ExaNode, VecExaNode, MatExaNode), (Real, AbstractVector{<:Real}, MatExaNode)]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA,<:$TV}, B::$TB)
        v = parent(a)
        @assert length(v) == size(B, 1)
        n = size(B, 2)
        return adjoint([LinearAlgebra.dot(v, [B[k, j] for k in 1:size(B, 1)]) for j in 1:n])
    end
end
function Base.:*(
    a::LinearAlgebra.Adjoint{<:ExaNode,<:VecExaNode}, B::AbstractMatrix{<:Real}
)
    v = parent(a)
    @assert length(v) == size(B, 1)
    n = size(B, 2)
    return adjoint([LinearAlgebra.dot(v, view(B, :, j)) for j in 1:n])
end

# ============================================================================
# Optimized scalar expansions
#
# det and norm expand to a single scalar expression graph. The small-size det
# cases are written out rather than recursed so the resulting graph stays flat.
# ============================================================================

# --- det (specialized for small sizes) ---

_det_1x1(A) = A[1, 1]

# 2x2: a11*a22 - a12*a21
function _det_2x2(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a21 = A[2, 1]
    a22 = A[2, 2]
    return a11 * a22 - a12 * a21
end

# 3x3: Sarrus' rule (optimized expansion)
function _det_3x3(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a21 = A[2, 1]
    a22 = A[2, 2]
    a23 = A[2, 3]
    a31 = A[3, 1]
    a32 = A[3, 2]
    a33 = A[3, 3]
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31)
end

# 4x4: cofactor expansion along first row
function _det_4x4(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a14 = A[1, 4]
    m11 =
        A[2, 2] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) +
        A[2, 4] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2])
    m12 =
        A[2, 1] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1])
    m13 =
        A[2, 1] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    m14 =
        A[2, 1] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1]) +
        A[2, 3] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    return a11 * m11 - a12 * m12 + a13 * m13 - a14 * m14
end

# General determinant via cofactor expansion (recursive, for N > 4)
function _det_recursive(A)
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    if n == 1
        return _det_1x1(A)
    elseif n == 2
        return _det_2x2(A)
    elseif n == 3
        return _det_3x3(A)
    elseif n == 4
        return _det_4x4(A)
    end
    s = A[1, 1] * _det_recursive(A[2:end, 2:end])
    for j in 2:n
        cols = vcat(1:(j - 1), (j + 1):n)
        minor = _det_recursive(A[2:end, cols])
        if iseven(j)
            s = s - A[1, j] * minor
        else
            s = s + A[1, j] * minor
        end
    end
    return s
end

# Dispatch det for matrices containing ExaNode elements
function LinearAlgebra.det(A::MatExaNode)
    @assert size(A, 1) == size(A, 2) "Matrix must be square for det"
    return _det_recursive(A)
end

# --- norm ---

# 2-norm for vectors of nodes: sqrt(sum(xi^2))
function LinearAlgebra.norm(v::VecExaNode)
    s = v[1]^2
    for i in 2:length(v)
        s = s + v[i]^2
    end
    return sqrt(s)
end

# p-norm for vectors of nodes: (sum(abs(xi)^p))^(1/p)
function LinearAlgebra.norm(v::VecExaNode, p::Real)
    if p == 2
        return LinearAlgebra.norm(v)
    elseif p == 1
        s = abs(v[1])
        for i in 2:length(v)
            s = s + abs(v[i])
        end
        return s
    elseif p == Inf
        error("Inf-norm is not differentiable and not supported for ExaNode vectors")
    else
        s = abs(v[1])^p
        for i in 2:length(v)
            s = s + abs(v[i])^p
        end
        return s^(1 / p)
    end
end

# Frobenius norm for matrices of nodes: sqrt(sum(aij^2))
function LinearAlgebra.norm(A::MatExaNode)
    s = A[1, 1]^2
    for j in axes(A, 2), i in axes(A, 1)
        (i == 1 && j == 1) && continue
        s = s + A[i, j]^2
    end
    return sqrt(s)
end

# --- cross product (3D only) ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.cross(a::$T1, b::$T2)
        @assert length(a) == 3 && length(b) == 3 "Cross product requires 3D vectors"
        return [
            a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]
        ]
    end
end

end # module CTParserExaModels
