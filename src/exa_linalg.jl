"""
    ExaLinAlg

Module providing trait-based linear algebra extensions for Array{<:ExaModels.AbstractNode}.
Extends Julia's standard Array interface without wrappers.

Supports operations on Vector, Matrix, and their standard wrappers:
- SubArray (views created by @view or slicing)
- ReshapedArray (created by reshape)
- ReinterpretArray (created by reinterpret, only for Real element types)

# Key Design: Direct Operator Overloading on Null Nodes
All operations use direct operator overloads (+, -, *) on Null types.
These specialized methods properly handle Null nodes, preserving constants
and avoiding unnecessary expression tree nodes.

# Public API (Exported)
- Canonical nodes: `zero`, `one`, `zeros`, `ones`
- Basic operations: `zero`, `one`, `adjoint`, `transpose`, `*`, `+`, `-`, `sum`
- Linear algebra: `dot`, `det`, `tr`, `norm`, `diag`, `diagm`
"""
module ExaLinAlg

using ExaModels: ExaModels
using LinearAlgebra

import Base: zero, one, adjoint, *, promote_rule, convert, +, -, transpose, sum
import Base: inv, abs, sqrt, cbrt, abs2, exp, exp2, exp10, log, log2, log10, log1p
import Base: sin, cos, tan, csc, sec, cot, asin, acos, atan, acot
import Base: sind, cosd, tand, cscd, secd, cotd, atand, acotd
import Base: sinh, cosh, tanh, csch, sech, coth, asinh, acosh, atanh, acoth
import Base: ^, zeros, ones
import LinearAlgebra: dot, Adjoint, det, tr, norm, diag, diagm

export zero, one, zeros, ones, adjoint, transpose, *, +, -, sum, dot, det, tr, norm, diag, diagm
export inv, abs, sqrt, cbrt, abs2, exp, exp2, exp10, log, log2, log10, log1p
export sin, cos, tan, csc, sec, cot, asin, acos, atan, acot
export sind, cosd, tand, cscd, secd, cotd, atand, acotd
export sinh, cosh, tanh, csch, sech, coth, asinh, acosh, atanh, acoth
export ^

# ============================================================================
# Type Aliases for Array Wrappers
# ============================================================================
#
# Union types enumerating concrete array wrapper types we support.
# This avoids ambiguities with Base methods (including SparseArrays) while
# being more general than plain Vector/Matrix types.
#
# We restrict SubArray to views of DenseArray to avoid conflicts with sparse arrays.
#
# Vec1DReal: 1D arrays of Real (includes ReinterpretArray from Complex)
# Vec1DNode: 1D arrays of AbstractNode (no ReinterpretArray)
# Mat2D: 2D arrays (both Real and AbstractNode)
# ============================================================================

# SubArray where parent is a DenseArray (excludes sparse arrays)
const DenseSubArray{T,N} = SubArray{T,N,<:DenseArray}

# ReshapedArray where parent is a DenseArray (excludes sparse arrays)
const DenseReshapedArray{T,N} = Base.ReshapedArray{T,N,<:DenseArray}

const Vec1DReal{T} = Union{Vector{T}, DenseSubArray{T,1}, DenseReshapedArray{T,1}, Base.ReinterpretArray{T,1}}
const Vec1DNode{T} = Union{Vector{T}, DenseSubArray{T,1}, DenseReshapedArray{T,1}}
const Mat2D{T} = Union{Matrix{T}, DenseSubArray{T,2}, DenseReshapedArray{T,2}}

# ============================================================================
# Section 1: Canonical Nodes (zero and one)
# ============================================================================
#
# Canonical encodings:
# - Zero: Null(0) represents zero
# - One:  Null(1) represents one
#
# We extend Base.zero and Base.one for both type-based and instance-based calls:
# - zero(ExaModels.AbstractNode) or zero(::Type{<:ExaModels.AbstractNode})
# - one(ExaModels.AbstractNode) or one(::Type{<:ExaModels.AbstractNode})
# ============================================================================

"""
    zero(::Type{<:ExaModels.AbstractNode}) -> Null{Int}

Return the canonical zero AbstractNode: Null(0).
"""
zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(0)

"""
    zero(::ExaModels.AbstractNode) -> Null{Int}

Return the canonical zero AbstractNode: Null(0).
"""
zero(::ExaModels.AbstractNode) = ExaModels.Null(0)

"""
    one(::Type{<:ExaModels.AbstractNode}) -> Null{Int}

Return the canonical one AbstractNode: Null(1).
"""
one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)

"""
    one(::ExaModels.AbstractNode) -> Null{Int}

Return the canonical one AbstractNode: Null(1).
"""
one(::ExaModels.AbstractNode) = ExaModels.Null(1)

"""
    zeros(::Type{T}, dims...) where {T <: ExaModels.AbstractNode}

Create an array of AbstractNode zeros with the specified dimensions.
Uses fill with the canonical zero node: Null(0).
"""
function zeros(::Type{T}, dims::Integer...) where {T <: ExaModels.AbstractNode}
    return fill(zero(T), dims...)
end

"""
    ones(::Type{T}, dims...) where {T <: ExaModels.AbstractNode}

Create an array of AbstractNode ones with the specified dimensions.
Uses fill with the canonical one node: Null(1).
"""
function ones(::Type{T}, dims::Integer...) where {T <: ExaModels.AbstractNode}
    return fill(one(T), dims...)
end

# ============================================================================
# Section 2: Scalar Operations on Null Nodes
# ============================================================================
#
# Direct operator overloads for Null nodes (more specific than ExaModels' AbstractNode methods).
# These preserve Null when operating on constants and handle special cases like 0 * x = Null(0).
#
# Rules for +:
#   Null(x) + Null(y) = Null(x + y)
#   Null(x) + e = x + e  (unwrap to scalar, use ExaModels native +)
#   e + Null(x) = e + x
#
# Rules for -:
#   Null(x) - Null(y) = Null(x - y)
#   Null(0) - e = -e
#   Null(x) - e = x - e when !iszero(x)
#   e - Null(x) = e - x
#
# Rules for *:
#   Null(x) * Null(y) = Null(x * y)
#   Null(0) * e = Null(0)  (zero optimization)
#   e * Null(0) = Null(0)
#   Null(x) * e = x * e when !iszero(x)
#   e * Null(x) = e * x when !iszero(x)
# ============================================================================

# Addition: Null{T} + Null{S} → Null
+(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value + y.value)
# Addition: Null{T} + AbstractNode → unwrap Null, use native +
+(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = x.value + y
+(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x + y.value
# Addition: Null{T} + Real → Null (more specific than ExaModels' AbstractNode + Real)
+(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value + y)
+(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x + y.value)

# Subtraction: Null{T} - Null{S} → Null
-(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value - y.value)
# Subtraction: Null{T} - AbstractNode → handle 0 - e = -e specially
-(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? (-y) : (x.value - y)
-(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x - y.value
# Subtraction: Null{T} - Real → Null (more specific than ExaModels' AbstractNode - Real)
-(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value - y)
-(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x - y.value)

# Multiplication: Null{T} * Null{S} → Null
*(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value * y.value)
# Multiplication: Null{T} * AbstractNode → zero optimization: 0 * e = Null(0)
*(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? ExaModels.Null(0) : (x.value * y)
*(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = iszero(y.value) ? ExaModels.Null(0) : (x * y.value)
# Multiplication: Null{T} * Real → zero optimization, more specific than ExaModels' AbstractNode * Real
*(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value * y)
*(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x * y.value)

# ============================================================================
# Section 2.5: Unary Functions on Null Nodes
# ============================================================================
#
# Apply unary functions to the inner value of Null nodes.
# This fixes broadcasting: cos.([Null(x)]) → [Null(cos(x))] instead of [cos(Null(x))]
#
# Pattern: f(x::Null{T}) = Null(f(x.value))
# ============================================================================

# Unary operators
-(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(-x.value)
+(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(+x.value)

# Basic functions
inv(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(inv(x.value))
abs(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(abs(x.value))
sqrt(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sqrt(x.value))
cbrt(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cbrt(x.value))
abs2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(abs2(x.value))

# Exponential and logarithmic functions
exp(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp(x.value))
exp2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp2(x.value))
exp10(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp10(x.value))
log(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log(x.value))
log2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log2(x.value))
log10(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log10(x.value))
log1p(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log1p(x.value))

# Trigonometric functions
sin(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sin(x.value))
cos(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cos(x.value))
tan(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tan(x.value))
csc(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(csc(x.value))
sec(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sec(x.value))
cot(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cot(x.value))
asin(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(asin(x.value))
acos(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acos(x.value))
atan(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atan(x.value))
acot(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acot(x.value))

# Degree-based trigonometric functions
sind(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sind(x.value))
cosd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cosd(x.value))
tand(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tand(x.value))
cscd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cscd(x.value))
secd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(secd(x.value))
cotd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cotd(x.value))
atand(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atand(x.value))
acotd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acotd(x.value))

# Hyperbolic functions
sinh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sinh(x.value))
cosh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cosh(x.value))
tanh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tanh(x.value))
csch(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(csch(x.value))
sech(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sech(x.value))
coth(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(coth(x.value))
asinh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(asinh(x.value))
acosh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acosh(x.value))
atanh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atanh(x.value))
acoth(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acoth(x.value))

# Power function (unary form: x^n is actually binary, but we handle Null^Real)
^(x::ExaModels.Null{T}, n::Real) where {T<:Real} = ExaModels.Null(x.value^n)
^(x::ExaModels.Null{T}, n::Integer) where {T<:Real} = ExaModels.Null(x.value^n)

# ============================================================================
# Section 3: sum (using direct + operator)
# ============================================================================

"""
    sum(arr::AbstractArray{<:ExaModels.AbstractNode})

Optimized sum for arrays of AbstractNode using the overloaded + operator.
Returns zero(ExaModels.AbstractNode) if the array is empty.
"""
function sum(arr::AbstractArray{<:ExaModels.AbstractNode})
    result = zero(ExaModels.AbstractNode)
    for x in arr
        result = result + x
    end
    return result
end

# ============================================================================
# Section 3.5: Basic Type Conversions and Promotions
# ============================================================================

# Scalar operations
adjoint(x::ExaModels.AbstractNode) = x
transpose(x::ExaModels.AbstractNode) = x

convert(::Type{ExaModels.AbstractNode}, x::Real) = iszero(x) ? zero(ExaModels.AbstractNode) : ExaModels.Null(x)

promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Real}) = ExaModels.AbstractNode

# ============================================================================
# Section 4: Dot Product (using *, sum)
# ============================================================================
#
# Note: We wrap Real values in Null before multiplication to ensure our
# optimized * (with zero handling) is called instead of ExaModels' native *.
# ============================================================================

function dot(v::Vec1DReal{<:Real}, w::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return sum(ExaModels.Null(v[i]) * w[i] for i in eachindex(v))
end

function dot(v::Vec1DNode{T}, w::Vec1DReal{<:Real}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return sum(v[i] * ExaModels.Null(w[i]) for i in eachindex(v))
end

function dot(v::Vec1DNode{T}, w::Vec1DNode{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return sum(v[i] * w[i] for i in eachindex(v))
end

# ============================================================================
# Section 5: Scalar × Vector/Matrix Multiplication (using *)
# ============================================================================
#
# Note: We wrap Real values in Null before multiplication to ensure our
# optimized * (with zero handling) is called instead of ExaModels' native *.
# ============================================================================

# Scalar × Vector
function *(a::T, v::Vec1DReal{<:Real}) where {T <: ExaModels.AbstractNode}
    return [a * ExaModels.Null(vi) for vi in v]
end

function *(a::Real, v::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    return [ExaModels.Null(a) * vi for vi in v]
end

function *(a::T, v::Vec1DNode{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [a * vi for vi in v]
end

# Vector × Scalar
function *(v::Vec1DNode{T}, a::Real) where {T <: ExaModels.AbstractNode}
    return [vi * ExaModels.Null(a) for vi in v]
end

function *(v::Vec1DReal{T}, a::S) where {T <: Real, S <: ExaModels.AbstractNode}
    return [ExaModels.Null(vi) * a for vi in v]
end

function *(v::Vec1DNode{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [vi * a for vi in v]
end

# Scalar × Matrix
function *(a::T, A::Mat2D{<:Real}) where {T <: ExaModels.AbstractNode}
    return [a * ExaModels.Null(A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::Real, A::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    return [ExaModels.Null(a) * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::T, A::Mat2D{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [a * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix × Scalar
function *(A::Mat2D{T}, a::Real) where {T <: ExaModels.AbstractNode}
    return [A[i, j] * ExaModels.Null(a) for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Mat2D{T}, a::S) where {T <: Real, S <: ExaModels.AbstractNode}
    return [ExaModels.Null(A[i, j]) * a for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Mat2D{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [A[i, j] * a for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 6: Matrix × Vector Product (uses dot)
# ============================================================================

function *(A::Mat2D{<:Real}, x::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Mat2D{T}, x::Vec1DReal{<:Real}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Mat2D{T}, x::Vec1DNode{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

# ============================================================================
# Section 7: Matrix × Matrix Product (uses dot)
# ============================================================================

function *(A::Mat2D{<:Real}, B::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Mat2D{T}, B::Mat2D{<:Real}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Mat2D{T}, B::Mat2D{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

# ============================================================================
# Section 8: Adjoint Vector × Matrix Product
# ============================================================================

function *(p::Adjoint{T, <:Vec1DNode{T}}, A::Mat2D{<:Real}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, <:Vec1DReal{T}}, A::Mat2D{S}) where {T <: Real, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, <:Vec1DNode{T}}, A::Mat2D{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

# ============================================================================
# Section 9: Adjoint and Transpose for Matrices
# ============================================================================

function adjoint(A::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

function transpose(A::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

# ============================================================================
# Section 10: Vector/Matrix Addition (using +)
# ============================================================================

# Vector + Vector
function +(v::Vec1DNode{T}, w::Vec1DReal{<:Real}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] + w[i] for i in eachindex(v)]
end

function +(v::Vec1DReal{<:Real}, w::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] + w[i] for i in eachindex(v)]
end

function +(v::Vec1DNode{T}, w::Vec1DNode{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] + w[i] for i in eachindex(v)]
end

# Matrix + Matrix
function +(A::Mat2D{T}, B::Mat2D{<:Real}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Mat2D{<:Real}, B::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Mat2D{T}, B::Mat2D{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 11: Vector/Matrix Subtraction (using -)
# ============================================================================

# Vector - Vector
function -(v::Vec1DNode{T}, w::Vec1DReal{<:Real}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] - w[i] for i in eachindex(v)]
end

function -(v::Vec1DReal{<:Real}, w::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] - w[i] for i in eachindex(v)]
end

function -(v::Vec1DNode{T}, w::Vec1DNode{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [v[i] - w[i] for i in eachindex(v)]
end

# Matrix - Matrix
function -(A::Mat2D{T}, B::Mat2D{<:Real}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Mat2D{<:Real}, B::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Mat2D{T}, B::Mat2D{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 12: Determinant (using +, -, *)
# ============================================================================

# Helper function to compute determinant (shared implementation)
function _det_impl(A)
    n, m = size(A)
    @assert n == m "Determinant is only defined for square matrices, got $(n)×$(m)"

    if n == 1
        return A[1, 1]
    elseif n == 2
        return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    elseif n == 3
        # Sarrus rule for 3×3 matrices
        pos1 = A[1, 1] * A[2, 2] * A[3, 3]
        pos2 = A[1, 2] * A[2, 3] * A[3, 1]
        pos3 = A[1, 3] * A[2, 1] * A[3, 2]
        neg1 = A[1, 3] * A[2, 2] * A[3, 1]
        neg2 = A[1, 1] * A[2, 3] * A[3, 2]
        neg3 = A[1, 2] * A[2, 1] * A[3, 3]
        pos_sum = pos1 + pos2 + pos3
        neg_sum = neg1 + neg2 + neg3
        return pos_sum - neg_sum
    else
        # Laplace expansion for n×n matrices (n ≥ 4)
        d = A[1, 1] * det(A[2:end, 2:end])  # Initialize with first term
        for j in 2:n
            minor = [A[2:end, 1:j-1] A[2:end, j+1:end]]
            sign_coeff = iseven(j) ? -1 : 1
            cofactor = sign_coeff * A[1, j] * det(minor)
            d = d + cofactor
        end
        return d
    end
end

# Separate methods for each wrapper type
function det(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return _det_impl(A)
end

# Note: We don't define det for SubArray to avoid potential ambiguities.
# Users should collect/copy the view first if needed.

function det(A::DenseReshapedArray{T,2}) where {T <: ExaModels.AbstractNode}
    return _det_impl(A)
end

# ============================================================================
# Section 13: Trace (using sum)
# ============================================================================

# Need separate methods to avoid ambiguity with LinearAlgebra.tr(::StridedMatrix)
function tr(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Trace is only defined for square matrices, got $(n)×$(m)"
    return sum(A[i, i] for i in 1:n)
end

# Note: We don't define tr for SubArray because it creates unavoidable ambiguity
# with LinearAlgebra.tr(::StridedMatrix). Users should collect/copy the view first.

function tr(A::DenseReshapedArray{T,2}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Trace is only defined for square matrices, got $(n)×$(m)"
    return sum(A[i, i] for i in 1:n)
end

# ============================================================================
# Section 14: Norms (using sum, *)
# ============================================================================

# Euclidean norm (2-norm) for vectors
function norm(v::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(vi * vi for vi in v))
end

# p-norm for vectors
function norm(v::Vec1DNode{T}, p::Real) where {T <: ExaModels.AbstractNode}
    if p == Inf
        # Infinity norm: max|vᵢ|
        error("Infinity norm not supported for symbolic AbstractNode vectors")
    elseif p == 1
        # 1-norm: sum of absolute values
        return sum(abs(vi) for vi in v)
    elseif p == 2
        # 2-norm: Euclidean norm
        return sqrt(sum(vi * vi for vi in v))
    else
        # General p-norm
        return sum(abs(vi)^p for vi in v)^(1/p)
    end
end

# Frobenius norm for matrices
function norm(A::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(A[i, j] * A[i, j] for i in axes(A, 1), j in axes(A, 2)))
end

# ============================================================================
# Section 15: Diagonal Operations
# ============================================================================

# Extract diagonal from matrix
function diag(A::Mat2D{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    k = min(n, m)
    return [A[i, i] for i in 1:k]
end

# Create diagonal matrix from vector
function diagm(v::Vec1DNode{T}) where {T <: ExaModels.AbstractNode}
    n = length(v)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = (i == j) ? v[i] : zero(ExaModels.AbstractNode)
    end
    return D
end

# diagm with pairs (more general form)
function diagm(kv::Pair{<:Integer, <:Vec1DNode{T}}) where {T <: ExaModels.AbstractNode}
    k, v = kv
    n = length(v) + abs(k)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = zero(ExaModels.AbstractNode)
    end
    if k >= 0
        for i in 1:length(v)
            D[i, i + k] = v[i]
        end
    else
        for i in 1:length(v)
            D[i - k, i] = v[i]
        end
    end
    return D
end

end  # module ExaLinAlg
