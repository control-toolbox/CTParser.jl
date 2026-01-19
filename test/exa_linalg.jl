"""
    ExaLinAlg

Module providing trait-based linear algebra extensions for Array{<:ExaModels.AbstractNode}.
Extends Julia's standard Array interface without wrappers.

# Key Design: Bottom-Up Optimization
All operations are built on optimized scalar operations (opt_add, opt_sub, opt_mul)
that properly handle Null nodes, avoiding unnecessary expression tree nodes.

# Public API (Exported)
- Canonical nodes: `zero_node`, `one_node`
- Basic operations: `zero`, `adjoint`, `transpose`, `*`, `+`, `-`, `sum`
- Linear algebra: `dot`, `det`, `tr`, `norm`, `diag`, `diagm`

# Internal Functions (Not Exported)
- Optimized primitives: `opt_add`, `opt_sub`, `opt_mul`, `opt_sum`
  These are implementation details used internally by public operations.
"""
module ExaLinAlg

using ExaModels: ExaModels
using LinearAlgebra

import Base: zero, adjoint, *, promote_rule, convert, +, -, transpose, sum
import LinearAlgebra: dot, Adjoint, det, tr, norm, diag, diagm

export zero, adjoint, transpose, *, +, -, sum, dot, det, tr, norm, diag, diagm
export zero_node, one_node

# ============================================================================
# Section 1: Canonical Nodes
# ============================================================================
#
# Canonical encodings:
# - Zero: Null(0) represents zero
# - One:  Null(1) represents one
# ============================================================================

"""
    zero_node() -> Null{Int}

Return the canonical zero AbstractNode: Null(0).
"""
zero_node() = ExaModels.Null(0)

"""
    one_node() -> Null{Int}

Return the canonical one AbstractNode: Null(1).
"""
one_node() = ExaModels.Null(1)

# ============================================================================
# Section 2: Optimized Scalar Operations
# ============================================================================
#
# Rules for opt_add:
#   Null(x) + Null(y) = Null(x + y)
#   Null(x) + e = x + e
#   e + Null(x) = e + x
#   e + f = e + f (native from ExaModels)
#
# Rules for opt_sub:
#   Null(x) - Null(y) = Null(x - y)
#   Null(0) - e = -e
#   Null(x) - e = x - e when !iszero(x)
#   e - Null(x) = e - x
#   e - f = e - f (native from ExaModels)
#
# Rules for opt_mul:
#   Null(x) * Null(y) = Null(x * y)
#   Null(0) * e = Null(0)
#   e * Null(0) = Null(0)
#   Null(x) * e = x * e when !iszero(x)
#   e * Null(x) = e * x when !iszero(x)
#   e * f = e * f (native from ExaModels)
# ============================================================================

# opt_add: Null{T} cases
opt_add(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value + y.value)
opt_add(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = x.value + y
opt_add(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x + y.value
# opt_add: AbstractNode cases (non-Null)
opt_add(x::ExaModels.AbstractNode, y::ExaModels.AbstractNode) = x + y
# opt_add: Number cases (convert to Null)
opt_add(x::Number, y::ExaModels.AbstractNode) = opt_add(ExaModels.Null(x), y)
opt_add(x::ExaModels.AbstractNode, y::Number) = opt_add(x, ExaModels.Null(y))
opt_add(x::Number, y::Number) = opt_add(ExaModels.Null(x), ExaModels.Null(y))

# opt_sub: Null{T} cases
opt_sub(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value - y.value)
opt_sub(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? (-y) : (x.value - y)
opt_sub(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x - y.value
# opt_sub: AbstractNode cases (non-Null)
opt_sub(x::ExaModels.AbstractNode, y::ExaModels.AbstractNode) = x - y
# opt_sub: Number cases (convert to Null)
opt_sub(x::Number, y::ExaModels.AbstractNode) = opt_sub(ExaModels.Null(x), y)
opt_sub(x::ExaModels.AbstractNode, y::Number) = opt_sub(x, ExaModels.Null(y))
opt_sub(x::Number, y::Number) = opt_sub(ExaModels.Null(x), ExaModels.Null(y))

# opt_mul: Null{T} cases
opt_mul(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value * y.value)
opt_mul(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? ExaModels.Null(0) : (x.value * y)
opt_mul(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = iszero(y.value) ? ExaModels.Null(0) : (x * y.value)
# opt_mul: AbstractNode cases (non-Null)
opt_mul(x::ExaModels.AbstractNode, y::ExaModels.AbstractNode) = x * y
# opt_mul: Number cases (convert to Null)
opt_mul(x::Number, y::ExaModels.AbstractNode) = opt_mul(ExaModels.Null(x), y)
opt_mul(x::ExaModels.AbstractNode, y::Number) = opt_mul(x, ExaModels.Null(y))
opt_mul(x::Number, y::Number) = opt_mul(ExaModels.Null(x), ExaModels.Null(y))

"""
    opt_sum(iter)

Optimized sum using opt_add.
Returns zero_node() if the iterator is empty.
"""
function opt_sum(iter)
    result = zero_node()
    for x in iter
        result = opt_add(result, x)
    end
    return result
end

# ============================================================================
# Section 3: sum (wrapper around opt_sum)
# ============================================================================

"""
    sum(arr::AbstractArray{<:ExaModels.AbstractNode})

Optimized sum for arrays of AbstractNode that skips zeros and uses opt_add.
"""
sum(arr::AbstractArray{<:ExaModels.AbstractNode}) = opt_sum(arr)

# ============================================================================
# Section 4: Basic Type Conversions and Promotions
# ============================================================================

zero(x::T) where {T <: ExaModels.AbstractNode} = 0

# Scalar operations
adjoint(x::ExaModels.AbstractNode) = x
transpose(x::ExaModels.AbstractNode) = x

convert(::Type{ExaModels.AbstractNode}, x::Number) = iszero(x) ? zero_node() : ExaModels.Null(x)

promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Number}) = ExaModels.AbstractNode

# ============================================================================
# Section 4: Dot Product (uses opt_mul, opt_sum)
# ============================================================================

function dot(v::Vector{<:Number}, w::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

function dot(v::Vector{T}, w::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

function dot(v::Vector{T}, w::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

# ============================================================================
# Section 5: Scalar × Vector/Matrix Multiplication (uses opt_mul)
# ============================================================================

# Scalar × Vector
function *(a::T, v::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, vi) for vi in v]
end

function *(a::Number, v::Vector{T}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, vi) for vi in v]
end

function *(a::T, v::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [opt_mul(a, vi) for vi in v]
end

# Vector × Scalar
function *(v::Vector{T}, a::Number) where {T <: ExaModels.AbstractNode}
    return [opt_mul(vi, a) for vi in v]
end

function *(v::Vector{T}, a::S) where {T <: Number, S <: ExaModels.AbstractNode}
    return [opt_mul(vi, a) for vi in v]
end

function *(v::Vector{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [opt_mul(vi, a) for vi in v]
end

# Scalar × Matrix
function *(a::T, A::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::Number, A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::T, A::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix × Scalar
function *(A::Matrix{T}, a::Number) where {T <: ExaModels.AbstractNode}
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{T}, a::S) where {T <: Number, S <: ExaModels.AbstractNode}
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 6: Matrix × Vector Product (uses dot)
# ============================================================================

function *(A::Matrix{<:Number}, x::Vector{T}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{T}, x::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{T}, x::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

# ============================================================================
# Section 7: Matrix × Matrix Product (uses dot)
# ============================================================================

function *(A::Matrix{<:Number}, B::Matrix{T}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Matrix{T}, B::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Matrix{T}, B::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

# ============================================================================
# Section 8: Adjoint Vector × Matrix Product
# ============================================================================

function *(p::Adjoint{T, Vector{T}}, A::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, Vector{T}}, A::Matrix{S}) where {T <: Number, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, Vector{T}}, A::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

# ============================================================================
# Section 9: Adjoint and Transpose for Matrices
# ============================================================================

function adjoint(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

function transpose(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

# ============================================================================
# Section 10: Vector/Matrix Addition (uses opt_add)
# ============================================================================

# Vector + Vector
function +(v::Vector{T}, w::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

function +(v::Vector{<:Number}, w::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

function +(v::Vector{T}, w::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

# Matrix + Matrix
function +(A::Matrix{T}, B::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{<:Number}, B::Matrix{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{T}, B::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 11: Vector/Matrix Subtraction (uses opt_sub)
# ============================================================================

# Vector - Vector
function -(v::Vector{T}, w::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

function -(v::Vector{<:Number}, w::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

function -(v::Vector{T}, w::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

# Matrix - Matrix
function -(A::Matrix{T}, B::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{<:Number}, B::Matrix{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{T}, B::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 12: Determinant (uses opt_mul, opt_add, opt_sub)
# ============================================================================

function det(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Determinant is only defined for square matrices, got $(n)×$(m)"

    if n == 1
        return A[1, 1]
    elseif n == 2
        return opt_sub(opt_mul(A[1, 1], A[2, 2]), opt_mul(A[1, 2], A[2, 1]))
    elseif n == 3
        # Sarrus rule for 3×3 matrices
        pos1 = opt_mul(opt_mul(A[1, 1], A[2, 2]), A[3, 3])
        pos2 = opt_mul(opt_mul(A[1, 2], A[2, 3]), A[3, 1])
        pos3 = opt_mul(opt_mul(A[1, 3], A[2, 1]), A[3, 2])
        neg1 = opt_mul(opt_mul(A[1, 3], A[2, 2]), A[3, 1])
        neg2 = opt_mul(opt_mul(A[1, 1], A[2, 3]), A[3, 2])
        neg3 = opt_mul(opt_mul(A[1, 2], A[2, 1]), A[3, 3])
        pos_sum = opt_add(opt_add(pos1, pos2), pos3)
        neg_sum = opt_add(opt_add(neg1, neg2), neg3)
        return opt_sub(pos_sum, neg_sum)
    else
        # Laplace expansion for n×n matrices (n ≥ 4)
        d = opt_mul(A[1, 1], det(A[2:end, 2:end]))  # Initialize with first term
        for j in 2:n
            minor = [A[2:end, 1:j-1] A[2:end, j+1:end]]
            sign_coeff = iseven(j) ? -1 : 1
            cofactor = opt_mul(sign_coeff, opt_mul(A[1, j], det(minor)))
            d = opt_add(d, cofactor)
        end
        return d
    end
end

# ============================================================================
# Section 13: Trace (uses opt_sum)
# ============================================================================

function tr(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Trace is only defined for square matrices, got $(n)×$(m)"
    return opt_sum(A[i, i] for i in 1:n)
end

# ============================================================================
# Section 14: Norms (uses opt_sum, opt_mul)
# ============================================================================

# Euclidean norm (2-norm) for vectors
function norm(v::Vector{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(opt_sum(opt_mul(vi, vi) for vi in v))
end

# p-norm for vectors
function norm(v::Vector{T}, p::Real) where {T <: ExaModels.AbstractNode}
    if p == Inf
        # Infinity norm: max|vᵢ|
        error("Infinity norm not supported for symbolic AbstractNode vectors")
    elseif p == 1
        # 1-norm: sum of absolute values
        return opt_sum(abs(vi) for vi in v)
    elseif p == 2
        # 2-norm: Euclidean norm
        return sqrt(opt_sum(opt_mul(vi, vi) for vi in v))
    else
        # General p-norm
        return opt_sum(abs(vi)^p for vi in v)^(1/p)
    end
end

# Frobenius norm for matrices
function norm(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(opt_sum(opt_mul(A[i, j], A[i, j]) for i in axes(A, 1), j in axes(A, 2)))
end

# ============================================================================
# Section 15: Diagonal Operations
# ============================================================================

# Extract diagonal from matrix
function diag(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    k = min(n, m)
    return [A[i, i] for i in 1:k]
end

# Create diagonal matrix from vector
function diagm(v::Vector{T}) where {T <: ExaModels.AbstractNode}
    n = length(v)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = (i == j) ? v[i] : zero_node()
    end
    return D
end

# diagm with pairs (more general form)
function diagm(kv::Pair{<:Integer, <:Vector{T}}) where {T <: ExaModels.AbstractNode}
    k, v = kv
    n = length(v) + abs(k)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = zero_node()
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
