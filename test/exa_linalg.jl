"""
    ExaLinAlg

Module providing trait-based linear algebra extensions for Array{<:ExaModels.AbstractNode}.
Extends Julia's standard Array interface without wrappers.

# Exports
- Basic operations: `zero`, `adjoint`, `transpose`, `*`, `+`, `-`
- Linear algebra: `dot`, `det`, `tr`, `norm`, `diag`, `diagm`
"""
module ExaLinAlg

using ExaModels: ExaModels
using LinearAlgebra

import Base: zero, adjoint, *, promote_rule, convert, +, -, transpose
import LinearAlgebra: dot, Adjoint, det, tr, norm, diag, diagm

export zero, adjoint, transpose, *, +, -, dot, det, tr, norm, diag, diagm, is_zero_value, zero_node

# ============================================================================
# Basic type conversions and promotions
# ============================================================================

zero(x::T) where {T <: ExaModels.AbstractNode} = 0

# Scalar operations
adjoint(x::ExaModels.AbstractNode) = x
transpose(x::ExaModels.AbstractNode) = x

convert(::Type{ExaModels.AbstractNode}, x::Number) = ExaModels.Null(x)

promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Number}) = ExaModels.AbstractNode

# ============================================================================
# Zero-value detection and optimization helpers
# ============================================================================

"""
    is_zero_value(x) -> Bool

Check if a value represents zero, handling both Number and AbstractNode types.
Returns true for: 0, 0.0, Null(0), Null(0.0), etc.
Returns false for: Null(nothing) and other values.
"""
function is_zero_value(x::Number)::Bool
    return iszero(x)
end

function is_zero_value(x::ExaModels.Null)::Bool
    # Null wraps a value in x.value
    # Null(0) or Null(0.0) are zeros
    # Null(nothing) is NOT a zero (it's used as identity in exa_linalg_w)
    val = x.value
    return val !== nothing && iszero(val)
end

function is_zero_value(x::ExaModels.AbstractNode)::Bool
    # For non-Null AbstractNode types, we cannot easily determine if they're zero
    # These are symbolic expressions, so we conservatively return false
    return false
end

"""
    zero_node() -> Null

Return a canonical zero AbstractNode (Null(0)).
"""
zero_node() = ExaModels.Null(0)

# ============================================================================
# Scalar multiplication with vectors and matrices
# ============================================================================

# Scalar × Vector
function *(a::T, v::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : a * vi for vi in v]
end

function *(a::Number, v::Vector{T}) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : a * vi for vi in v]
end

function *(a::T, v::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : a * vi for vi in v]
end

# Vector × Scalar
function *(v::Vector{T}, a::Number) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : vi * a for vi in v]
end

function *(v::Vector{T}, a::S) where {T <: Number, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : vi * a for vi in v]
end

function *(v::Vector{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), length(v))
    return [is_zero_value(vi) ? zero_node() : vi * a for vi in v]
end

# Scalar × Matrix
function *(a::T, A::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : a * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::Number, A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : a * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::T, A::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : a * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix × Scalar
function *(A::Matrix{T}, a::Number) where {T <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : A[i, j] * a for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{T}, a::S) where {T <: Number, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : A[i, j] * a for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{T}, a::S) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    is_zero_value(a) && return fill(zero_node(), size(A))
    return [is_zero_value(A[i, j]) ? zero_node() : A[i, j] * a for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Dot product (returns AbstractNode)
# ============================================================================

function dot(v::Vector{<:Number}, x::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(x) "Vectors must have the same length: got $(length(v)) and $(length(x))"
    return sum(v .* x)
end

function dot(v::Vector{T}, x::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(x) "Vectors must have the same length: got $(length(v)) and $(length(x))"
    return sum(v .* x)
end

function dot(v::Vector{T}, x::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(x) "Vectors must have the same length: got $(length(v)) and $(length(x))"
    return sum(v .* x)
end

# ============================================================================
# Matrix × Vector product
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
# Matrix × Matrix product
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
# Adjoint Vector × Matrix product
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
# Adjoint and transpose for matrices
# ============================================================================

function adjoint(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

function transpose(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return permutedims(A)
end

# ============================================================================
# Broadcasting operations
# ============================================================================

# Broadcasting automatically works for Array{<:ExaModels.AbstractNode} because:
# 1. ExaModels.AbstractNode supports element-wise operations (+, -, *, /, ^, etc.)
# 2. ExaModels.AbstractNode supports mathematical functions (cos, sin, exp, log, etc.)
# 3. Julia's broadcast system handles the rest automatically
#
# Examples that work out of the box:
# - cos.(vec_nodes)   # Apply cos to each element
# - sin.(mat_nodes)   # Apply sin to each element
# - vec_nodes .+ 1.0  # Add scalar to each element
# - vec_nodes .* v    # Element-wise multiplication
# - exp.(vec_nodes)   # Apply exp to each element
#
# No additional code needed - broadcasting is fully supported!

# ============================================================================
# Determinant (returns AbstractNode)
# ============================================================================

function det(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Determinant is only defined for square matrices, got $(n)×$(m)"

    if n == 1
        return A[1, 1]
    elseif n == 2
        return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    elseif n == 3
        # Sarrus rule for 3×3 matrices
        return (A[1, 1] * A[2, 2] * A[3, 3] +
                A[1, 2] * A[2, 3] * A[3, 1] +
                A[1, 3] * A[2, 1] * A[3, 2] -
                A[1, 3] * A[2, 2] * A[3, 1] -
                A[1, 1] * A[2, 3] * A[3, 2] -
                A[1, 2] * A[2, 1] * A[3, 3])
    else
        # Laplace expansion for n×n matrices (n ≥ 4)
        d = A[1, 1] * det(A[2:end, 2:end])  # Initialize with first term
        for j in 2:n
            minor = [A[2:end, 1:j-1] A[2:end, j+1:end]]
            cofactor = (-1)^(1 + j) * A[1, j] * det(minor)
            d = d + cofactor
        end
        return d
    end
end

# ============================================================================
# Trace (returns AbstractNode)
# ============================================================================

function tr(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Trace is only defined for square matrices, got $(n)×$(m)"
    return sum(A[i, i] for i in 1:n)
end

# ============================================================================
# Norms (return AbstractNode)
# ============================================================================

# Euclidean norm (2-norm) for vectors
function norm(v::Vector{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(vi * vi for vi in v))
end

# p-norm for vectors
function norm(v::Vector{T}, p::Real) where {T <: ExaModels.AbstractNode}
    if p == Inf
        # Infinity norm: max|vᵢ|
        # For symbolic, we can't easily compute max, so we raise an error
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

# Frobenius norm for matrices (default)
function norm(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(A[i, j] * A[i, j] for i in axes(A, 1), j in axes(A, 2)))
end

# ============================================================================
# Array addition and subtraction
# ============================================================================

# Vector + Vector
function +(v::Vector{T}, w::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(w[i]) ? v[i] :
            is_zero_value(v[i]) ? ExaModels.Null(w[i]) :
            v[i] + w[i] for i in eachindex(v)]
end

function +(v::Vector{<:Number}, w::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(v[i]) ? w[i] :
            is_zero_value(w[i]) ? ExaModels.Null(v[i]) :
            v[i] + w[i] for i in eachindex(v)]
end

function +(v::Vector{T}, w::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(w[i]) ? v[i] :
            is_zero_value(v[i]) ? w[i] :
            v[i] + w[i] for i in eachindex(v)]
end

# Vector - Vector
function -(v::Vector{T}, w::Vector{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(w[i]) ? v[i] : v[i] - w[i] for i in eachindex(v)]
end

function -(v::Vector{<:Number}, w::Vector{T}) where {T <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(w[i]) ? ExaModels.Null(v[i]) : v[i] - w[i] for i in eachindex(v)]
end

function -(v::Vector{T}, w::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [is_zero_value(w[i]) ? v[i] : v[i] - w[i] for i in eachindex(v)]
end

# Matrix + Matrix
function +(A::Matrix{T}, B::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(B[i, j]) ? A[i, j] :
            is_zero_value(A[i, j]) ? ExaModels.Null(B[i, j]) :
            A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{<:Number}, B::Matrix{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(A[i, j]) ? B[i, j] :
            is_zero_value(B[i, j]) ? ExaModels.Null(A[i, j]) :
            A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{T}, B::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(B[i, j]) ? A[i, j] :
            is_zero_value(A[i, j]) ? B[i, j] :
            A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix - Matrix
function -(A::Matrix{T}, B::Matrix{<:Number}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(B[i, j]) ? A[i, j] : A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{<:Number}, B::Matrix{T}) where {T <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(B[i, j]) ? ExaModels.Null(A[i, j]) : A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{T}, B::Matrix{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [is_zero_value(B[i, j]) ? A[i, j] : A[i, j] - B[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Diagonal operations
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
    # Create a matrix filled with Float64 zeros, then set diagonal with AbstractNodes
    D = zeros(Float64, n, n)
    # Convert to Any array to allow mixed types
    D = convert(Matrix{Any}, D)
    for i in 1:n
        D[i, i] = v[i]
    end
    return D
end

# diagm with pairs (more general form)
function diagm(kv::Pair{<:Integer, <:Vector{T}}) where {T <: ExaModels.AbstractNode}
    k, v = kv
    n = length(v) + abs(k)
    # Create a matrix filled with Float64 zeros
    D = zeros(Float64, n, n)
    # Convert to Any array to allow mixed types
    D = convert(Matrix{Any}, D)
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
