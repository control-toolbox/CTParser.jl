"""
    ExaLinAlg

Module providing trait-based linear algebra extensions for Sym = Union{AbstractNode, Real}.
Extends Julia's standard Array interface to work seamlessly with mixed symbolic/numeric arrays.

# Key Design: Bottom-Up Optimization with Sym Type
All operations work on `Sym = Union{ExaModels.AbstractNode, Real}` types.
Optimized scalar operations (opt_add, opt_sub, opt_mul) handle zero and one values,
eliminating unnecessary Null nodes from expression trees.

# Key Optimizations
- 0 * x = 0 (numeric zero, not Null node) - prevents unnecessary symbolic nodes
- 0 + x = x, 1 * x = x (identity elements)
- Uses native iszero() and isone() for detection

# Public API (Exported)
- Type: `Sym` (Union type for symbolic/numeric values)
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

# Sym represents symbolic or numeric values: either AbstractNode or Real
const Sym = Union{ExaModels.AbstractNode, Real}

export zero, adjoint, transpose, *, +, -, sum, dot, det, tr, norm, diag, diagm
export Sym


# ============================================================================
# Section 2: Optimized Scalar Operations
# ============================================================================
#
# These operations work on Sym = Union{AbstractNode, Real} types.
# Detection uses native iszero() and isone() functions.
#
# (i) Addition
#     0 + x = x
#     x + 0 = x
#
# (ii) Subtraction
#     0 - x = -x   [numeric for Real, Node1(-, x) for AbstractNode]
#     x - 0 = x
#
# (iii) Multiplication by zero (KEY OPTIMIZATION)
#     0 * x = 0   [numeric zero, not Null node]
#     x * 0 = 0   [numeric zero, not Null node]
#
# (iv) Multiplication by one
#     1 * x = x
#     x * 1 = x
# ============================================================================

"""
    opt_add(x, y)

Optimized addition.
Note: 0 + x = x + 0 = x is handled natively by ExaModels.
This function is kept for consistency and potential future updates.
Returns Sym (Union{AbstractNode, Real}).
"""
opt_add(x::T, y::S) where {T<:Sym, S<:Sym} = x + y

"""
    opt_sub(x, y)

Optimized subtraction that handles zero values.
- x - 0 = x
- 0 - y = -y (numeric for Real, symbolic Node1 for AbstractNode)
Returns Sym (Union{AbstractNode, Real}).
"""
function opt_sub(x, y)
    # x - 0 = x
    if iszero(y)
        return x
    end
    # 0 - y = -y
    if iszero(x)
        # If y is Real, return numeric negation
        # If y is AbstractNode, return symbolic unary minus
        return y isa Real ? -y : ExaModels.Node1(-, y)
    end
    # Both non-zero: perform subtraction
    return x - y
end

"""
    opt_mul(x, y)

Optimized multiplication that handles zero and one values.
- 0 * x = 0, x * 0 = 0 (THE KEY OPTIMIZATION for symbolic expressions)
- 1 * x = x, x * 1 = x
Returns Sym (Union{AbstractNode, Real}).
"""
function opt_mul(x, y)
    return x * y
end

function opt_mul(x::Real, y::ExaModels.AbstractNode)
    if iszero(x)
        return 0
    else
        return x * y
    end
end

function opt_mul(x::ExaModels.AbstractNode, y::Real)
    if iszero(y)
        return 0
    else
        return x * y
    end
end

"""
    opt_sum(iter)

Optimized sum that skips zero values entirely.
Returns numeric 0 if all elements are zero.
Returns Sym (Union{AbstractNode, Real}).
"""
function opt_sum(iter)
    result = nothing  # Sentinel for "no non-zero terms yet"
    for x in iter
        iszero(x) && continue  # Skip zeros entirely
        if result === nothing
            # First non-zero term
            result = x
        else
            # Add to accumulator
            result = opt_add(result, x)
        end
    end
    return result === nothing ? 0 : result  # Return numeric 0
end

# ============================================================================
# Section 3: sum (wrapper around opt_sum)
# ============================================================================

"""
    sum(arr::AbstractArray{<:Sym})

Optimized sum for arrays of Sym that skips zeros and uses opt_add.
"""
sum(arr::AbstractArray{<:Sym}) = opt_sum(arr)

# ============================================================================
# Section 4: Basic Type Conversions and Promotions
# ============================================================================

zero(x::T) where {T <: ExaModels.AbstractNode} = 0

# Scalar operations
adjoint(x::ExaModels.AbstractNode) = x
transpose(x::ExaModels.AbstractNode) = x

# Note: convert() and promote_rule() removed - not needed with Sym approach
# Julia naturally infers Union types when mixing AbstractNode and Real

# ============================================================================
# Section 4: Dot Product (uses opt_mul, opt_sum)
# ============================================================================

function dot(v::Vector{<:Real}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

function dot(v::Vector{<:Sym}, w::Vector{<:Real})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

function dot(v::Vector{<:Sym}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return opt_sum(opt_mul(v[i], w[i]) for i in eachindex(v))
end

# ============================================================================
# Section 5: Scalar × Vector/Matrix Multiplication (uses opt_mul)
# ============================================================================

# Scalar × Vector
function *(a::T, v::Vector{<:Real}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, vi) for vi in v]
end

function *(a::Number, v::Vector{<:Sym})
    return [opt_mul(a, vi) for vi in v]
end

function *(a::T, v::Vector{<:Sym}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, vi) for vi in v]
end

# Vector × Scalar
function *(v::Vector{<:Sym}, a::Number)
    return [opt_mul(vi, a) for vi in v]
end

function *(v::Vector{<:Real}, a::ExaModels.AbstractNode)
    return [opt_mul(vi, a) for vi in v]
end

function *(v::Vector{<:Sym}, a::ExaModels.AbstractNode)
    return [opt_mul(vi, a) for vi in v]
end

# Scalar × Matrix
function *(a::T, A::Matrix{<:Real}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::Number, A::Matrix{<:Sym})
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function *(a::T, A::Matrix{<:Sym}) where {T <: ExaModels.AbstractNode}
    return [opt_mul(a, A[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix × Scalar
function *(A::Matrix{<:Sym}, a::Number)
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{<:Real}, a::ExaModels.AbstractNode)
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

function *(A::Matrix{<:Sym}, a::ExaModels.AbstractNode)
    return [opt_mul(A[i, j], a) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 6: Matrix × Vector Product (uses dot)
# ============================================================================

function *(A::Matrix{<:Real}, x::Vector{<:Sym})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{<:Sym}, x::Vector{<:Real})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{<:Sym}, x::Vector{<:Sym})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $n columns but vector has $(length(x)) elements"
    return [dot(A[i, :], x) for i in 1:m]
end

# ============================================================================
# Section 7: Matrix × Matrix Product (uses dot)
# ============================================================================

function *(A::Matrix{<:Real}, B::Matrix{<:Sym})
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Matrix{<:Sym}, B::Matrix{<:Real})
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

function *(A::Matrix{<:Sym}, B::Matrix{<:Sym})
    m, n = size(A)
    p, q = size(B)
    @assert n == p "Dimension mismatch: A has $n columns but B has $p rows"
    return [dot(A[i, :], B[:, j]) for i in 1:m, j in 1:q]
end

# ============================================================================
# Section 8: Adjoint Vector × Matrix Product
# ============================================================================

function *(p::Adjoint{T, Vector{T}}, A::Matrix{<:Real}) where {T <: Sym}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, Vector{T}}, A::Matrix{<:Sym}) where {T <: Real}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

function *(p::Adjoint{T, Vector{T}}, A::Matrix{<:Sym}) where {T <: Sym}
    m, n = size(A)
    @assert m == length(p) "Dimension mismatch: vector has $(length(p)) elements but matrix has $m rows"
    return [p * A[:, j] for j in 1:n]'
end

# ============================================================================
# Section 9: Adjoint and Transpose for Matrices
# ============================================================================

function adjoint(A::Matrix{<:Sym})
    return permutedims(A)
end

function transpose(A::Matrix{<:Sym})
    return permutedims(A)
end

# ============================================================================
# Section 10: Vector/Matrix Addition (uses opt_add)
# ============================================================================

# Vector + Vector
function +(v::Vector{<:Sym}, w::Vector{<:Real})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

function +(v::Vector{<:Real}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

function +(v::Vector{<:Sym}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_add(v[i], w[i]) for i in eachindex(v)]
end

# Matrix + Matrix
function +(A::Matrix{<:Sym}, B::Matrix{<:Real})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{<:Real}, B::Matrix{<:Sym})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function +(A::Matrix{<:Sym}, B::Matrix{<:Sym})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_add(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 11: Vector/Matrix Subtraction (uses opt_sub)
# ============================================================================

# Vector - Vector
function -(v::Vector{<:Sym}, w::Vector{<:Real})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

function -(v::Vector{<:Real}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

function -(v::Vector{<:Sym}, w::Vector{<:Sym})
    @assert length(v) == length(w) "Vectors must have the same length: got $(length(v)) and $(length(w))"
    return [opt_sub(v[i], w[i]) for i in eachindex(v)]
end

# Matrix - Matrix
function -(A::Matrix{<:Sym}, B::Matrix{<:Real})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{<:Real}, B::Matrix{<:Sym})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

function -(A::Matrix{<:Sym}, B::Matrix{<:Sym})
    @assert size(A) == size(B) "Matrices must have the same size: got $(size(A)) and $(size(B))"
    return [opt_sub(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 12: Determinant (uses opt_mul, opt_add, opt_sub)
# ============================================================================

function det(A::Matrix{<:Sym})
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

function tr(A::Matrix{<:Sym})
    n, m = size(A)
    @assert n == m "Trace is only defined for square matrices, got $(n)×$(m)"
    return opt_sum(A[i, i] for i in 1:n)
end

# ============================================================================
# Section 14: Norms (uses opt_sum, opt_mul)
# ============================================================================

# Euclidean norm (2-norm) for vectors
function norm(v::Vector{<:Sym})
    return sqrt(opt_sum(opt_mul(vi, vi) for vi in v))
end

# p-norm for vectors
function norm(v::Vector{<:Sym}, p::Real)
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
function norm(A::Matrix{<:Sym})
    return sqrt(opt_sum(opt_mul(A[i, j], A[i, j]) for i in axes(A, 1), j in axes(A, 2)))
end

# ============================================================================
# Section 15: Diagonal Operations
# ============================================================================

# Extract diagonal from matrix
function diag(A::Matrix{<:Sym})
    n, m = size(A)
    k = min(n, m)
    return [A[i, i] for i in 1:k]
end

# Create diagonal matrix from vector
function diagm(v::Vector{<:Sym})
    n = length(v)
    # Create a matrix with Sym element type to allow mixed AbstractNode and Real
    D = Matrix{Sym}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = (i == j) ? v[i] : 0  # Use numeric 0, not zero_node()
    end
    return D
end

# diagm with pairs (more general form)
function diagm(kv::Pair{<:Integer, <:Vector{<:Sym}})
    k, v = kv
    n = length(v) + abs(k)
    # Create a matrix with Sym element type to allow mixed AbstractNode and Real
    D = Matrix{Sym}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = 0  # Use numeric 0, not zero_node()
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
