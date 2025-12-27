"""
    SymbolicOps

Trait-based implementations for symbolic linear algebra operations.

This module provides the internal implementations that dispatch on trait types,
keeping the architecture clean and avoiding direct type piracy in the core logic.
"""

using ExaModels: AbstractNode, Null
using LinearAlgebra
include("symbolic_traits.jl")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    sym_add(a, b)

Symbolic addition that absorbs `Null(nothing)` as additive identity.
"""
function sym_add(a, b)
    a isa Null{Nothing} && return b
    b isa Null{Nothing} && return a
    return a + b
end

"""
    sym_mul(a, b)

Symbolic multiplication.
"""
sym_mul(a, b) = a * b

# =============================================================================
# MATRIX × VECTOR MULTIPLICATION (trait-based)
# =============================================================================

"""
    _matmul_vec(trait_A, trait_x, A, x)

Internal matrix-vector multiplication dispatched on symbolic traits.
"""
function _matmul_vec(
    ::IsSymbolic,
    ::IsNotSymbolic,
    A::AbstractMatrix{<:AbstractNode},
    x::AbstractVector{<:Number}
)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: A is $(m)×$(n), x has length $(length(x))"
    result = Vector{AbstractNode}(undef, m)
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

function _matmul_vec(
    ::IsNotSymbolic,
    ::IsSymbolic,
    A::AbstractMatrix{<:Number},
    x::AbstractVector{<:AbstractNode}
)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: A is $(m)×$(n), x has length $(length(x))"
    result = Vector{AbstractNode}(undef, m)
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

function _matmul_vec(
    ::IsSymbolic,
    ::IsSymbolic,
    A::AbstractMatrix{<:AbstractNode},
    x::AbstractVector{<:AbstractNode}
)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: A is $(m)×$(n), x has length $(length(x))"
    result = Vector{AbstractNode}(undef, m)
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# Fallback to Base for numeric × numeric
function _matmul_vec(
    ::IsNotSymbolic,
    ::IsNotSymbolic,
    A::AbstractMatrix{<:Number},
    x::AbstractVector{<:Number}
)
    return A * x
end

# =============================================================================
# ROW VECTOR × MATRIX (via Adjoint, trait-based)
# =============================================================================

"""
    _rowvec_matmul(trait_x, trait_A, x_adj, A)

Internal row-vector-matrix multiplication dispatched on symbolic traits.
"""
function _rowvec_matmul(
    ::IsSymbolic,
    ::IsNotSymbolic,
    x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:AbstractNode}},
    A::AbstractMatrix{<:Number}
)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch"
    result = Vector{AbstractNode}(undef, p)
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

function _rowvec_matmul(
    ::IsNotSymbolic,
    ::IsSymbolic,
    x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}},
    A::AbstractMatrix{<:AbstractNode}
)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch"
    result = Vector{AbstractNode}(undef, p)
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

function _rowvec_matmul(
    ::IsSymbolic,
    ::IsSymbolic,
    x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:AbstractNode}},
    A::AbstractMatrix{<:AbstractNode}
)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch"
    result = Vector{AbstractNode}(undef, p)
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# Fallback to Base
function _rowvec_matmul(
    ::IsNotSymbolic,
    ::IsNotSymbolic,
    x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}},
    A::AbstractMatrix{<:Number}
)
    return x * A
end

# =============================================================================
# MATRIX × MATRIX MULTIPLICATION (trait-based)
# =============================================================================

"""
    _matmul_mat(trait_A, trait_B, A, B)

Internal matrix-matrix multiplication dispatched on symbolic traits.
"""
function _matmul_mat(
    ::IsSymbolic,
    ::IsNotSymbolic,
    A::AbstractMatrix{<:AbstractNode},
    B::AbstractMatrix{<:Number}
)
    m, k = size(A)
    k2, n = size(B)
    @assert k == k2 "Dimension mismatch"
    result = Matrix{AbstractNode}(undef, m, n)
    for i in 1:m, j in 1:n
        acc = Null(nothing)
        for l in 1:k
            acc = sym_add(acc, sym_mul(A[i, l], B[l, j]))
        end
        result[i, j] = acc
    end
    return result
end

function _matmul_mat(
    ::IsNotSymbolic,
    ::IsSymbolic,
    A::AbstractMatrix{<:Number},
    B::AbstractMatrix{<:AbstractNode}
)
    m, k = size(A)
    k2, n = size(B)
    @assert k == k2 "Dimension mismatch"
    result = Matrix{AbstractNode}(undef, m, n)
    for i in 1:m, j in 1:n
        acc = Null(nothing)
        for l in 1:k
            acc = sym_add(acc, sym_mul(A[i, l], B[l, j]))
        end
        result[i, j] = acc
    end
    return result
end

function _matmul_mat(
    ::IsSymbolic,
    ::IsSymbolic,
    A::AbstractMatrix{<:AbstractNode},
    B::AbstractMatrix{<:AbstractNode}
)
    m, k = size(A)
    k2, n = size(B)
    @assert k == k2 "Dimension mismatch"
    result = Matrix{AbstractNode}(undef, m, n)
    for i in 1:m, j in 1:n
        acc = Null(nothing)
        for l in 1:k
            acc = sym_add(acc, sym_mul(A[i, l], B[l, j]))
        end
        result[i, j] = acc
    end
    return result
end

# Fallback to Base
function _matmul_mat(
    ::IsNotSymbolic,
    ::IsNotSymbolic,
    A::AbstractMatrix{<:Number},
    B::AbstractMatrix{<:Number}
)
    return A * B
end

# =============================================================================
# DOT PRODUCT (trait-based)
# =============================================================================

"""
    _dot_product(trait_x, trait_y, x, y)

Internal dot product dispatched on symbolic traits.
"""
function _dot_product(
    ::IsSymbolic,
    ::IsSymbolic,
    x::AbstractVector{<:AbstractNode},
    y::AbstractVector{<:AbstractNode}
)
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

function _dot_product(
    ::IsSymbolic,
    ::IsNotSymbolic,
    x::AbstractVector{<:AbstractNode},
    y::AbstractVector{<:Number}
)
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

function _dot_product(
    ::IsNotSymbolic,
    ::IsSymbolic,
    x::AbstractVector{<:Number},
    y::AbstractVector{<:AbstractNode}
)
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# Fallback to LinearAlgebra
function _dot_product(
    ::IsNotSymbolic,
    ::IsNotSymbolic,
    x::AbstractVector{<:Number},
    y::AbstractVector{<:Number}
)
    return dot(x, y)
end

# =============================================================================
# NORM (trait-based)
# =============================================================================

"""
    _norm_p(trait_x, x, p)

Internal p-norm dispatched on symbolic trait.
"""
function _norm_p(::IsSymbolic, x::AbstractVector{<:AbstractNode}, p::Real)
    if p == 2
        # 2-norm: sqrt(sum(x[i]^2))
        acc = Null(nothing)
        for xi in x
            acc = sym_add(acc, sym_mul(xi, xi))
        end
        return sqrt(acc)
    elseif p == 1
        acc = Null(nothing)
        for xi in x
            acc = sym_add(acc, abs(xi))
        end
        return acc
    elseif isinf(p)
        error("Infinity norm not supported for symbolic vectors")
    else
        acc = Null(nothing)
        for xi in x
            acc = sym_add(acc, abs(xi)^p)
        end
        return acc^(1/p)
    end
end

function _norm_p(::IsNotSymbolic, x::AbstractVector{<:Number}, p::Real)
    return norm(x, p)
end

# Matrix Frobenius norm
function _norm_frob(::IsSymbolic, A::AbstractMatrix{<:AbstractNode})
    acc = Null(nothing)
    for i in eachindex(A)
        acc = sym_add(acc, A[i] * A[i])
    end
    return sqrt(acc)
end

function _norm_frob(::IsNotSymbolic, A::AbstractMatrix{<:Number})
    return norm(A)
end

# =============================================================================
# ADJOINT/TRANSPOSE (trait-based)
# =============================================================================

"""
    _adjoint_mat(trait_A, A)

Internal matrix adjoint dispatched on symbolic trait.
"""
function _adjoint_mat(::IsSymbolic, A::AbstractMatrix{<:AbstractNode})
    # For symbolic expressions, adjoint is just transpose (assuming real)
    return permutedims(A)
end

function _adjoint_mat(::IsNotSymbolic, A::AbstractMatrix{<:Number})
    return adjoint(A)
end

"""
    _transpose_mat(trait_A, A)

Internal matrix transpose dispatched on symbolic trait.
"""
function _transpose_mat(::IsSymbolic, A::AbstractMatrix{<:AbstractNode})
    return permutedims(A)
end

function _transpose_mat(::IsNotSymbolic, A::AbstractMatrix{<:Number})
    return transpose(A)
end
