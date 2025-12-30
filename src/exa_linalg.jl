# =============================================================================
# exa_linalg.jl - Symbolic wrappers for ExaModels.AbstractNode
#
# This file defines wrapper types that completely eliminate type piracy:
# - SymNumber: wraps scalar AbstractNode values
# - SymVector, SymMatrix: wrap arrays of AbstractNode
#
# Key design principle: NO type piracy on AbstractNode
# All operations dispatch on our wrapper types, never on AbstractNode directly
# =============================================================================

using ExaModels: AbstractNode, Null
using LinearAlgebra
using LinearAlgebra: norm_sqr
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, AbstractCompressedVector

# =============================================================================
# EXPORTS
# =============================================================================
export sym_add, sym_mul, norm_sqr
export SymVector, SymMatrix, SymArray, AbstractSymArray
export SymNumber, unwrap_scalar

# =============================================================================
# 1. TYPE DEFINITIONS
# =============================================================================

# Scalar wrapper - wraps a single AbstractNode value
struct SymNumber{T<:AbstractNode}
    value::T
end

# Abstract supertype for all symbolic array wrappers
abstract type AbstractSymArray{N} <: AbstractArray{SymNumber, N} end

# Symbolic vector wrapper (now stores SymNumber elements)
struct SymVector <: AbstractSymArray{1}
    data::Vector{SymNumber}
end

# Symbolic matrix wrapper (now stores SymNumber elements)
struct SymMatrix <: AbstractSymArray{2}
    data::Matrix{SymNumber}
end

# Generic N-dimensional wrapper (for future extensibility)
struct SymArray{N} <: AbstractSymArray{N}
    data::Array{SymNumber, N}
end

# =============================================================================
# 2. CONSTRUCTORS
# =============================================================================

# Convenience constructors
# Note: The struct definition automatically provides:
#   SymNumber{T}(value::T) where T<:AbstractNode (inner constructor)
#   SymNumber(x::AbstractNode) (outer constructor)
# We only need to add constructors for non-AbstractNode types:
SymNumber(x::Number) = SymNumber{Null{typeof(x)}}(Null(x))

# Construct vectors from AbstractNode arrays
function SymVector(v::AbstractVector{<:AbstractNode})
    return SymVector([SymNumber(x) for x in v])
end

# Construct matrices from AbstractNode arrays
function SymMatrix(m::AbstractMatrix{<:AbstractNode})
    return SymMatrix([SymNumber(x) for x in m])
end

# Construct from SymNumber arrays (already wrapped)
SymVector(v::AbstractVector{SymNumber}) = SymVector(collect(v))
SymMatrix(m::AbstractMatrix{SymNumber}) = SymMatrix(collect(m))

# Convenience constructors with UndefInitializer
SymVector(::UndefInitializer, n::Int) = SymVector(Vector{SymNumber}(undef, n))
SymMatrix(::UndefInitializer, m::Int, n::Int) = SymMatrix(Matrix{SymNumber}(undef, m, n))
SymArray{N}(::UndefInitializer, dims::Vararg{Int,N}) where {N} = SymArray{N}(Array{SymNumber,N}(undef, dims...))

# =============================================================================
# 3. UNWRAP HELPERS
# =============================================================================

# Extract underlying AbstractNode from SymNumber
unwrap_scalar(x::SymNumber) = x.value
unwrap_scalar(x::AbstractNode) = x  # Pass-through if already unwrapped
unwrap_scalar(x::Number) = Null(x)  # Convert numbers to Null

# Extract underlying data from array wrappers (returns array of AbstractNode)
unwrap(v::SymVector) = [unwrap_scalar(x) for x in v.data]
unwrap(m::SymMatrix) = [unwrap_scalar(x) for x in m.data]
unwrap(a::SymArray) = [unwrap_scalar(x) for x in a.data]
unwrap(x::SymNumber) = unwrap_scalar(x)
unwrap(x::AbstractNode) = x  # Pass-through
unwrap(x::Number) = x  # Pass-through
unwrap(x::AbstractArray) = x  # Pass-through

# =============================================================================
# 4. SYMNUMBER OPERATIONS (NO TYPE PIRACY - dispatching on OUR type)
# =============================================================================

# Arithmetic operations
Base.:+(x::SymNumber, y::SymNumber) = SymNumber(unwrap_scalar(x) + unwrap_scalar(y))
Base.:-(x::SymNumber, y::SymNumber) = SymNumber(unwrap_scalar(x) - unwrap_scalar(y))
Base.:*(x::SymNumber, y::SymNumber) = SymNumber(unwrap_scalar(x) * unwrap_scalar(y))
Base.:/(x::SymNumber, y::SymNumber) = SymNumber(unwrap_scalar(x) / unwrap_scalar(y))
Base.:^(x::SymNumber, p::Real) = SymNumber(unwrap_scalar(x)^p)
Base.:^(x::SymNumber, y::SymNumber) = SymNumber(unwrap_scalar(x)^unwrap_scalar(y))

Base.:-(x::SymNumber) = SymNumber(-unwrap_scalar(x))
Base.:+(x::SymNumber) = x

# Mixed operations with numbers
Base.:+(x::SymNumber, y::Number) = SymNumber(unwrap_scalar(x) + y)
Base.:+(x::Number, y::SymNumber) = SymNumber(x + unwrap_scalar(y))
Base.:-(x::SymNumber, y::Number) = SymNumber(unwrap_scalar(x) - y)
Base.:-(x::Number, y::SymNumber) = SymNumber(x - unwrap_scalar(y))
Base.:*(x::SymNumber, y::Number) = SymNumber(unwrap_scalar(x) * y)
Base.:*(x::Number, y::SymNumber) = SymNumber(x * unwrap_scalar(y))
Base.:/(x::SymNumber, y::Number) = SymNumber(unwrap_scalar(x) / y)
Base.:/(x::Number, y::SymNumber) = SymNumber(x / unwrap_scalar(y))

# Math functions
Base.abs(x::SymNumber) = SymNumber(abs(unwrap_scalar(x)))
Base.abs2(x::SymNumber) = SymNumber(abs2(unwrap_scalar(x)))
Base.sqrt(x::SymNumber) = SymNumber(sqrt(unwrap_scalar(x)))
Base.exp(x::SymNumber) = SymNumber(exp(unwrap_scalar(x)))
Base.log(x::SymNumber) = SymNumber(log(unwrap_scalar(x)))
Base.sin(x::SymNumber) = SymNumber(sin(unwrap_scalar(x)))
Base.cos(x::SymNumber) = SymNumber(cos(unwrap_scalar(x)))
Base.tan(x::SymNumber) = SymNumber(tan(unwrap_scalar(x)))

# Identity elements
Base.zero(::Type{<:SymNumber}) = SymNumber(Null(nothing))
Base.zero(::SymNumber) = SymNumber(Null(nothing))
Base.one(::Type{<:SymNumber}) = SymNumber(Null(1))
Base.one(::SymNumber) = SymNumber(Null(1))

# Adjoint/transpose
Base.adjoint(x::SymNumber) = x
Base.transpose(x::SymNumber) = x
Base.conj(x::SymNumber) = x

# Scalar properties
Base.broadcastable(x::SymNumber) = Ref(x)
Base.iterate(::SymNumber) = nothing
Base.length(::SymNumber) = 1
Base.size(::SymNumber) = ()
Base.ndims(::SymNumber) = 0
Base.ndims(::Type{<:SymNumber}) = 0
Base.IteratorSize(::Type{<:SymNumber}) = Base.HasShape{0}()

# Promotion and conversion
Base.promote_rule(::Type{<:SymNumber}, ::Type{<:Number}) = SymNumber
Base.promote_rule(::Type{SymNumber{S}}, ::Type{SymNumber{T}}) where {S,T} = SymNumber
Base.convert(::Type{SymNumber}, x::Number) = SymNumber(Null(x))
Base.convert(::Type{SymNumber}, x::SymNumber) = x
Base.convert(::Type{SymNumber}, x::AbstractNode) = SymNumber(x)
Base.convert(::Type{SymNumber{T}}, x::SymNumber) where {T} = x  # Allow conversion between SymNumber types
Base.convert(::Type{SymNumber{T}}, x::AbstractNode) where {T<:AbstractNode} = SymNumber(x)

# =============================================================================
# 5. ARRAY INTERFACE
# =============================================================================

# Size and shape
Base.size(v::SymVector) = size(v.data)
Base.size(m::SymMatrix) = size(m.data)
Base.size(a::SymArray) = size(a.data)

Base.length(v::SymVector) = length(v.data)
Base.length(m::SymMatrix) = length(m.data)
Base.length(a::SymArray) = length(a.data)

Base.axes(v::SymVector) = axes(v.data)
Base.axes(m::SymMatrix) = axes(m.data)
Base.axes(a::SymArray) = axes(a.data)

# Index style
Base.IndexStyle(::Type{<:AbstractSymArray}) = IndexLinear()

# Indexing - Returns SymNumber (wrapped scalar)
Base.getindex(v::SymVector, i::Int) = v.data[i]
Base.getindex(m::SymMatrix, i::Int, j::Int) = m.data[i, j]
Base.getindex(m::SymMatrix, i::Int) = m.data[i]  # Linear indexing
Base.getindex(a::SymArray, i::Int) = a.data[i]
Base.getindex(a::SymArray{N}, I::Vararg{Int,N}) where {N} = a.data[I...]

# Setindex - accepts both SymNumber and AbstractNode
Base.setindex!(v::SymVector, val::SymNumber, i::Int) = (v.data[i] = val)
Base.setindex!(v::SymVector, val::AbstractNode, i::Int) = (v.data[i] = SymNumber(val))
Base.setindex!(m::SymMatrix, val::SymNumber, i::Int, j::Int) = (m.data[i, j] = val)
Base.setindex!(m::SymMatrix, val::AbstractNode, i::Int, j::Int) = (m.data[i, j] = SymNumber(val))
Base.setindex!(m::SymMatrix, val::SymNumber, i::Int) = (m.data[i] = val)
Base.setindex!(m::SymMatrix, val::AbstractNode, i::Int) = (m.data[i] = SymNumber(val))
Base.setindex!(a::SymArray, val::SymNumber, i::Int) = (a.data[i] = val)
Base.setindex!(a::SymArray, val::AbstractNode, i::Int) = (a.data[i] = SymNumber(val))
Base.setindex!(a::SymArray{N}, val::SymNumber, I::Vararg{Int,N}) where {N} = (a.data[I...] = val)
Base.setindex!(a::SymArray{N}, val::AbstractNode, I::Vararg{Int,N}) where {N} = (a.data[I...] = SymNumber(val))

# Slicing - returns wrapped types
Base.getindex(v::SymVector, r::AbstractRange) = SymVector(v.data[r])
Base.getindex(v::SymVector, inds::AbstractVector{Int}) = SymVector(v.data[inds])

Base.getindex(m::SymMatrix, ::Colon, j::Int) = SymVector(m.data[:, j])
Base.getindex(m::SymMatrix, i::Int, ::Colon) = SymVector(m.data[i, :])
Base.getindex(m::SymMatrix, r1::AbstractRange, r2::AbstractRange) = SymMatrix(m.data[r1, r2])

# Similar - returns wrapped type
Base.similar(v::SymVector) = SymVector(similar(v.data))
Base.similar(v::SymVector, ::Type{SymNumber}) = SymVector(similar(v.data))
Base.similar(v::SymVector, ::Type{SymNumber}, dims::Tuple{Vararg{Int}}) =
    SymVector(similar(v.data, SymNumber, dims))

Base.similar(m::SymMatrix) = SymMatrix(similar(m.data))
Base.similar(m::SymMatrix, ::Type{SymNumber}) = SymMatrix(similar(m.data))
Base.similar(m::SymMatrix, ::Type{SymNumber}, dims::Tuple{Vararg{Int}}) =
    SymMatrix(similar(m.data, SymNumber, dims))

Base.similar(a::SymArray{N}) where {N} = SymArray{N}(similar(a.data))
Base.similar(a::SymArray{N}, ::Type{SymNumber}) where {N} = SymArray{N}(similar(a.data))

# =============================================================================
# 6. CONVERSION
# =============================================================================

# Convert AbstractArray{<:AbstractNode} to wrapped types
Base.convert(::Type{SymVector}, v::AbstractVector{<:AbstractNode}) = SymVector(v)
Base.convert(::Type{SymMatrix}, m::AbstractMatrix{<:AbstractNode}) = SymMatrix(m)

# Convert wrapped types back to regular arrays
Base.convert(::Type{Vector{<:AbstractNode}}, v::SymVector) = unwrap(v)
Base.convert(::Type{Matrix{<:AbstractNode}}, m::SymMatrix) = unwrap(m)

# =============================================================================
# 7. SYMBOLIC ARITHMETIC HELPERS
# =============================================================================

# Symbolic addition with Null(nothing) as additive identity
function sym_add(a::SymNumber, b::SymNumber)
    av = unwrap_scalar(a)
    bv = unwrap_scalar(b)
    av isa Null{Nothing} && return b
    bv isa Null{Nothing} && return a
    return SymNumber(av + bv)
end

sym_add(a::AbstractNode, b::AbstractNode) = sym_add(SymNumber(a), SymNumber(b))
sym_add(a::SymNumber, b::AbstractNode) = sym_add(a, SymNumber(b))
sym_add(a::AbstractNode, b::SymNumber) = sym_add(SymNumber(a), b)

# Symbolic multiplication
sym_mul(a::SymNumber, b::SymNumber) = a * b
sym_mul(a::AbstractNode, b::AbstractNode) = SymNumber(a) * SymNumber(b)
sym_mul(a::SymNumber, b::AbstractNode) = a * SymNumber(b)
sym_mul(a::AbstractNode, b::SymNumber) = SymNumber(a) * b
sym_mul(a::SymNumber, b::Number) = a * b
sym_mul(a::Number, b::SymNumber) = a * b

# =============================================================================
# 8. MATRIX-VECTOR PRODUCTS
# =============================================================================

# Numeric matrix x Symbolic vector -> SymVector
function Base.:*(A::AbstractMatrix{<:Number}, x::SymVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(undef, m)
    for i in 1:m
        acc = zero(SymNumber)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# Disambiguation: Diagonal x SymVector
function Base.:*(D::LinearAlgebra.Diagonal{<:Number}, x::SymVector)
    n = length(D.diag)
    @assert n == length(x) "Dimension mismatch: diagonal has $(n) elements, vector has $(length(x)) elements"

    result = SymVector(undef, n)
    for i in 1:n
        result[i] = SymNumber(D.diag[i] * unwrap_scalar(x[i]))
    end
    return result
end

# Symbolic matrix x Numeric vector -> SymVector
function Base.:*(A::SymMatrix, x::AbstractVector{<:Number})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(undef, m)
    for i in 1:m
        acc = zero(SymNumber)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# Symbolic matrix x Symbolic vector -> SymVector
function Base.:*(A::SymMatrix, x::SymVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(undef, m)
    for i in 1:m
        acc = zero(SymNumber)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# =============================================================================
# 9. ROW VECTOR x MATRIX (via Adjoint)
# =============================================================================

# Symbolic row x Numeric matrix -> Adjoint{SymVector}
function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}, A::AbstractMatrix{<:Number})
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    result = SymVector(undef, p)
    for j in 1:p
        acc = zero(SymNumber)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# Numeric row x Symbolic matrix -> Adjoint{SymVector}
# Use DenseVector to avoid ambiguity with other Adjoint methods
function Base.:*(x::LinearAlgebra.Adjoint{<:Number, <:DenseVector{<:Number}}, A::SymMatrix)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    result = SymVector(undef, p)
    for j in 1:p
        acc = zero(SymNumber)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# Disambiguation: Generic Adjoint x SymMatrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Number, <:AbstractVector}, A::SymMatrix)
    # Delegate to the DenseVector version by converting parent to array
    xp_dense = collect(parent(x))
    return adjoint(xp_dense) * A
end

# Symbolic row x Symbolic matrix -> Adjoint{SymVector}
function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}, A::SymMatrix)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    result = SymVector(undef, p)
    for j in 1:p
        acc = zero(SymNumber)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# =============================================================================
# 10. MATRIX x MATRIX PRODUCTS
# =============================================================================

# Numeric matrix x Symbolic matrix -> SymMatrix
function Base.:*(A::AbstractMatrix{<:Number}, B::SymMatrix)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    result = SymMatrix(undef, m, p)
    for i in 1:m
        for j in 1:p
            acc = zero(SymNumber)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# Symbolic matrix x Numeric matrix -> SymMatrix
function Base.:*(A::SymMatrix, B::AbstractMatrix{<:Number})
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    result = SymMatrix(undef, m, p)
    for i in 1:m
        for j in 1:p
            acc = zero(SymNumber)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# Symbolic matrix x Symbolic matrix -> SymMatrix
function Base.:*(A::SymMatrix, B::SymMatrix)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    result = SymMatrix(undef, m, p)
    for i in 1:m
        for j in 1:p
            acc = zero(SymNumber)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# =============================================================================
# 11. DOT PRODUCTS (return SymNumber scalar)
# =============================================================================

# Internal implementation of dot product
function _sym_dot(x, y, nx::Int, ny::Int)
    @assert nx == ny "Dimension mismatch: vectors have lengths $(nx) and $(ny)"
    acc = zero(SymNumber)
    for i in 1:nx
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# Symbolic dot Symbolic -> SymNumber
function LinearAlgebra.dot(x::SymVector, y::SymVector)
    return _sym_dot(x, y, length(x), length(y))
end

# Numeric dot Symbolic -> SymNumber
# Use DenseVector to be more specific and avoid ambiguity with SparseArrays
function LinearAlgebra.dot(x::DenseVector{<:Number}, y::SymVector)
    return _sym_dot(x, y, length(x), length(y))
end

# Symbolic dot Numeric -> SymNumber
# Use DenseVector to be more specific and avoid ambiguity with SparseArrays
function LinearAlgebra.dot(x::SymVector, y::DenseVector{<:Number})
    return _sym_dot(x, y, length(x), length(y))
end

# =============================================================================
# 11a. SPARSE VECTOR DISAMBIGUATION
# =============================================================================
# These methods resolve ambiguities with SparseArrays.dot methods.
# SparseArrays defines:
#   dot(::AbstractCompressedVector, ::AbstractVector)
#   dot(::AbstractVector, ::AbstractCompressedVector)
# We need explicit methods for sparse vectors with SymVector.

# Type alias for SparseArrays' compressed vector types
const SparseVecLike = Union{
    AbstractCompressedVector,
    SubArray{<:Any, 1, <:AbstractSparseMatrixCSC, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, false},
    SubArray{<:Any, 1, <:AbstractSparseVector, Tuple{Base.Slice{Base.OneTo{Int}}}, false}
}

# Sparse vector dot SymVector -> SymNumber (disambiguation)
function LinearAlgebra.dot(x::SparseVecLike, y::SymVector)
    return _sym_dot(x, y, length(x), length(y))
end

# SymVector dot Sparse vector -> SymNumber (disambiguation)
function LinearAlgebra.dot(x::SymVector, y::SparseVecLike)
    return _sym_dot(x, y, length(x), length(y))
end

# =============================================================================
# 11b. ADDITIONAL DISAMBIGUATION FOR ADJOINT/TRANSPOSE
# =============================================================================

# Disambiguate: Adjoint{SymNumber, SymVector} * Adjoint{Number, Transpose{T, Vector}}
# This handles x' * (v')' cases
function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector},
                 y::LinearAlgebra.Adjoint{<:Number, <:LinearAlgebra.Transpose{<:Any, <:AbstractVector}})
    # x' * (v')' = x' * v = dot(x, v)
    # Unwrap the double transpose and compute dot product
    return dot(parent(x), parent(parent(y)))
end

# Disambiguate: (Adjoint or Transpose of numeric vector) * SymVector
# This handles row vector * column vector -> scalar
function Base.:*(x::Union{LinearAlgebra.Adjoint{<:Number, <:AbstractVector},
                         LinearAlgebra.Transpose{<:Number, <:AbstractVector}},
                y::SymVector)
    # Row vector * column vector = dot product (scalar)
    return dot(parent(x), y)
end

# Disambiguate: Transpose{Number, Vector} * SymMatrix
# This handles row vector * matrix -> row vector
function Base.:*(x::LinearAlgebra.Transpose{<:Number, <:AbstractVector}, A::SymMatrix)
    # Convert transpose to adjoint and delegate
    # For numeric vectors, transpose = adjoint
    return adjoint(parent(x)) * A
end

# =============================================================================
# 12. INNER PRODUCTS VIA ADJOINT (x' * y)
# =============================================================================

# These delegate to dot products, which return SymNumber

function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}, y::SymVector)
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}, y::DenseVector{<:Number})
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{<:Number, <:DenseVector{<:Number}}, y::SymVector)
    return dot(parent(x), y)
end

# Disambiguation with LinearAlgebra.Transpose
function Base.:*(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}, y::LinearAlgebra.Transpose{<:Number, <:LinearAlgebra.Adjoint{<:Any, <:AbstractVector}})
    # This is an edge case: SymVector' * (v')'
    # Just delegate to the unwrapped version
    return x * parent(parent(y))
end

# =============================================================================
# 13. MATRIX ADJOINT / TRANSPOSE
# =============================================================================

# Vector adjoint returns Adjoint wrapper (for x' * A and x' * y syntax)
Base.adjoint(v::SymVector) = LinearAlgebra.Adjoint(v)
Base.transpose(v::SymVector) = LinearAlgebra.Transpose(v)

# Matrix adjoint/transpose return wrapped SymMatrix
function Base.adjoint(A::SymMatrix)
    # For symbolic (assumed real), adjoint = transpose
    return SymMatrix(permutedims(A.data))
end

function Base.transpose(A::SymMatrix)
    return SymMatrix(permutedims(A.data))
end

# Adjoint of adjoint returns the original
Base.adjoint(x::LinearAlgebra.Adjoint{SymNumber, <:SymVector}) = parent(x)
Base.transpose(x::LinearAlgebra.Transpose{SymNumber, <:SymVector}) = parent(x)

# =============================================================================
# 14. NORMS (return SymNumber scalar)
# =============================================================================

# norm_sqr for vectors
function LinearAlgebra.norm_sqr(x::SymVector)
    return dot(x, x)
end

# L2 norm (default)
function LinearAlgebra.norm(x::SymVector)
    return sqrt(norm_sqr(x))
end

# Lp norm
function LinearAlgebra.norm(x::SymVector, p::Real)
    if p == 2
        return norm(x)
    elseif p == 1
        # L1 norm: sum of absolute values
        acc = zero(SymNumber)
        for i in 1:length(x)
            acc = sym_add(acc, abs(x[i]))
        end
        return acc
    elseif p == Inf
        error("Infinity norm not yet implemented for symbolic vectors")
    else
        # General Lp norm
        acc = zero(SymNumber)
        for i in 1:length(x)
            acc = sym_add(acc, abs(x[i])^p)
        end
        return acc^(1/p)
    end
end

# Frobenius norm for matrices
function LinearAlgebra.norm(A::SymMatrix)
    m, n = size(A)
    acc = zero(SymNumber)
    for i in 1:m
        for j in 1:n
            acc = sym_add(acc, abs2(A[i, j]))
        end
    end
    return sqrt(acc)
end

# =============================================================================
# 15. BROADCASTING
# =============================================================================

# Custom broadcast style for symbolic wrappers
struct SymArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

SymArrayStyle(::Val{N}) where N = SymArrayStyle{N}()
SymArrayStyle{M}(::Val{N}) where {M,N} = SymArrayStyle{N}()

# Register style for wrapper types
Base.BroadcastStyle(::Type{<:SymVector}) = SymArrayStyle{1}()
Base.BroadcastStyle(::Type{<:SymMatrix}) = SymArrayStyle{2}()
Base.BroadcastStyle(::Type{<:SymArray{N}}) where {N} = SymArrayStyle{N}()

# SymArrayStyle wins over DefaultArrayStyle
Base.BroadcastStyle(::SymArrayStyle{N}, ::DefaultArrayStyle{M}) where {N,M} =
    SymArrayStyle{max(N,M)}()
Base.BroadcastStyle(::DefaultArrayStyle{M}, ::SymArrayStyle{N}) where {N,M} =
    SymArrayStyle{max(N,M)}()

# Combine SymArrayStyles
Base.BroadcastStyle(::SymArrayStyle{N}, ::SymArrayStyle{M}) where {N,M} =
    SymArrayStyle{max(N,M)}()

# Allocate output for broadcast - returns wrapped type
function Base.similar(bc::Broadcasted{SymArrayStyle{1}}, ::Type{SymNumber})
    sz = map(length, axes(bc))
    return SymVector(undef, sz[1])
end

function Base.similar(bc::Broadcasted{SymArrayStyle{2}}, ::Type{SymNumber})
    sz = map(length, axes(bc))
    return SymMatrix(undef, sz...)
end

function Base.similar(bc::Broadcasted{SymArrayStyle{N}}, ::Type{SymNumber}) where {N}
    sz = map(length, axes(bc))
    return SymArray{N}(undef, sz...)
end

# Materialize broadcast
function Base.copy(bc::Broadcasted{SymArrayStyle{1}})
    result = similar(bc, SymNumber)
    @inbounds for I in CartesianIndices(result.data)
        val = bc[I]
        result.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return result
end

function Base.copy(bc::Broadcasted{SymArrayStyle{2}})
    result = similar(bc, SymNumber)
    @inbounds for I in CartesianIndices(result.data)
        val = bc[I]
        result.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return result
end

function Base.copy(bc::Broadcasted{SymArrayStyle{N}}) where N
    result = similar(bc, SymNumber)
    @inbounds for I in CartesianIndices(result.data)
        val = bc[I]
        result.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return result
end

# In-place broadcast
function Base.copyto!(dest::SymVector, bc::Broadcasted{SymArrayStyle{1}})
    @inbounds for I in CartesianIndices(dest.data)
        val = bc[I]
        dest.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return dest
end

function Base.copyto!(dest::SymMatrix, bc::Broadcasted{SymArrayStyle{2}})
    @inbounds for I in CartesianIndices(dest.data)
        val = bc[I]
        dest.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return dest
end

function Base.copyto!(dest::SymArray{N}, bc::Broadcasted{SymArrayStyle{N}}) where {N}
    @inbounds for I in CartesianIndices(dest.data)
        val = bc[I]
        dest.data[I] = val isa SymNumber ? val : SymNumber(val)
    end
    return dest
end
