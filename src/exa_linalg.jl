# =============================================================================
# exa_linalg.jl - Symbolic array wrappers for ExaModels.AbstractNode
#
# This file defines wrapper types (SymVector, SymMatrix) that eliminate type
# piracy on array operations while maintaining full ExaModels compatibility.
#
# Key design principles:
# - Wrapper types own the dispatch (no type piracy on array ops)
# - Scalar operations still use AbstractNode (unavoidable type piracy)
# - Array operations return wrapped types
# - Scalar-returning operations return unwrapped AbstractNode
# =============================================================================

using ExaModels: AbstractNode, Null
using LinearAlgebra
using LinearAlgebra: norm_sqr
using Base.Broadcast: Broadcasted, DefaultArrayStyle

# =============================================================================
# EXPORTS
# =============================================================================
export sym_add, sym_mul, norm_sqr
export SymVector, SymMatrix, SymArray, AbstractSymArray
export SymNumber  # Type alias for AbstractNode

# =============================================================================
# 1. TYPE DEFINITIONS
# =============================================================================

# Abstract supertype for all symbolic array wrappers
abstract type AbstractSymArray{T<:AbstractNode, N} <: AbstractArray{T, N} end

# Symbolic vector wrapper
struct SymVector{T<:AbstractNode} <: AbstractSymArray{T, 1}
    data::Vector{T}
end

# Symbolic matrix wrapper
struct SymMatrix{T<:AbstractNode} <: AbstractSymArray{T, 2}
    data::Matrix{T}
end

# Generic N-dimensional wrapper (for future extensibility)
struct SymArray{T<:AbstractNode, N} <: AbstractSymArray{T, N}
    data::Array{T, N}
end

# Type alias for scalars (just AbstractNode, not a wrapper)
const SymNumber = AbstractNode

# =============================================================================
# 2. CONSTRUCTORS
# =============================================================================

# Convenience constructors from AbstractArray (copy to concrete array)
SymVector(v::AbstractVector{<:AbstractNode}) = SymVector(collect(v))
SymMatrix(m::AbstractMatrix{<:AbstractNode}) = SymMatrix(collect(m))

# Convenience constructors with UndefInitializer
SymVector{T}(::UndefInitializer, n::Int) where T<:AbstractNode = SymVector{T}(Vector{T}(undef, n))
SymMatrix{T}(::UndefInitializer, m::Int, n::Int) where T<:AbstractNode = SymMatrix{T}(Matrix{T}(undef, m, n))
SymArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}) where {T<:AbstractNode, N} = SymArray{T,N}(Array{T,N}(undef, dims...))

# =============================================================================
# 3. ARRAY INTERFACE
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

# Indexing - CRITICAL: Returns unwrapped AbstractNode
Base.getindex(v::SymVector, i::Int) = v.data[i]
Base.getindex(m::SymMatrix, i::Int, j::Int) = m.data[i, j]
Base.getindex(m::SymMatrix, i::Int) = m.data[i]  # Linear indexing
Base.getindex(a::SymArray, i::Int) = a.data[i]
Base.getindex(a::SymArray{T,N}, I::Vararg{Int,N}) where {T,N} = a.data[I...]

# Setindex - mutable construction
Base.setindex!(v::SymVector, val, i::Int) = (v.data[i] = val)
Base.setindex!(m::SymMatrix, val, i::Int, j::Int) = (m.data[i, j] = val)
Base.setindex!(m::SymMatrix, val, i::Int) = (m.data[i] = val)
Base.setindex!(a::SymArray, val, i::Int) = (a.data[i] = val)
Base.setindex!(a::SymArray{T,N}, val, I::Vararg{Int,N}) where {T,N} = (a.data[I...] = val)

# Slicing - returns wrapped types
Base.getindex(v::SymVector, r::AbstractRange) = SymVector(v.data[r])
Base.getindex(v::SymVector, inds::AbstractVector{Int}) = SymVector(v.data[inds])

Base.getindex(m::SymMatrix, ::Colon, j::Int) = SymVector(m.data[:, j])
Base.getindex(m::SymMatrix, i::Int, ::Colon) = SymVector(m.data[i, :])
Base.getindex(m::SymMatrix, r1::AbstractRange, r2::AbstractRange) = SymMatrix(m.data[r1, r2])

# Similar - returns wrapped type
Base.similar(v::SymVector) = SymVector(similar(v.data))
Base.similar(v::SymVector, ::Type{T}) where T<:AbstractNode = SymVector(similar(v.data, T))
Base.similar(v::SymVector, ::Type{T}, dims::Tuple{Vararg{Int}}) where T<:AbstractNode =
    SymVector(similar(v.data, T, dims))

Base.similar(m::SymMatrix) = SymMatrix(similar(m.data))
Base.similar(m::SymMatrix, ::Type{T}) where T<:AbstractNode = SymMatrix(similar(m.data, T))
Base.similar(m::SymMatrix, ::Type{T}, dims::Tuple{Vararg{Int}}) where T<:AbstractNode =
    SymMatrix(similar(m.data, T, dims))

Base.similar(a::SymArray{T,N}) where {T,N} = SymArray(similar(a.data))
Base.similar(a::SymArray{T,N}, ::Type{S}) where {T,N,S<:AbstractNode} = SymArray(similar(a.data, S))

# =============================================================================
# 4. UNWRAP HELPER
# =============================================================================

# Extract underlying data from wrappers
unwrap(v::SymVector) = v.data
unwrap(m::SymMatrix) = m.data
unwrap(a::SymArray) = a.data
unwrap(x::AbstractNode) = x  # Pass-through for scalars
unwrap(x::Number) = x         # Pass-through for numbers
unwrap(x::AbstractArray) = x  # Pass-through for regular arrays

# =============================================================================
# 5. CONVERSION
# =============================================================================

# Convert AbstractArray{<:AbstractNode} to wrapped types
Base.convert(::Type{SymVector}, v::AbstractVector{<:AbstractNode}) = SymVector(collect(v))
Base.convert(::Type{SymMatrix}, m::AbstractMatrix{<:AbstractNode}) = SymMatrix(collect(m))

# Convert wrapped types back to regular arrays
Base.convert(::Type{Vector{T}}, v::SymVector{T}) where T = v.data
Base.convert(::Type{Matrix{T}}, m::SymMatrix{T}) where T = m.data

# =============================================================================
# 6. SCALAR AbstractNode OPERATIONS (unavoidable type piracy)
# =============================================================================

Base.zero(::Type{<:AbstractNode}) = Null(nothing)
Base.zero(::AbstractNode) = Null(nothing)
Base.one(::Type{<:AbstractNode}) = Null(1)
Base.one(::AbstractNode) = Null(1)

Base.adjoint(x::AbstractNode) = x
Base.transpose(x::AbstractNode) = x
Base.conj(x::AbstractNode) = x

Base.broadcastable(x::AbstractNode) = Ref(x)
Base.iterate(::AbstractNode) = nothing
Base.length(::AbstractNode) = 1
Base.size(::AbstractNode) = ()
Base.ndims(::AbstractNode) = 0
Base.ndims(::Type{<:AbstractNode}) = 0
Base.IteratorSize(::Type{<:AbstractNode}) = Base.HasShape{0}()

Base.promote_rule(::Type{<:AbstractNode}, ::Type{<:Number}) = AbstractNode
Base.convert(::Type{AbstractNode}, x::Number) = Null(x)
Base.convert(::Type{AbstractNode}, x::AbstractNode) = x

# =============================================================================
# 7. SYMBOLIC ARITHMETIC HELPERS
# =============================================================================

# Symbolic addition with Null(nothing) as additive identity
function sym_add(a, b)
    a isa Null{Nothing} && return b
    b isa Null{Nothing} && return a
    return a + b
end

# Symbolic multiplication
sym_mul(a, b) = a * b

# =============================================================================
# 8. MATRIX-VECTOR PRODUCTS
# =============================================================================

# Numeric matrix × Symbolic vector -> SymVector
function Base.:*(A::AbstractMatrix{<:Number}, x::SymVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(Vector{AbstractNode}(undef, m))
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# Symbolic matrix × Numeric vector -> SymVector
function Base.:*(A::SymMatrix, x::AbstractVector{<:Number})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(Vector{AbstractNode}(undef, m))
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# Symbolic matrix × Symbolic vector -> SymVector
function Base.:*(A::SymMatrix, x::SymVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: matrix has $(n) columns, vector has $(length(x)) elements"

    result = SymVector(Vector{AbstractNode}(undef, m))
    for i in 1:m
        acc = Null(nothing)
        for j in 1:n
            acc = sym_add(acc, sym_mul(A[i, j], x[j]))
        end
        result[i] = acc
    end
    return result
end

# =============================================================================
# 9. ROW VECTOR × MATRIX (via Adjoint)
# =============================================================================

# Symbolic row × Numeric matrix -> Adjoint{SymVector}
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymVector}, A::AbstractMatrix{<:Number})
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    # T = eltype(xp)
    result = SymVector(Vector{AbstractNode}(undef, p))
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# Numeric row × Symbolic matrix -> Adjoint{SymVector}
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}}, A::SymMatrix)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    # T = eltype(A)
    result = SymVector(Vector{AbstractNode}(undef, p))
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# Symbolic row × Symbolic matrix -> Adjoint{SymVector}
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymVector}, A::SymMatrix)
    xp = parent(x)
    n, p = size(A)
    @assert length(xp) == n "Dimension mismatch: vector has $(length(xp)) elements, matrix has $(n) rows"

    # T = promote_type(eltype(xp), eltype(A))
    result = SymVector(Vector{AbstractNode}(undef, p))
    for j in 1:p
        acc = Null(nothing)
        for i in 1:n
            acc = sym_add(acc, sym_mul(xp[i], A[i, j]))
        end
        result[j] = acc
    end
    return adjoint(result)
end

# =============================================================================
# 10. MATRIX × MATRIX PRODUCTS
# =============================================================================

# Numeric matrix × Symbolic matrix -> SymMatrix
function Base.:*(A::AbstractMatrix{<:Number}, B::SymMatrix)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    # T = eltype(B)
    result = SymMatrix(Matrix{AbstractNode}(undef, m, p))
    for i in 1:m
        for j in 1:p
            acc = Null(nothing)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# Symbolic matrix × Numeric matrix -> SymMatrix
function Base.:*(A::SymMatrix, B::AbstractMatrix{<:Number})
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    # T = eltype(A)
    result = SymMatrix(Matrix{AbstractNode}(undef, m, p))
    for i in 1:m
        for j in 1:p
            acc = Null(nothing)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# Symbolic matrix × Symbolic matrix -> SymMatrix
function Base.:*(A::SymMatrix, B::SymMatrix)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch: first matrix has $(n) columns, second has $(n2) rows"

    # T = promote_type(eltype(A), eltype(B))
    result = SymMatrix(Matrix{AbstractNode}(undef, m, p))
    for i in 1:m
        for j in 1:p
            acc = Null(nothing)
            for k in 1:n
                acc = sym_add(acc, sym_mul(A[i, k], B[k, j]))
            end
            result[i, j] = acc
        end
    end
    return result
end

# =============================================================================
# 11. DOT PRODUCTS (return unwrapped AbstractNode scalar)
# =============================================================================

# Symbolic dot Symbolic -> AbstractNode
function LinearAlgebra.dot(x::SymVector, y::SymVector)
    n = length(x)
    @assert n == length(y) "Dimension mismatch: vectors have lengths $(n) and $(length(y))"

    acc = Null(nothing)
    for i in 1:n
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# Numeric dot Symbolic -> AbstractNode
function LinearAlgebra.dot(x::AbstractVector{<:Number}, y::SymVector)
    n = length(x)
    @assert n == length(y) "Dimension mismatch: vectors have lengths $(n) and $(length(y))"

    acc = Null(nothing)
    for i in 1:n
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# Symbolic dot Numeric -> AbstractNode
function LinearAlgebra.dot(x::SymVector, y::AbstractVector{<:Number})
    n = length(x)
    @assert n == length(y) "Dimension mismatch: vectors have lengths $(n) and $(length(y))"

    acc = Null(nothing)
    for i in 1:n
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# =============================================================================
# 12. INNER PRODUCTS VIA ADJOINT (x' * y)
# =============================================================================

# These delegate to dot products, which return unwrapped AbstractNode

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymVector}, y::SymVector)
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymVector}, y::AbstractVector{<:Number})
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}}, y::SymVector)
    return dot(parent(x), y)
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
Base.adjoint(x::LinearAlgebra.Adjoint{<:Any, <:SymVector}) = parent(x)
Base.transpose(x::LinearAlgebra.Transpose{<:Any, <:SymVector}) = parent(x)

# =============================================================================
# 14. NORMS (return unwrapped AbstractNode scalar)
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
        acc = Null(nothing)
        for i in 1:length(x)
            acc = sym_add(acc, abs(x[i]))
        end
        return acc
    elseif p == Inf
        error("Infinity norm not yet implemented for symbolic vectors")
    else
        # General Lp norm
        acc = Null(nothing)
        for i in 1:length(x)
            acc = sym_add(acc, abs(x[i])^p)
        end
        return acc^(1/p)
    end
end

# Frobenius norm for matrices
function LinearAlgebra.norm(A::SymMatrix)
    m, n = size(A)
    acc = Null(nothing)
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
Base.BroadcastStyle(::Type{<:SymArray{T,N}}) where {T,N} = SymArrayStyle{N}()

# SymArrayStyle wins over DefaultArrayStyle
Base.BroadcastStyle(::SymArrayStyle{N}, ::DefaultArrayStyle{M}) where {N,M} =
    SymArrayStyle{max(N,M)}()
Base.BroadcastStyle(::DefaultArrayStyle{M}, ::SymArrayStyle{N}) where {N,M} =
    SymArrayStyle{max(N,M)}()

# Combine SymArrayStyles
Base.BroadcastStyle(::SymArrayStyle{N}, ::SymArrayStyle{M}) where {N,M} =
    SymArrayStyle{max(N,M)}()

# Allocate output for broadcast - returns wrapped type
function Base.similar(bc::Broadcasted{SymArrayStyle{1}}, ::Type{T}) where T
    sz = map(length, axes(bc))
    return SymVector(Vector{AbstractNode}(undef, sz[1]))
end

function Base.similar(bc::Broadcasted{SymArrayStyle{2}}, ::Type{T}) where T
    sz = map(length, axes(bc))
    return SymMatrix(Matrix{AbstractNode}(undef, sz...))
end

function Base.similar(bc::Broadcasted{SymArrayStyle{N}}, ::Type{T}) where {N, T}
    sz = map(length, axes(bc))
    return SymArray(Array{AbstractNode, N}(undef, sz...))
end

# Materialize broadcast
function Base.copy(bc::Broadcasted{SymArrayStyle{1}})
    result = similar(bc, AbstractNode)
    @inbounds for I in CartesianIndices(result.data)
        result.data[I] = bc[I]
    end
    return result
end

function Base.copy(bc::Broadcasted{SymArrayStyle{2}})
    result = similar(bc, AbstractNode)
    @inbounds for I in CartesianIndices(result.data)
        result.data[I] = bc[I]
    end
    return result
end

function Base.copy(bc::Broadcasted{SymArrayStyle{N}}) where N
    result = similar(bc, AbstractNode)
    @inbounds for I in CartesianIndices(result.data)
        result.data[I] = bc[I]
    end
    return result
end

# In-place broadcast
function Base.copyto!(dest::SymVector, bc::Broadcasted{SymArrayStyle{1}})
    @inbounds for I in CartesianIndices(dest.data)
        dest.data[I] = bc[I]
    end
    return dest
end

function Base.copyto!(dest::SymMatrix, bc::Broadcasted{SymArrayStyle{2}})
    @inbounds for I in CartesianIndices(dest.data)
        dest.data[I] = bc[I]
    end
    return dest
end

function Base.copyto!(dest::SymArray{T,N}, bc::Broadcasted{SymArrayStyle{N}}) where {T,N}
    @inbounds for I in CartesianIndices(dest.data)
        dest.data[I] = bc[I]
    end
    return dest
end
