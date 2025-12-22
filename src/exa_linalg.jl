using ExaModels: AbstractNode, Null
using LinearAlgebra
using LinearAlgebra: norm_sqr

# =============================================================================
# EXPORTS
# =============================================================================
export sym_add, sym_mul, norm_sqr

# =============================================================================
# 1. ZERO / ONE
# =============================================================================
Base.zero(::Type{<:AbstractNode}) = Null(nothing)
Base.zero(::AbstractNode) = Null(nothing)
Base.one(::Type{<:AbstractNode}) = Null(1)
Base.one(::AbstractNode) = Null(1)

# =============================================================================
# 2. ADJOINT / TRANSPOSE / CONJ for scalar nodes
# =============================================================================
Base.adjoint(x::AbstractNode) = x
Base.transpose(x::AbstractNode) = x
Base.conj(x::AbstractNode) = x

# =============================================================================
# 3. AbstractNode IS A SCALAR (not iterable, not a container)
# =============================================================================

# Tell Julia that AbstractNode is a scalar for broadcasting
Base.broadcastable(x::AbstractNode) = Ref(x)

# AbstractNode should not be iterated (it's a scalar)
Base.iterate(::AbstractNode) = nothing
Base.length(::AbstractNode) = 1
Base.size(::AbstractNode) = ()
Base.ndims(::AbstractNode) = 0
Base.ndims(::Type{<:AbstractNode}) = 0
Base.IteratorSize(::Type{<:AbstractNode}) = Base.HasShape{0}()

# =============================================================================
# 4. TYPE PROMOTION & CONVERSION
# =============================================================================
Base.promote_rule(::Type{<:AbstractNode}, ::Type{<:Number}) = AbstractNode
Base.convert(::Type{AbstractNode}, x::Number) = Null(x)
Base.convert(::Type{AbstractNode}, x::AbstractNode) = x

# =============================================================================
# 5. SYMBOLIC ARITHMETIC HELPERS
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
# 6. TYPE ALIASES
# =============================================================================

const SymbolicVector = AbstractVector{<:AbstractNode}
const SymbolicMatrix = AbstractMatrix{<:AbstractNode}
const SymbolicVecOrMat = Union{SymbolicVector, SymbolicMatrix}

# =============================================================================
# 7. MATRIX-VECTOR PRODUCTS
# =============================================================================

# Numeric matrix × Symbolic vector
function Base.:*(A::AbstractMatrix{<:Number}, x::SymbolicVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch: A is $m×$n, x has length $(length(x))"
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

# Symbolic matrix × Numeric vector
function Base.:*(A::SymbolicMatrix, x::AbstractVector{<:Number})
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch"
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

# Symbolic matrix × Symbolic vector
function Base.:*(A::SymbolicMatrix, x::SymbolicVector)
    m, n = size(A)
    @assert n == length(x) "Dimension mismatch"
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

# =============================================================================
# 8. ROW VECTOR × MATRIX (via Adjoint)
# =============================================================================

# Symbolic row × Numeric matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, A::AbstractMatrix{<:Number})
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

# Numeric row × Symbolic matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}}, A::SymbolicMatrix)
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

# Symbolic row × Symbolic matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, A::SymbolicMatrix)
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

# =============================================================================
# 9. MATRIX × MATRIX
# =============================================================================

# Numeric × Symbolic
function Base.:*(A::AbstractMatrix{<:Number}, B::SymbolicMatrix)
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

# Symbolic × Numeric
function Base.:*(A::SymbolicMatrix, B::AbstractMatrix{<:Number})
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

# Symbolic × Symbolic
function Base.:*(A::SymbolicMatrix, B::SymbolicMatrix)
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

# =============================================================================
# 10. DOT PRODUCTS
# =============================================================================

function LinearAlgebra.dot(x::SymbolicVector, y::SymbolicVector)
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

function LinearAlgebra.dot(x::AbstractVector{<:Number}, y::SymbolicVector)
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

function LinearAlgebra.dot(x::SymbolicVector, y::AbstractVector{<:Number})
    @assert length(x) == length(y) "Dimension mismatch"
    acc = Null(nothing)
    for i in eachindex(x, y)
        acc = sym_add(acc, sym_mul(x[i], y[i]))
    end
    return acc
end

# =============================================================================
# 11. INNER PRODUCTS VIA ADJOINT: x' * y
# =============================================================================

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, y::SymbolicVector)
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, y::AbstractVector{<:Number})
    return dot(parent(x), y)
end

function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}}, y::SymbolicVector)
    return dot(parent(x), y)
end

# =============================================================================
# 12. MATRIX ADJOINT / TRANSPOSE
# =============================================================================

function Base.adjoint(A::SymbolicMatrix)
    return permutedims(A)
end

function Base.transpose(A::SymbolicMatrix)
    return permutedims(A)
end

# =============================================================================
# 13. NORMS
# =============================================================================

function LinearAlgebra.norm_sqr(x::SymbolicVector)
    return dot(x, x)
end

# Symbolic norm: returns symbolic expression √(∑ xᵢ²)
function LinearAlgebra.norm(x::SymbolicVector)
    return sqrt(norm_sqr(x))
end

function LinearAlgebra.norm(x::SymbolicVector, p::Real)
    if p == 2
        return sqrt(norm_sqr(x))
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

# Matrix norms
function LinearAlgebra.norm(A::SymbolicMatrix)
    # Frobenius norm by default
    acc = Null(nothing)
    for i in eachindex(A)
        acc = sym_add(acc, A[i] * A[i])
    end
    return sqrt(acc)
end

# =============================================================================
# 14. BROADCASTING
# =============================================================================

using Base.Broadcast: Broadcasted, DefaultArrayStyle

# Custom broadcast style for symbolic arrays
struct SymbolicArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

SymbolicArrayStyle(::Val{N}) where N = SymbolicArrayStyle{N}()
SymbolicArrayStyle{M}(::Val{N}) where {M,N} = SymbolicArrayStyle{N}()

# Register style for symbolic arrays
Base.BroadcastStyle(::Type{<:AbstractArray{<:AbstractNode, N}}) where N = SymbolicArrayStyle{N}()

# SymbolicArrayStyle wins over DefaultArrayStyle
Base.BroadcastStyle(::SymbolicArrayStyle{N}, ::DefaultArrayStyle{M}) where {N,M} = SymbolicArrayStyle{max(N,M)}()
Base.BroadcastStyle(::DefaultArrayStyle{M}, ::SymbolicArrayStyle{N}) where {N,M} = SymbolicArrayStyle{max(N,M)}()

# Combine two SymbolicArrayStyles
Base.BroadcastStyle(::SymbolicArrayStyle{N}, ::SymbolicArrayStyle{M}) where {N,M} = SymbolicArrayStyle{max(N,M)}()

# Allocate output array for broadcast
function Base.similar(bc::Broadcasted{SymbolicArrayStyle{N}}, ::Type{T}) where {N, T}
    sz = map(length, axes(bc))
    return Array{AbstractNode, N}(undef, sz...)
end

# Materialize the broadcast
function Base.copy(bc::Broadcasted{SymbolicArrayStyle{N}}) where N
    sz = map(length, axes(bc))
    result = Array{AbstractNode, N}(undef, sz...)
    @inbounds for I in CartesianIndices(result)
        result[I] = bc[I]
    end
    return result
end

# In-place broadcast
function Base.copyto!(dest::AbstractArray{<:AbstractNode}, bc::Broadcasted{SymbolicArrayStyle{N}}) where N
    @inbounds for I in CartesianIndices(dest)
        dest[I] = bc[I]
    end
    return dest
end

# =============================================================================
# 15. ABS / ABS2 for symbolic nodes
# =============================================================================

# Note: abs2 is already defined in ExaModels, so we don't redefine it here
