using ExaModels: AbstractNode, Null
using LinearAlgebra
using LinearAlgebra: norm_sqr

# Load trait-based implementations
include("symbolic_ops.jl")

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

# Note: sym_add and sym_mul are defined in symbolic_ops.jl (included above)
# They are available here and exported for backward compatibility

# =============================================================================
# 6. TYPE ALIASES
# =============================================================================

const SymbolicVector = AbstractVector{<:AbstractNode}
const SymbolicMatrix = AbstractMatrix{<:AbstractNode}
const SymbolicVecOrMat = Union{SymbolicVector, SymbolicMatrix}

# =============================================================================
# 7. MATRIX-VECTOR PRODUCTS (using trait-based dispatch)
# =============================================================================

# Numeric matrix × Symbolic vector
function Base.:*(A::AbstractMatrix{<:Number}, x::SymbolicVector)
    return _matmul_vec(symbolic_trait(A), symbolic_trait(x), A, x)
end

# Symbolic matrix × Numeric vector
function Base.:*(A::SymbolicMatrix, x::AbstractVector{<:Number})
    return _matmul_vec(symbolic_trait(A), symbolic_trait(x), A, x)
end

# Symbolic matrix × Symbolic vector
function Base.:*(A::SymbolicMatrix, x::SymbolicVector)
    return _matmul_vec(symbolic_trait(A), symbolic_trait(x), A, x)
end

# =============================================================================
# 8. ROW VECTOR × MATRIX (via Adjoint, using trait-based dispatch)
# =============================================================================

# Symbolic row × Numeric matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, A::AbstractMatrix{<:Number})
    return _rowvec_matmul(symbolic_trait(parent(x)), symbolic_trait(A), x, A)
end

# Numeric row × Symbolic matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:AbstractVector{<:Number}}, A::SymbolicMatrix)
    return _rowvec_matmul(symbolic_trait(parent(x)), symbolic_trait(A), x, A)
end

# Symbolic row × Symbolic matrix
function Base.:*(x::LinearAlgebra.Adjoint{<:Any, <:SymbolicVector}, A::SymbolicMatrix)
    return _rowvec_matmul(symbolic_trait(parent(x)), symbolic_trait(A), x, A)
end

# =============================================================================
# 9. MATRIX × MATRIX (using trait-based dispatch)
# =============================================================================

# Numeric × Symbolic
function Base.:*(A::AbstractMatrix{<:Number}, B::SymbolicMatrix)
    return _matmul_mat(symbolic_trait(A), symbolic_trait(B), A, B)
end

# Symbolic × Numeric
function Base.:*(A::SymbolicMatrix, B::AbstractMatrix{<:Number})
    return _matmul_mat(symbolic_trait(A), symbolic_trait(B), A, B)
end

# Symbolic × Symbolic
function Base.:*(A::SymbolicMatrix, B::SymbolicMatrix)
    return _matmul_mat(symbolic_trait(A), symbolic_trait(B), A, B)
end

# =============================================================================
# 10. DOT PRODUCTS (using trait-based dispatch)
# =============================================================================

function LinearAlgebra.dot(x::SymbolicVector, y::SymbolicVector)
    return _dot_product(symbolic_trait(x), symbolic_trait(y), x, y)
end

function LinearAlgebra.dot(x::AbstractVector{<:Number}, y::SymbolicVector)
    return _dot_product(symbolic_trait(x), symbolic_trait(y), x, y)
end

function LinearAlgebra.dot(x::SymbolicVector, y::AbstractVector{<:Number})
    return _dot_product(symbolic_trait(x), symbolic_trait(y), x, y)
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
# 12. MATRIX ADJOINT / TRANSPOSE (using trait-based dispatch)
# =============================================================================

function Base.adjoint(A::SymbolicMatrix)
    return _adjoint_mat(symbolic_trait(A), A)
end

function Base.transpose(A::SymbolicMatrix)
    return _transpose_mat(symbolic_trait(A), A)
end

# =============================================================================
# 13. NORMS (using trait-based dispatch)
# =============================================================================

function LinearAlgebra.norm_sqr(x::SymbolicVector)
    return dot(x, x)
end

# Symbolic norm: returns symbolic expression √(∑ xᵢ²)
function LinearAlgebra.norm(x::SymbolicVector)
    return _norm_p(symbolic_trait(x), x, 2)
end

function LinearAlgebra.norm(x::SymbolicVector, p::Real)
    return _norm_p(symbolic_trait(x), x, p)
end

# Matrix norms
function LinearAlgebra.norm(A::SymbolicMatrix)
    return _norm_frob(symbolic_trait(A), A)
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
