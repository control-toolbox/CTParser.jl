# exa_linalg.jl

import Base: zero, adjoint, *
import LinearAlgebra: dot, Adjoint

zero(x::T) where {T <: ExaModels.AbstractNode} = 0

adjoint(x::ExaModels.AbstractNode) = x

## Convert Number to AbstractNode using Null
Base.convert(::Type{ExaModels.AbstractNode}, x::Number) = ExaModels.Null(x)
#
## Also handle the identity conversion
#Base.convert(::Type{T}, x::T) where {T<:ExaModels.AbstractNode} = x
#
## If you need more specific typing:
#Base.convert(::Type{ExaModels.Null{T}}, x::Number) where {T} = 
#    ExaModels.Null{typeof(x)}(x)

# works to define matrices (of vectors) with symbolic entries
Base.promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Number}) = ExaModels.AbstractNode

function dot(v::Vector{T}, x::Vector{S}) where {T, S <: ExaModels.AbstractNode}
    @assert length(v) == length(x)
    return sum(v .* x)
end

function *(A::Matrix{T}, x::Vector{S}) where {T, S <: ExaModels.AbstractNode}
    m, n = size(A)
    @assert n == length(x)
    return [dot(A[i, :], x) for i in 1:m]
end

function *(p::Adjoint{T, Vector{T}}, A::Matrix{S}) where {T <: ExaModels.AbstractNode, S}
    m, n = size(A)
    @assert m == length(p)
    return [p * A[:, j] for j in 1:n]'
end