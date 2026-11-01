# Type aliases, promotion, zero/one and adjoint for nodes.
# Part of the extension CTParserExaModels, included by CTParserExaModels.jl.

# ============================================================================
# Type aliases, promotion, and adjoint for nodes
# ============================================================================

const ExaNode = Union{
    ExaModels.AbstractNode,ExaModels.AbstractAdjointNode,ExaModels.AbstractSecondAdjointNode
}
# Arrays of nodes, as `@def` builds them (dense vectors and matrices, from comprehensions)
# and as user code may slice or reshape them (strided views, reshapes, reinterprets).
# `Strided`, not `Abstract`: a method on `AbstractVector{<:ExaNode}` is ambiguous with every
# method another package defines on its own array type (`Diagonal`, sparse, `SMatrix`, …),
# and those are the constants users write. The constants keep `AbstractVector{<:Real}` /
# `AbstractMatrix{<:Real}` in the signatures; the few collisions left there have their own
# methods (structured.jl, and the SparseArrays and StaticArrays extensions).
const VecExaNode = StridedVector{<:ExaNode}
const MatExaNode = StridedMatrix{<:ExaNode}
# CPU arrays of nodes, for `norm`: GPUArrays also defines it on `AnyGPUArray`, which is a
# `DenseArray`, hence `Strided`, so a `Strided` signature would be ambiguous with it.
const CPUVecExaNode = Union{
    Vector{<:ExaNode},SubArray{<:ExaNode,1,<:Array},Base.ReshapedArray{<:ExaNode,1,<:Array}
}
const CPUMatExaNode = Union{
    Matrix{<:ExaNode},SubArray{<:ExaNode,2,<:Array},Base.ReshapedArray{<:ExaNode,2,<:Array}
}

# Type promotion: [x, 0] should give Vector{AbstractNode} with Null(0), not Vector{Any}
Base.promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Real}) = ExaModels.AbstractNode
function Base.convert(::Type{ExaModels.AbstractNode}, x::Real)
    return iszero(x) ? zero(ExaModels.AbstractNode) : ExaModels.Null(x)
end

# zero/one for ExaNode types — needed by stdlib (e.g. tr) and general array ops
Base.zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(0)
Base.zero(::ExaNode) = ExaModels.Null(0)
Base.one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)
Base.one(::ExaNode) = ExaModels.Null(1)

# adjoint/transpose for scalar ExaNode — nodes are real-valued, so both are identity
Base.adjoint(x::ExaModels.AbstractNode) = x
Base.adjoint(x::ExaModels.AbstractAdjointNode) = x
Base.adjoint(x::ExaModels.AbstractSecondAdjointNode) = x
Base.transpose(x::ExaModels.AbstractNode) = x
Base.transpose(x::ExaModels.AbstractAdjointNode) = x
Base.transpose(x::ExaModels.AbstractSecondAdjointNode) = x

# adjoint/transpose for matrices of ExaNode — materialize to plain Matrix
Base.adjoint(A::MatExaNode) = [A[j, i] for i in axes(A, 2), j in axes(A, 1)]
Base.transpose(A::MatExaNode) = [A[j, i] for i in axes(A, 2), j in axes(A, 1)]

# Dispatch pair constants for 3-way type combos
const _VEC_PAIRS = [
    (VecExaNode, VecExaNode),
    (AbstractVector{<:Real}, VecExaNode),
    (VecExaNode, AbstractVector{<:Real}),
]
const _MAT_PAIRS = [
    (MatExaNode, MatExaNode),
    (AbstractMatrix{<:Real}, MatExaNode),
    (MatExaNode, AbstractMatrix{<:Real}),
]
