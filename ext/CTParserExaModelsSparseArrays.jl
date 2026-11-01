"""
    CTParserExaModelsSparseArrays

Sparse numeric constants with arrays of ExaModels nodes, in the code `@def` emits for the
`:exa` backend: `dot(q, x(t))` with a sparse vector `q`, and a matrix of nodes times a
sparse vector.

`CTParserExaModels` defines these operations for any numeric `AbstractVector`, and
SparseArrays defines them for its sparse vectors with any other vector: for a sparse
vector with a vector of nodes, both apply and neither is more specific, so the call is
ambiguous (control-toolbox/CTParser.jl#365). The methods below cover exactly those
intersections, and only sum over the stored entries: the expression graph has one term per
nonzero, not one per index.

Type piracy, accepted as for `CTParserExaModels`: CTParser owns neither `LinearAlgebra.dot`,
`Base.:*`, the sparse vectors nor the node types.
"""
module CTParserExaModelsSparseArrays

using CTBase: CTBase
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays

# Same aliases as in `CTParserExaModels` (types.jl); a test checks they are equal.
const ExaNode = Union{
    ExaModels.AbstractNode,ExaModels.AbstractAdjointNode,ExaModels.AbstractSecondAdjointNode
}
const VecExaNode = StridedVector{T} where {T<:ExaNode}
const MatExaNode = StridedMatrix{T} where {T<:ExaNode}

# A sparse numeric vector, or a column or slice view of one. `SparseVectorUnion` is not
# public, but it is the exact type SparseArrays dispatches `dot` on, unchanged from Julia
# 1.10 to 1.12.
const RealSparseVector = SparseArrays.SparseVectorUnion{T} where {T<:Real}

function _dimension_mismatch(op, reason)
    throw(
        CTBase.PreconditionError(
            "operand dimensions do not match for $op";
            reason=reason,
            suggestion="pass operands with compatible dimensions",
            context="linear algebra on ExaModels nodes",
        ),
    )
end

# The stored entries of a sparse vector (or view), as (index, value) pairs, explicit zeros
# left out so that they add nothing to the graph.
function _stored(q)
    s = SparseArrays.SparseVector(q)
    return ((i, v) for (i, v) in zip(SparseArrays.findnz(s)...) if !iszero(v))
end

# Σ over the stored entries of q of f(q_i, x_i), starting from a structural zero, which the
# zero elimination of `CTParserExaModels` folds away at the first term.
function _sparse_sum(f, q, x)
    length(q) == length(x) ||
        _dimension_mismatch("dot", "lengths $(length(q)) and $(length(x))")
    s = ExaModels.Null(0)
    for (i, v) in _stored(q)
        s = s + f(v, x[i])
    end
    return s
end

LinearAlgebra.dot(q::RealSparseVector, x::VecExaNode) = _sparse_sum((v, xi) -> v * xi, q, x)
LinearAlgebra.dot(x::VecExaNode, q::RealSparseVector) = _sparse_sum((v, xi) -> xi * v, q, x)

# Matrix of nodes times a sparse numeric vector: each row is a sum over the stored entries.
function _row_sum(A, i, entries)
    s = ExaModels.Null(0)
    for (j, v) in entries
        s = s + A[i, j] * v
    end
    return s
end

function Base.:*(A::MatExaNode, q::SparseArrays.AbstractSparseVector{<:Real})
    size(A, 2) == length(q) || _dimension_mismatch(
        "*", "matrix of size $(size(A)) and vector of length $(length(q))"
    )
    entries = collect(_stored(q))
    return [_row_sum(A, i, entries) for i in 1:size(A, 1)]
end

# Matrix of nodes times a sparse numeric matrix. Up to Julia 1.10, SparseArrays defines `*`
# for any dense matrix and a sparse matrix, a column-slice view of one, or its adjoint or
# transpose, which meets `CTParserExaModels`' method on `AbstractMatrix{<:Real}` with neither
# more specific (#365). The product is the plain one: unlike for vectors, a sparse matrix
# product is not worth a stored-entries loop of its own here.
const RealSparseMatrix = Union{
    Union{
        SparseArrays.AbstractSparseMatrixCSC{T,Ti},
        SubArray{
            T,2,<:SparseArrays.AbstractSparseMatrixCSC{T,Ti},Tuple{Base.Slice{Base.OneTo{Int}},I}
        } where {I<:AbstractUnitRange},
    } where {T<:Real,Ti},
    LinearAlgebra.Adjoint{<:Real,<:SparseArrays.AbstractSparseMatrixCSC},
    LinearAlgebra.Transpose{<:Real,<:SparseArrays.AbstractSparseMatrixCSC},
}

function Base.:*(A::MatExaNode, B::RealSparseMatrix)
    size(A, 2) == size(B, 1) ||
        _dimension_mismatch("*", "matrices of sizes $(size(A)) and $(size(B))")
    return [
        _row_dot(A, i, view(B, :, j)) for i in 1:size(A, 1), j in 1:size(B, 2)
    ]
end

function _row_dot(A, i, b)
    s = ExaModels.Null(0)
    for k in eachindex(b)
        s = s + A[i, k] * b[k]
    end
    return s
end

end # module CTParserExaModelsSparseArrays
