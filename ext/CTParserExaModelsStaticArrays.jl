"""
    CTParserExaModelsStaticArrays

Static numeric constants (`SVector`, `SMatrix`, …) with arrays of ExaModels nodes, in the
code `@def` emits for the `:exa` backend: `x(t) + q`, `q - x(t)`, `A * x(t)` with `q` and
`A` static.

`CTParserExaModels` defines these operations for any numeric `AbstractArray`, and
StaticArrays defines them for its static arrays with any other array: for a static array
with an array of nodes, both apply and neither is more specific, so the call is ambiguous
(control-toolbox/CTParser.jl#365). The methods below cover exactly those intersections. The
result is an ordinary `Vector` / `Matrix` of nodes, as with any other constant: nodes are
not stored in static arrays.

Type piracy, accepted as for `CTParserExaModels`: CTParser owns neither `Base.:+`, `-`, `*`,
the static arrays nor the node types.
"""
module CTParserExaModelsStaticArrays

using CTBase: CTBase
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using StaticArrays: StaticArrays

# Same aliases as in `CTParserExaModels` (types.jl); a test checks they are equal.
const ExaNode = Union{
    ExaModels.AbstractNode,ExaModels.AbstractAdjointNode,ExaModels.AbstractSecondAdjointNode
}
const VecExaNode = StridedVector{T} where {T<:ExaNode}
const MatExaNode = StridedMatrix{T} where {T<:ExaNode}

# The static arrays exactly as StaticArrays' own `+`, `-` and `*` methods take them, with
# numeric entries: written like this, and not as `SVector` / `SMatrix`, so that each method
# below is the intersection of the two colliding signatures and nothing wider.
const RealStaticVector = StaticArrays.StaticArray{S,<:Real,1} where {S<:Tuple}
const RealStaticMatrix = StaticArrays.StaticArray{S,<:Real,2} where {S<:Tuple}
const RealStaticMatrixNM = StaticArrays.StaticArray{Tuple{N,M},<:Real,2} where {N,M}

function _dimension_mismatch(op, reason)
    return throw(
        CTBase.PreconditionError(
            "operand dimensions do not match for $op";
            reason=reason,
            suggestion="pass operands with compatible dimensions",
            context="linear algebra on ExaModels nodes",
        ),
    )
end

function _entrywise(op, a, b)
    size(a) == size(b) || _dimension_mismatch(string(op), "sizes $(size(a)) and $(size(b))")
    return [op(a[I], b[I]) for I in CartesianIndices(size(a))]
end

for op in (:+, :-),
    (TA, TB) in [
        (RealStaticVector, VecExaNode),
        (VecExaNode, RealStaticVector),
        (RealStaticMatrix, MatExaNode),
        (MatExaNode, RealStaticMatrix),
    ]

    @eval Base.$op(a::$TA, b::$TB) = _entrywise($op, a, b)
end

function _row_dot(A, i, x)
    s = A[i, 1] * x[1]
    for j in 2:size(A, 2)
        s = s + A[i, j] * x[j]
    end
    return s
end

function Base.:*(A::RealStaticMatrixNM, x::VecExaNode)
    size(A, 2) == length(x) || _dimension_mismatch(
        "*", "matrix of size $(size(A)) and vector of length $(length(x))"
    )
    # A comprehension over a plain range: `map` over the axes of a static matrix would
    # give a static vector of nodes.
    return [_row_dot(A, i, x) for i in 1:size(A, 1)]
end

end # module CTParserExaModelsStaticArrays
