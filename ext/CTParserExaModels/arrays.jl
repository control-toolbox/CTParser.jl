# Linear algebra on vectors and matrices of nodes, rebuilt from scalar operations.
# Part of the extension CTParserExaModels, included by CTParserExaModels.jl.

# ============================================================================
# Plain Julia scalar decompositions
#
# Every operation below rebuilds the result element by element from scalar node
# arithmetic. Nodes are expression graph vertices, not numbers, so the stdlib's
# BLAS-backed paths do not apply.
# ============================================================================

# --- sum ---

function Base.sum(v::VecExaNode)
    s = v[1]
    for i in 2:length(v)
        s = s + v[i]
    end
    return s
end

# --- dot ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.dot(a::$T1, b::$T2)
        length(a) == length(b) ||
            _dimension_mismatch("dot", "lengths $(length(a)) and $(length(b))")
        s = a[1] * b[1]
        for i in 2:length(a)
            s = s + a[i] * b[i]
        end
        return s
    end
end

# --- scalar * vector ---

for (T1, T2) in
    [(Real, VecExaNode), (ExaNode, AbstractVector{<:Real}), (ExaNode, VecExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i] for i in eachindex(b)]
end
for (T1, T2) in
    [(VecExaNode, Real), (AbstractVector{<:Real}, ExaNode), (VecExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i] * b for i in eachindex(a)]
end

# --- scalar * matrix ---

for (T1, T2) in
    [(Real, MatExaNode), (ExaNode, AbstractMatrix{<:Real}), (ExaNode, MatExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a * b[i, j] for i in axes(b, 1), j in axes(b, 2)]
end
for (T1, T2) in
    [(MatExaNode, Real), (AbstractMatrix{<:Real}, ExaNode), (MatExaNode, ExaNode)]
    @eval Base.:*(a::$T1, b::$T2) = [a[i, j] * b for i in axes(a, 1), j in axes(a, 2)]
end

# --- matrix * vector (inline dot to avoid dispatch issues with view types) ---

function _dot_row(A, i, x)
    n = size(A, 2)
    s = A[i, 1] * x[1]
    for j in 2:n
        s = s + A[i, j] * x[j]
    end
    return s
end

for (T1, T2) in [
    (MatExaNode, VecExaNode),
    (AbstractMatrix{<:Real}, VecExaNode),
    (MatExaNode, AbstractVector{<:Real}),
]
    @eval function Base.:*(A::$T1, x::$T2)
        m = size(A, 1)
        size(A, 2) == length(x) || _dimension_mismatch(
            "*", "matrix of size $(size(A)) and vector of length $(length(x))"
        )
        return [_dot_row(A, i, x) for i in 1:m]
    end
end

# --- matrix * matrix (inline dot to avoid dispatch issues) ---

function _dot_col(A, i, B, j)
    n = size(A, 2)
    s = A[i, 1] * B[1, j]
    for k in 2:n
        s = s + A[i, k] * B[k, j]
    end
    return s
end

for (T1, T2) in _MAT_PAIRS
    @eval function Base.:*(A::$T1, B::$T2)
        size(A, 2) == size(B, 1) ||
            _dimension_mismatch("*", "matrices of sizes $(size(A)) and $(size(B))")
        m, n = size(A, 1), size(B, 2)
        return [_dot_col(A, i, B, j) for i in 1:m, j in 1:n]
    end
end

# --- vector +/- ---

for op in (:+, :-)
    for (T1, T2) in _VEC_PAIRS
        @eval function Base.$op(a::$T1, b::$T2)
            length(a) == length(b) || _dimension_mismatch(
                $(string(op)), "lengths $(length(a)) and $(length(b))"
            )
            return [$op(a[i], b[i]) for i in eachindex(a)]
        end
    end
end

# Win dispatch over Base's +(::Array, ::Array...) from arraymath.jl
for (T1, T2) in [(ExaNode, ExaNode), (Real, ExaNode), (ExaNode, Real)]
    @eval function Base.:+(a::Array{<:$T1,1}, b::Array{<:$T2,1})
        length(a) == length(b) ||
            _dimension_mismatch("+", "lengths $(length(a)) and $(length(b))")
        return [a[i] + b[i] for i in eachindex(a)]
    end
    @eval function Base.:+(A::Array{<:$T1,2}, B::Array{<:$T2,2})
        size(A) == size(B) ||
            _dimension_mismatch("+", "sizes $(size(A)) and $(size(B))")
        return [A[i, j] + B[i, j] for i in axes(A, 1), j in axes(A, 2)]
    end
end

# Unary minus for vector/matrix of nodes
Base.:-(a::VecExaNode) = [-a[i] for i in eachindex(a)]
Base.:-(A::MatExaNode) = [-A[i, j] for i in axes(A, 1), j in axes(A, 2)]

# --- matrix +/- ---

for op in (:+, :-)
    for (T1, T2) in _MAT_PAIRS
        @eval function Base.$op(A::$T1, B::$T2)
            size(A) == size(B) || _dimension_mismatch(
                $(string(op)), "sizes $(size(A)) and $(size(B))"
            )
            return [$op(A[i, j], B[i, j]) for i in axes(A, 1), j in axes(A, 2)]
        end
    end
end

# --- diag ---

function LinearAlgebra.diag(A::MatExaNode)
    n = minimum(size(A))
    return [A[i, i] for i in 1:n]
end

# --- diagm ---

function LinearAlgebra.diagm(v::VecExaNode)
    n = length(v)
    T = typeof(v[1])
    M = Matrix{Union{T,ExaModels.Null{Int}}}(undef, n, n)
    for i in 1:n, j in 1:n
        if i == j
            M[i, j] = v[i]
        else
            M[i, j] = ExaModels.Null(0)
        end
    end
    return M
end

# --- adjoint/transpose operations ---

# v' * w = dot(v, w)
for (TA, TV, TB) in [
    (ExaNode, VecExaNode, VecExaNode),
    (ExaNode, VecExaNode, AbstractVector{<:Real}),
    (Real, AbstractVector{<:Real}, VecExaNode),
]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA,<:$TV}, b::$TB)
        return LinearAlgebra.dot(parent(a), b)
    end
end

# v' * A
for (TA, TV, TB) in
    [(ExaNode, VecExaNode, MatExaNode), (Real, AbstractVector{<:Real}, MatExaNode)]
    @eval function Base.:*(a::LinearAlgebra.Adjoint{<:$TA,<:$TV}, B::$TB)
        v = parent(a)
        length(v) == size(B, 1) || _dimension_mismatch(
            "*", "adjoint vector of length $(length(v)) and matrix of size $(size(B))"
        )
        n = size(B, 2)
        return adjoint([LinearAlgebra.dot(v, [B[k, j] for k in 1:size(B, 1)]) for j in 1:n])
    end
end
function Base.:*(
    a::LinearAlgebra.Adjoint{<:ExaNode,<:VecExaNode}, B::AbstractMatrix{<:Real}
)
    v = parent(a)
    length(v) == size(B, 1) || _dimension_mismatch(
        "*", "adjoint vector of length $(length(v)) and matrix of size $(size(B))"
    )
    n = size(B, 2)
    return adjoint([LinearAlgebra.dot(v, view(B, :, j)) for j in 1:n])
end
