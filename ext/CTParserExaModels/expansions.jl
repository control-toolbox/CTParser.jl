# det, norm and cross, expanded to a single scalar expression graph.
# Part of the extension CTParserExaModels, included by CTParserExaModels.jl.

# ============================================================================
# Optimized scalar expansions
#
# det and norm expand to a single scalar expression graph. The small-size det
# cases are written out rather than recursed so the resulting graph stays flat.
# ============================================================================

# --- det (specialized for small sizes) ---

_det_1x1(A) = A[1, 1]

# 2x2: a11*a22 - a12*a21
function _det_2x2(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a21 = A[2, 1]
    a22 = A[2, 2]
    return a11 * a22 - a12 * a21
end

# 3x3: Sarrus' rule (optimized expansion)
function _det_3x3(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a21 = A[2, 1]
    a22 = A[2, 2]
    a23 = A[2, 3]
    a31 = A[3, 1]
    a32 = A[3, 2]
    a33 = A[3, 3]
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31)
end

# 4x4: cofactor expansion along first row
function _det_4x4(A)
    a11 = A[1, 1]
    a12 = A[1, 2]
    a13 = A[1, 3]
    a14 = A[1, 4]
    m11 =
        A[2, 2] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) +
        A[2, 4] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2])
    m12 =
        A[2, 1] * (A[3, 3] * A[4, 4] - A[3, 4] * A[4, 3]) -
        A[2, 3] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1])
    m13 =
        A[2, 1] * (A[3, 2] * A[4, 4] - A[3, 4] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 4] - A[3, 4] * A[4, 1]) +
        A[2, 4] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    m14 =
        A[2, 1] * (A[3, 2] * A[4, 3] - A[3, 3] * A[4, 2]) -
        A[2, 2] * (A[3, 1] * A[4, 3] - A[3, 3] * A[4, 1]) +
        A[2, 3] * (A[3, 1] * A[4, 2] - A[3, 2] * A[4, 1])
    return a11 * m11 - a12 * m12 + a13 * m13 - a14 * m14
end

# General determinant via cofactor expansion (recursive, for N > 4)
function _det_recursive(A)
    n = size(A, 1) # square: checked once by det, and every minor of a square matrix is square
    if n == 1
        return _det_1x1(A)
    elseif n == 2
        return _det_2x2(A)
    elseif n == 3
        return _det_3x3(A)
    elseif n == 4
        return _det_4x4(A)
    end
    s = A[1, 1] * _det_recursive(A[2:end, 2:end])
    for j in 2:n
        cols = vcat(1:(j - 1), (j + 1):n)
        minor = _det_recursive(A[2:end, cols])
        if iseven(j)
            s = s - A[1, j] * minor
        else
            s = s + A[1, j] * minor
        end
    end
    return s
end

# Dispatch det for matrices containing ExaNode elements
function LinearAlgebra.det(A::MatExaNode)
    size(A, 1) == size(A, 2) ||
        _bad_operand("det", "matrix of size $(size(A))", "a square matrix")
    return _det_recursive(A)
end

# --- norm ---

# 2-norm for vectors of nodes: sqrt(sum(xi^2))
function LinearAlgebra.norm(v::CPUVecExaNode)
    s = v[1]^2
    for i in 2:length(v)
        s = s + v[i]^2
    end
    return sqrt(s)
end

# p-norm for vectors of nodes: (sum(abs(xi)^p))^(1/p)
function LinearAlgebra.norm(v::CPUVecExaNode, p::Real)
    if p == 2
        return LinearAlgebra.norm(v)
    elseif p == 1
        s = abs(v[1])
        for i in 2:length(v)
            s = s + abs(v[i])
        end
        return s
    elseif p == Inf
        # the Inf-norm is not differentiable, so it has no node expression
        _bad_operand("norm", "p = Inf", "a finite p")
    else
        s = abs(v[1])^p
        for i in 2:length(v)
            s = s + abs(v[i])^p
        end
        return s^(1 / p)
    end
end

# Frobenius norm for matrices of nodes: sqrt(sum(aij^2))
function LinearAlgebra.norm(A::CPUMatExaNode)
    s = A[1, 1]^2
    for j in axes(A, 2), i in axes(A, 1)
        (i == 1 && j == 1) && continue
        s = s + A[i, j]^2
    end
    return sqrt(s)
end

# --- cross product (3D only) ---

for (T1, T2) in _VEC_PAIRS
    @eval function LinearAlgebra.cross(a::$T1, b::$T2)
        (length(a) == 3 && length(b) == 3) || _bad_operand(
            "cross", "vectors of lengths $(length(a)) and $(length(b))", "two 3-vectors"
        )
        return [
            a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]
        ]
    end
end
