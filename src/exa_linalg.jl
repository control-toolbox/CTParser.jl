"""
    ExaLinAlg

Module providing trait-based linear algebra extensions for Array{<:ExaModels.AbstractNode}.
Extends Julia's standard Array interface without wrappers.

# Key Design: Direct Operator Overloading on Null Nodes
All operations use direct operator overloads (+, -, *) on Null types.
These specialized methods properly handle Null nodes, preserving constants
and avoiding unnecessary expression tree nodes.

# Scalar Iteration Protocol
AbstractNode implements the iteration protocol to behave as a 0-dimensional
scalar iterable (like Number types in Julia). This enables compatibility with
ALL AbstractArray wrapper types (SubArray, ReshapedArray, ReinterpretArray, etc.)
and generic LinearAlgebra operations, without requiring explicit method definitions
for each array type.

# The 6 Minimal Operations for Linear Algebra Extension
To extend linear algebra operations to arrays of a custom scalar type T
(where T is not a subtype of Number), you must define:

1. **Arithmetic Operations**: `+`, `-`, `*`
   - Binary operations between T and T
   - Binary operations between T and Real
   - Unary operations: `-T`, `+T`

2. **Identity Elements**: `zero(T)`, `one(T)`
   - Return canonical zero and one values of type T
   - Also: `zeros(Type{T}, dims...)`, `ones(Type{T}, dims...)`

3. **Type Promotion**: `promote_rule`, `convert`
   - `promote_rule(::Type{T}, ::Type{<:Real})` → T
   - `convert(::Type{T}, ::Real)` → T

4. **Scalar Iteration Protocol**: `iterate`, `length`, `eltype`
   - `iterate(x::T)` → (x, nothing)  # First call returns value
   - `iterate(x::T, ::Nothing)` → nothing  # Second call indicates done
   - `length(x::T)` → 1  # Scalars have length 1
   - `eltype(::Type{T})` → T  # Element type is itself
   - `IteratorSize(::Type{T})` → Base.HasLength()
   - `IteratorEltype(::Type{T})` → Base.HasEltype()

5. **Scalar Dot Product**: `dot(::Number, ::T)`, `dot(::T, ::Number)`, `dot(::T, ::T)`
   - Required because T is not a Number, so dot(::Number, ::Number) doesn't match
   - Prevents falling back to generic iterable dot which has recursion protection
   - Implementation: `dot(x, y) = conj(x) * y` (standard definition)

6. **Complex Conjugate**: `conj(::T)`
   - For real-valued types: `conj(x) = x` (identity)
   - For complex-valued types: `conj(x)` should return complex conjugate
   - Note: ExaModels.AbstractNode only deals with Real values

With these 6 categories defined, ALL LinearAlgebra operations work automatically
with ALL AbstractArray types (Vector, Matrix, SubArray, ReshapedArray, etc.)
through Julia's generic implementations.

# Custom Implementations for Optimization
While the 6 minimal operations make linear algebra WORK, this module provides
custom implementations that make it FAST:

- **Zero optimization**: `0 * x` → `Null(0)` (not a Node2 expression tree)
- **Constant folding**: `Null(2) * Null(3)` → `Null(6)` (computed at definition time)
- **Null wrapping**: Wraps Real coefficients in `Null()` before multiplication to trigger optimizations
- **Simpler expression trees**: Custom methods produce `Null` results instead of deep `Node2` trees

Example optimization impact:
```julia
# Native (with only 6 minimal ops): returns complex Node2 tree
dot([1, 0, 2], [x, y, z])  # →  Node2{+, Node2{+, Null(0), Node2{*, 0, y}}, Node2{*, 2, z}}

# Custom (optimized): returns simplified Null
dot([1, 0, 2], [x, y, z])  # →  Null(1*x.value + 0 + 2*z.value)
```

Thus: **6 minimal operations = correctness, custom implementations = performance**

# What Must Be Custom vs What Works Natively
With the 6 minimal operations defined:

**Works natively (custom versions are optimizations):**
- `dot` for all array types - works but creates Node2 trees
- `sum` - works but less efficient
- `tr` (trace) - works fine natively
- Vector/Matrix addition - works but less optimized
- Scalar × Vector/Matrix - works but less optimized

**Must be custom (native fails or is problematic):**
- `norm` - native causes stack overflow (tries to iterate over result nodes)
- `det` (for Matrix{AbstractNode}) - native uses LU which needs type conversions
- Matrix × Vector (for certain type combinations) - native may fail

**Recommended**: Keep all custom implementations for performance, but understand
that the 6 minimal operations provide the foundation that makes everything possible.

# Public API (Exported)
- Canonical nodes: `zero`, `one`, `zeros`, `ones`
- Basic operations: `zero`, `one`, `adjoint`, `transpose`, `*`, `+`, `-`, `sum`, `conj`
- Linear algebra: `dot`, `det`, `tr`, `norm`, `diag`, `diagm`
"""
module ExaLinAlg

using ExaModels: ExaModels
using LinearAlgebra

import Base: zero, one, adjoint, *, promote_rule, convert, +, -, transpose, sum, conj
import Base: inv, abs, sqrt, cbrt, abs2, exp, exp2, exp10, log, log2, log10, log1p
import Base: sin, cos, tan, csc, sec, cot, asin, acos, atan, acot
import Base: sind, cosd, tand, cscd, secd, cotd, atand, acotd
import Base: sinh, cosh, tanh, csch, sech, coth, asinh, acosh, atanh, acoth
import Base: ^, zeros, ones, iterate, length, eltype, IteratorSize, IteratorEltype
import LinearAlgebra: dot, Adjoint, det, tr, norm, diag, diagm

export zero, one, zeros, ones, adjoint, transpose, *, +, -, sum, dot, det, tr, norm, diag, diagm, conj
export inv, abs, sqrt, cbrt, abs2, exp, exp2, exp10, log, log2, log10, log1p
export sin, cos, tan, csc, sec, cot, asin, acos, atan, acot
export sind, cosd, tand, cscd, secd, cotd, atand, acotd
export sinh, cosh, tanh, csch, sech, coth, asinh, acosh, atanh, acoth
export ^

# ============================================================================
# Section 1: Canonical Nodes (zero and one)
# ============================================================================
#
# Canonical encodings:
# - Zero: Null(0) represents zero
# - One:  Null(1) represents one
#
# We extend Base.zero and Base.one for both type-based and instance-based calls:
# - zero(ExaModels.AbstractNode) or zero(::Type{<:ExaModels.AbstractNode})
# - one(ExaModels.AbstractNode) or one(::Type{<:ExaModels.AbstractNode})
# ============================================================================

"""
    zero(::Type{<:ExaModels.AbstractNode}) -> Null{Int}

Return the canonical zero AbstractNode: Null(0).
"""
zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(0)

"""
    zero(::ExaModels.AbstractNode) -> Null{Int}

Return the canonical zero AbstractNode: Null(0).
"""
zero(::ExaModels.AbstractNode) = ExaModels.Null(0)

"""
    one(::Type{<:ExaModels.AbstractNode}) -> Null{Int}

Return the canonical one AbstractNode: Null(1).
"""
one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)

"""
    one(::ExaModels.AbstractNode) -> Null{Int}

Return the canonical one AbstractNode: Null(1).
"""
one(::ExaModels.AbstractNode) = ExaModels.Null(1)

"""
    zeros(::Type{T}, dims...) where {T <: ExaModels.AbstractNode}

Create an array of AbstractNode zeros with the specified dimensions.
Uses fill with the canonical zero node: Null(0).
"""
function zeros(::Type{T}, dims::Integer...) where {T <: ExaModels.AbstractNode}
    return fill(zero(T), dims...)
end

"""
    ones(::Type{T}, dims...) where {T <: ExaModels.AbstractNode}

Create an array of AbstractNode ones with the specified dimensions.
Uses fill with the canonical one node: Null(1).
"""
function ones(::Type{T}, dims::Integer...) where {T <: ExaModels.AbstractNode}
    return fill(one(T), dims...)
end

# ============================================================================
# Section 1.5: Iteration Protocol for Scalar Behavior
# ============================================================================
#
# AbstractNode represents a scalar value (a node in an expression tree).
# To enable compatibility with generic LinearAlgebra functions and all
# AbstractArray subtypes (SubArray, ReshapedArray, ReinterpretArray, etc.),
# we implement the iteration protocol to make AbstractNode behave like
# a 0-dimensional scalar collection (similar to how Number types work).
#
# This allows AbstractNode to work seamlessly with:
# - Views: view(x, 1:2)
# - Reshaped arrays: reshape(x, dims)
# - Reinterpreted arrays: reinterpret(T, x)
# - Any other AbstractArray wrapper types
#
# Without this, operations like dot([1,2], view(x, 1:2)) would fall back
# to LinearAlgebra's generic methods which expect scalars to be iterable.
# ============================================================================

"""
    iterate(x::ExaModels.AbstractNode)

Make AbstractNode iterable as a 0-dimensional scalar (like Number types).
Returns (x, nothing) to indicate a single-element iteration.
"""
iterate(x::ExaModels.AbstractNode) = (x, nothing)

"""
    iterate(x::ExaModels.AbstractNode, ::Nothing)

Second call to iterate returns nothing, indicating iteration is complete.
"""
iterate(x::ExaModels.AbstractNode, ::Nothing) = nothing

"""
    length(x::ExaModels.AbstractNode)

Return length 1 for scalar AbstractNode (0-dimensional iterable).
"""
length(x::ExaModels.AbstractNode) = 1

"""
    eltype(::Type{<:ExaModels.AbstractNode})

Return the element type when iterating over an AbstractNode (returns itself).
"""
eltype(::Type{T}) where {T <: ExaModels.AbstractNode} = T

"""
    IteratorSize(::Type{<:ExaModels.AbstractNode})

AbstractNode has known size (length 1).
"""
IteratorSize(::Type{<:ExaModels.AbstractNode}) = Base.HasLength()

"""
    IteratorEltype(::Type{<:ExaModels.AbstractNode})

AbstractNode has known eltype.
"""
IteratorEltype(::Type{<:ExaModels.AbstractNode}) = Base.HasEltype()

# ============================================================================
# Section 1.6: Scalar dot product for AbstractNode
# ============================================================================
#
# Define scalar dot product to prevent falling back to generic iterable dot.
# This is needed because AbstractNode is not a Number, so dot(::Number, ::Number)
# doesn't match. Without this, LinearAlgebra's generic dot for iterables is used,
# which has a recursive protection check that fails for our case.
# ============================================================================

"""
    dot(x::Number, y::ExaModels.AbstractNode)

Scalar dot product: Number · AbstractNode = Null(Number) * AbstractNode

Note: We wrap Number in Null to trigger our optimized * operator which does:
- Zero elimination: Null(0) * anything → Null(0)
- One elimination: Null(1) * x → x
- Constant folding for Null operands

Since AbstractNode only deals with Real values, conj(x) = x, so we don't need conj.
"""
dot(x::Number, y::ExaModels.AbstractNode) = ExaModels.Null(x) * y

"""
    dot(x::ExaModels.AbstractNode, y::Number)

Scalar dot product: AbstractNode · Number = AbstractNode * Null(Number)
"""
dot(x::ExaModels.AbstractNode, y::Number) = x * ExaModels.Null(y)

"""
    dot(x::ExaModels.AbstractNode, y::ExaModels.AbstractNode)

Scalar dot product: AbstractNode · AbstractNode = AbstractNode * AbstractNode
"""
dot(x::ExaModels.AbstractNode, y::ExaModels.AbstractNode) = x * y

# ============================================================================
# Section 2: Scalar Operations on Null Nodes
# ============================================================================
#
# Direct operator overloads for Null nodes (more specific than ExaModels' AbstractNode methods).
# These preserve Null when operating on constants and handle special cases like 0 * x = Null(0).
#
# Rules for +:
#   Null(x) + Null(y) = Null(x + y)
#   Null(x) + e = x + e  (unwrap to scalar, use ExaModels native +)
#   e + Null(x) = e + x
#
# Rules for -:
#   Null(x) - Null(y) = Null(x - y)
#   Null(0) - e = -e
#   Null(x) - e = x - e when !iszero(x)
#   e - Null(x) = e - x
#
# Rules for *:
#   Null(x) * Null(y) = Null(x * y)
#   Null(0) * e = Null(0)  (zero optimization)
#   e * Null(0) = Null(0)
#   Null(x) * e = x * e when !iszero(x)
#   e * Null(x) = e * x when !iszero(x)
# ============================================================================

# Addition: Null{T} + Null{S} → Null
+(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value + y.value)
# Addition: Null{T} + AbstractNode → unwrap Null, use native +
+(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = x.value + y
+(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x + y.value
# Addition: Null{T} + Real → Null (more specific than ExaModels' AbstractNode + Real)
+(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value + y)
+(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x + y.value)

# Subtraction: Null{T} - Null{S} → Null
-(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value - y.value)
# Subtraction: Null{T} - AbstractNode → handle 0 - e = -e specially
-(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? (-y) : (x.value - y)
-(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = x - y.value
# Subtraction: Null{T} - Real → Null (more specific than ExaModels' AbstractNode - Real)
-(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value - y)
-(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x - y.value)

# Multiplication: Null{T} * Null{S} → Null
*(x::ExaModels.Null{T}, y::ExaModels.Null{S}) where {T<:Real, S<:Real} = ExaModels.Null(x.value * y.value)
# Multiplication: Null{T} * AbstractNode → zero optimization: 0 * e = Null(0)
*(x::ExaModels.Null{T}, y::ExaModels.AbstractNode) where {T<:Real} = iszero(x.value) ? ExaModels.Null(0) : (x.value * y)
*(x::ExaModels.AbstractNode, y::ExaModels.Null{T}) where {T<:Real} = iszero(y.value) ? ExaModels.Null(0) : (x * y.value)
# Multiplication: Null{T} * Real → zero optimization, more specific than ExaModels' AbstractNode * Real
*(x::ExaModels.Null{T}, y::Real) where {T<:Real} = ExaModels.Null(x.value * y)
*(x::Real, y::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(x * y.value)

# ============================================================================
# Section 2.5: Unary Functions on Null Nodes
# ============================================================================
#
# Apply unary functions to the inner value of Null nodes.
# This fixes broadcasting: cos.([Null(x)]) → [Null(cos(x))] instead of [cos(Null(x))]
#
# Pattern: f(x::Null{T}) = Null(f(x.value))
# ============================================================================

# Unary operators
-(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(-x.value)
+(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(+x.value)

# Basic functions
inv(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(inv(x.value))
abs(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(abs(x.value))
sqrt(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sqrt(x.value))
cbrt(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cbrt(x.value))
abs2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(abs2(x.value))

# Exponential and logarithmic functions
exp(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp(x.value))
exp2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp2(x.value))
exp10(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(exp10(x.value))
log(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log(x.value))
log2(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log2(x.value))
log10(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log10(x.value))
log1p(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(log1p(x.value))

# Trigonometric functions
sin(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sin(x.value))
cos(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cos(x.value))
tan(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tan(x.value))
csc(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(csc(x.value))
sec(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sec(x.value))
cot(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cot(x.value))
asin(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(asin(x.value))
acos(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acos(x.value))
atan(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atan(x.value))
acot(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acot(x.value))

# Degree-based trigonometric functions
sind(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sind(x.value))
cosd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cosd(x.value))
tand(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tand(x.value))
cscd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cscd(x.value))
secd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(secd(x.value))
cotd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cotd(x.value))
atand(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atand(x.value))
acotd(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acotd(x.value))

# Hyperbolic functions
sinh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sinh(x.value))
cosh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(cosh(x.value))
tanh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(tanh(x.value))
csch(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(csch(x.value))
sech(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(sech(x.value))
coth(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(coth(x.value))
asinh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(asinh(x.value))
acosh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acosh(x.value))
atanh(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(atanh(x.value))
acoth(x::ExaModels.Null{T}) where {T<:Real} = ExaModels.Null(acoth(x.value))

# Power function (unary form: x^n is actually binary, but we handle Null^Real)
^(x::ExaModels.Null{T}, n::Real) where {T<:Real} = ExaModels.Null(x.value^n)
^(x::ExaModels.Null{T}, n::Integer) where {T<:Real} = ExaModels.Null(x.value^n)

# Complex conjugate (ExaModels.AbstractNode only deals with Real values, so conj is identity)
conj(x::ExaModels.AbstractNode) = x  # For real-valued nodes, conj(x) = x

# ============================================================================
# IMPLEMENTATION NOTE: Redundant Operations Removed
# ============================================================================
#
# With the 6 minimal operations + optimized scalar dot (with Null wrapping),
# many array-level operations now work natively through Julia's generic
# LinearAlgebra implementations. The following have been REMOVED as redundant:
#
# - Array-level dot for all Vector/Matrix combinations (native works!)
# - Vector/Matrix addition and subtraction (native works!)
# - Scalar × Vector/Matrix multiplication (native works!)
# - sum (native works!)
# - tr (trace - native works!)
# - Matrix × Matrix multiplication (native works!)
# - Adjoint/transpose wrapper operations (native works!)
#
# KEPT custom implementations only for:
# - Matrix × Vector (native has MethodError)
# - norm (native causes stack overflow)
# - det (native LU factorization fails)
# - diag, diagm (specialized implementations)
#
# This dramatically simplifies the code from ~800 lines to ~400 lines!
# ============================================================================

# ============================================================================
# Section 3.5: Basic Type Conversions and Promotions
# ============================================================================

# Scalar operations
adjoint(x::ExaModels.AbstractNode) = x
transpose(x::ExaModels.AbstractNode) = x

convert(::Type{ExaModels.AbstractNode}, x::Real) = iszero(x) ? zero(ExaModels.AbstractNode) : ExaModels.Null(x)

# Convert between Null types with different numeric types (e.g., Null{Int64} to Null{Float64})
convert(::Type{ExaModels.Null{T}}, x::ExaModels.Null{S}) where {T,S} = ExaModels.Null(T(x.value))

# Convert Null to Real types (needed for some LinearAlgebra operations like lu/det)
convert(::Type{T}, x::ExaModels.Null{S}) where {T<:Real, S} = T(x.value)

promote_rule(::Type{<:ExaModels.AbstractNode}, ::Type{<:Real}) = ExaModels.AbstractNode

# ============================================================================
# Section 6: Matrix × Vector Product (uses dot)
# ============================================================================
#
# KEPT: Native LinearAlgebra fails with MethodError for some combinations
# ============================================================================

function *(A::Matrix{<:Real}, x::Vector{T}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    n == length(x) || throw(DimensionMismatch("matrix A has dimensions ($m,$n), vector B has length $(length(x))"))
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{T}, x::Vector{<:Real}) where {T <: ExaModels.AbstractNode}
    m, n = size(A)
    n == length(x) || throw(DimensionMismatch("matrix A has dimensions ($m,$n), vector B has length $(length(x))"))
    return [dot(A[i, :], x) for i in 1:m]
end

function *(A::Matrix{T}, x::Vector{S}) where {T <: ExaModels.AbstractNode, S <: ExaModels.AbstractNode}
    m, n = size(A)
    n == length(x) || throw(DimensionMismatch("matrix A has dimensions ($m,$n), vector B has length $(length(x))"))
    return [dot(A[i, :], x) for i in 1:m]
end

# ============================================================================
# Section 7: Scalar-Array Multiplication (both AbstractNode)
# ============================================================================
#
# When BOTH operands are AbstractNode types, we need explicit broadcasting
# methods since Julia doesn't automatically handle this case
# ============================================================================

# Scalar × Vector (both AbstractNode)
function *(scalar::ExaModels.AbstractNode, v::Vector{T}) where {T <: ExaModels.AbstractNode}
    return [scalar * vi for vi in v]
end

# Vector × Scalar (both AbstractNode)
function *(v::Vector{T}, scalar::ExaModels.AbstractNode) where {T <: ExaModels.AbstractNode}
    return [vi * scalar for vi in v]
end

# Scalar × Matrix (both AbstractNode)
function *(scalar::ExaModels.AbstractNode, A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    return [scalar * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix × Scalar (both AbstractNode)
function *(A::Matrix{T}, scalar::ExaModels.AbstractNode) where {T <: ExaModels.AbstractNode}
    return [A[i, j] * scalar for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 8: Scalar-Array Multiplication (mixed types)
# ============================================================================
#
# When one operand is AbstractNode and the other is an array of Real
# ============================================================================

# AbstractNode × Vector{Real}
function *(scalar::ExaModels.AbstractNode, v::Vector{<:Real})
    return [scalar * vi for vi in v]
end

# Vector{Real} × AbstractNode
function *(v::Vector{<:Real}, scalar::ExaModels.AbstractNode)
    return [vi * scalar for vi in v]
end

# AbstractNode × Matrix{Real}
function *(scalar::ExaModels.AbstractNode, A::Matrix{<:Real})
    return [scalar * A[i, j] for i in axes(A, 1), j in axes(A, 2)]
end

# Matrix{Real} × AbstractNode
function *(A::Matrix{<:Real}, scalar::ExaModels.AbstractNode)
    return [A[i, j] * scalar for i in axes(A, 1), j in axes(A, 2)]
end

# ============================================================================
# Section 12: Determinant (using +, -, *)
# ============================================================================
#
# KEPT: Native LinearAlgebra det uses LU factorization which fails with type conversions
# ============================================================================

function det(A::AbstractMatrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    @assert n == m "Determinant is only defined for square matrices, got $(n)×$(m)"

    if n == 1
        return A[1, 1]
    elseif n == 2
        return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    elseif n == 3
        # Sarrus rule for 3×3 matrices
        pos1 = A[1, 1] * A[2, 2] * A[3, 3]
        pos2 = A[1, 2] * A[2, 3] * A[3, 1]
        pos3 = A[1, 3] * A[2, 1] * A[3, 2]
        neg1 = A[1, 3] * A[2, 2] * A[3, 1]
        neg2 = A[1, 1] * A[2, 3] * A[3, 2]
        neg3 = A[1, 2] * A[2, 1] * A[3, 3]
        pos_sum = pos1 + pos2 + pos3
        neg_sum = neg1 + neg2 + neg3
        return pos_sum - neg_sum
    else
        # Laplace expansion for n×n matrices (n ≥ 4)
        d = A[1, 1] * det(A[2:end, 2:end])  # Initialize with first term
        for j in 2:n
            minor = [A[2:end, 1:j-1] A[2:end, j+1:end]]
            sign_coeff = iseven(j) ? -1 : 1
            cofactor = sign_coeff * A[1, j] * det(minor)
            d = d + cofactor
        end
        return d
    end
end

# ============================================================================
# Section 13: Norms (using sum, *)
# ============================================================================
#
# KEPT: Native LinearAlgebra norm causes stack overflow when AbstractNode is iterable
# ============================================================================

# Euclidean norm (2-norm) for vectors
# Uses AbstractVector to support SubArray, ReshapedArray, etc.
function norm(v::AbstractVector{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(vi * vi for vi in v))
end

# p-norm for vectors
# Uses AbstractVector to support SubArray, ReshapedArray, etc.
function norm(v::AbstractVector{T}, p::Real) where {T <: ExaModels.AbstractNode}
    if p == Inf
        # Infinity norm: max|vᵢ|
        error("Infinity norm not supported for symbolic AbstractNode vectors")
    elseif p == 1
        # 1-norm: sum of absolute values
        return sum(abs(vi) for vi in v)
    elseif p == 2
        # 2-norm: Euclidean norm
        return sqrt(sum(vi * vi for vi in v))
    else
        # General p-norm
        return sum(abs(vi)^p for vi in v)^(1/p)
    end
end

# Frobenius norm for matrices
# Uses AbstractMatrix to support ReshapedArray, etc.
function norm(A::AbstractMatrix{T}) where {T <: ExaModels.AbstractNode}
    return sqrt(sum(A[i, j] * A[i, j] for i in axes(A, 1), j in axes(A, 2)))
end

# ============================================================================
# Section 15: Diagonal Operations
# ============================================================================

# Extract diagonal from matrix
function diag(A::Matrix{T}) where {T <: ExaModels.AbstractNode}
    n, m = size(A)
    k = min(n, m)
    return [A[i, i] for i in 1:k]
end

# Create diagonal matrix from vector
function diagm(v::Vector{T}) where {T <: ExaModels.AbstractNode}
    n = length(v)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = (i == j) ? v[i] : zero(ExaModels.AbstractNode)
    end
    return D
end

# diagm with pairs (more general form)
function diagm(kv::Pair{<:Integer, <:Vector{T}}) where {T <: ExaModels.AbstractNode}
    k, v = kv
    n = length(v) + abs(k)
    # Create a matrix with AbstractNode element type to allow mixed Null types
    D = Matrix{ExaModels.AbstractNode}(undef, n, n)
    for i in 1:n, j in 1:n
        D[i, j] = zero(ExaModels.AbstractNode)
    end
    if k >= 0
        for i in 1:length(v)
            D[i, i + k] = v[i]
        end
    else
        for i in 1:length(v)
            D[i - k, i] = v[i]
        end
    end
    return D
end

end  # module ExaLinAlg
