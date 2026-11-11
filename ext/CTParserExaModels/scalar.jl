# Scalar arithmetic on ExaModels nodes: Null folding, Null/Constant, Null/Real, Integer, and an upstream ambiguity.
# Part of the extension CTParserExaModels, included by CTParserExaModels.jl.

# ============================================================================
# Null node pass-throughs for LinearAlgebra functions
#
# A Null wraps a constant, so a unary function of it is again a constant and can
# be folded now rather than left as a Node1 for the AD kernels to walk.
# ============================================================================

for f in [:inv, :abs, :sqrt, :cbrt, :abs2, :exp, :log, :sin, :cos, :tan]
    @eval @inline function Base.$f(x::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$f(x.value))
    end
end

# ============================================================================
# Scalar Null arithmetic with zero/one elimination
# ============================================================================

# Null op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(
        a::ExaModels.Null{T}, b::ExaModels.Null{S}
    ) where {T<:Real,S<:Real}
        return ExaModels.Null(Base.$op(a.value, b.value))
    end
end

# Null + AbstractNode / AbstractNode + Null (zero elimination)
@inline function Base.:+(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return a.value == zero(T) ? b : ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return b.value == zero(T) ? a : ExaModels.Node2(+, a, b)
end

# Null * AbstractNode / AbstractNode * Null (zero/one elimination)
@inline function Base.:*(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return if a.value == zero(T)
        ExaModels.Null(zero(T))
    elseif a.value == one(T)
        b
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return if b.value == zero(T)
        ExaModels.Null(zero(T))
    elseif b.value == one(T)
        a
    else
        ExaModels.Node2(*, a, b)
    end
end

# Null - AbstractNode / AbstractNode - Null
@inline function Base.:-(a::ExaModels.Null{T}, b::ExaModels.AbstractNode) where {T<:Real}
    return a.value == zero(T) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::ExaModels.Null{T}) where {T<:Real}
    return b.value == zero(T) ? a : ExaModels.Node2(-, a, b)
end

# ----------------------------------------------------------------------------
# Null op Constant: intersections with ExaModels' Constant folding rules
#
# ExaModels folds a structural zero or one written as a `Constant{V}`
# (src/specialization.jl, every 0.12.x): `+(::Constant{0}, ::AbstractNode)`,
# `*(::AbstractNode, ::Constant{1})`, … for the zero and one of Int, Float64, Float32 and
# Float16 — `+` and `-` for the zeros, `*` for the zeros and the ones. A call with a
# `Null` on the other side matched one of those and one of the `Null` rules above, and
# neither was more specific (control-toolbox/CTParser.jl#364).
#
# Both operands are constants, so the result is folded to a `Null`, as for `Null op Real`
# below; the value is the one both sides agree on (`Null(a) - Constant(0)` is `Null(a)`,
# `Constant(0) * Null(a)` is `Null(0)`). Exactly these 32 signatures: a `Constant` with
# any other value is not ambiguous and stays with the `Null op AbstractNode` rules.
# ----------------------------------------------------------------------------

for z in (0, 0.0, Float32(0), Float16(0)), op in (:+, :-, :*)
    @eval @inline function Base.$op(
        a::ExaModels.Null{T}, ::ExaModels.Constant{$z}
    ) where {T<:Real}
        return ExaModels.Null(Base.$op(a.value, $z))
    end
    @eval @inline function Base.$op(
        ::ExaModels.Constant{$z}, b::ExaModels.Null{T}
    ) where {T<:Real}
        return ExaModels.Null(Base.$op($z, b.value))
    end
end
for o in (1, 1.0, Float32(1), Float16(1))
    @eval @inline function Base.:*(
        a::ExaModels.Null{T}, ::ExaModels.Constant{$o}
    ) where {T<:Real}
        return ExaModels.Null(a.value * $o)
    end
    @eval @inline function Base.:*(
        ::ExaModels.Constant{$o}, b::ExaModels.Null{T}
    ) where {T<:Real}
        return ExaModels.Null($o * b.value)
    end
end

# Null op Real / Real op Null
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Real) where {T<:Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Real, b::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Null op Integer / Integer op Null — disambiguate Null{T} op Real vs AbstractNode op Integer
for op in (:+, :-, :*)
    @eval @inline function Base.$op(a::ExaModels.Null{T}, b::Integer) where {T<:Real}
        return ExaModels.Null(Base.$op(a.value, b))
    end
    @eval @inline function Base.$op(a::Integer, b::ExaModels.Null{T}) where {T<:Real}
        return ExaModels.Null(Base.$op(a, b.value))
    end
end

# Integer × AbstractNode zero/one elimination (more specific than core's Real × AbstractNode)
# Fixes: 0 * x → Null(0), 1 * x → x, 0 + x → x, etc.
@inline function Base.:*(a::Integer, b::ExaModels.AbstractNode)
    return if iszero(a)
        ExaModels.Null(zero(a))
    elseif isone(a)
        b
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:*(a::ExaModels.AbstractNode, b::Integer)
    return if iszero(b)
        ExaModels.Null(zero(b))
    elseif isone(b)
        a
    else
        ExaModels.Node2(*, a, b)
    end
end
@inline function Base.:+(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? b : ExaModels.Node2(+, a, b)
end
@inline function Base.:+(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(+, a, b)
end
@inline function Base.:-(a::Integer, b::ExaModels.AbstractNode)
    return iszero(a) ? ExaModels.Node1(-, b) : ExaModels.Node2(-, a, b)
end
@inline function Base.:-(a::ExaModels.AbstractNode, b::Integer)
    return iszero(b) ? a : ExaModels.Node2(-, a, b)
end

# ----------------------------------------------------------------------------
# Upstream ambiguity, surfaced by the zero elimination above
#
# ExaModels declares both
#     _hdrpass_val(::Type{<:SecondAdjointNull}, ::Type)
#     _hdrpass_val(::Type, ::Type{<:SecondAdjointNull})
# and no method for the case where *both* arguments are `SecondAdjointNull`, so
# that call is ambiguous (simdfunction.jl:142-143). Folding a structural zero to
# `Null` is what makes both operands Null at once, so the ambiguity only shows up
# once the eliminations above are in play — it hits second-order derivatives of a
# dynamics written with `dot`, under the trapeze scheme.
#
# The value is forced, not a judgement call: both declared methods return `Val(0)`,
# so the intersection must too. This is the "Possible fix" Julia itself prints.
# Delete along with the rest of this file once upstream fixes it.
# ----------------------------------------------------------------------------

function ExaModels._hdrpass_val(
    ::Type{<:ExaModels.SecondAdjointNull}, ::Type{<:ExaModels.SecondAdjointNull}
)
    return Val(0)
end
