# Pirated methods of the extensions of CTParser, accepted (control-toolbox/CTParser.jl#365),
# by extension and printed signature, with the reason. Read by test_aqua.jl, which runs the
# ct-conventions TST8 check (`CTBase.DevTools.test_extensions_quality`): it fails on any
# pirated method missing here, and on any entry here that is no longer pirated.
#
# Generated once from Aqua's piracy hunt, then reviewed. When a method of an extension is
# added, removed or its signature changes, update this file by hand. The signatures print
# the same on Julia 1.10, 1.12 and 1.13 (the extensions name their type variables for that),
# but they depend on what `Main` imports: they are written as a test run prints them, with
# `using LinearAlgebra` in runtests.jl and SparseArrays / StaticArrays loaded qualified. If
# runtests.jl changes its imports, regenerate this file.

const EXTENSION_PIRACIES = Dict(
    "CTParserExaModels" => Dict(
        "Tuple{typeof(*), AbstractMatrix{<:Real}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), AbstractMatrix{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), AbstractMatrix{<:Real}, Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), AbstractVector{<:Real}, Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, AbstractMatrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Adjoint{<:Real, <:AbstractVector}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Union{Adjoint{<:Real, <:Transpose{<:Any, <:AbstractVector}}, Transpose{<:Real, <:Adjoint{<:Any, <:AbstractVector}}}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), Adjoint{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, <:StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Union{LinearAlgebra.AbstractTriangular{<:Real}, Diagonal{T, V} where {T<:Real, V<:AbstractVector{T}}}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{1.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{1.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{1.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Constant{1}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Diagonal{T, V} where V<:AbstractVector{T}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}} where T<:Real" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), ExaModels.AbstractNode, Integer}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), ExaModels.AbstractNode, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Integer, ExaModels.AbstractNode}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Integer, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), LinearAlgebra.AbstractTriangular{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{1.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{1.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{1.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, Constant{1}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(*), Null{T}, ExaModels.AbstractNode} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Null{T}, Integer} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Null{T}, Null{S}} where {T<:Real, S<:Real}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Null{T}, Real} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Real, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Real, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Real, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractMatrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Real}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Union{LinearAlgebra.AbstractTriangular{<:Real}, Diagonal{T, V} where {T<:Real, V<:AbstractVector{T}}}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Real}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Union{Adjoint{<:Real, <:AbstractVector}, Transpose{<:Real, <:AbstractVector}}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), Union{Adjoint{<:Real, <:AbstractVector}, Transpose{<:Real, <:AbstractVector}}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(*), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractMatrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(*), Union{LinearAlgebra.AbstractTriangular{<:Real}, Diagonal{T, V} where {T<:Real, V<:AbstractVector{T}}}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "intersection with a LinearAlgebra method on a structured matrix or a transposed row (#365)",
        "Tuple{typeof(+), AbstractMatrix{<:Real}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), AbstractVector{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Constant{0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), ExaModels.AbstractNode, Integer}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), ExaModels.AbstractNode, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Integer, ExaModels.AbstractNode}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Integer, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Matrix{<:Real}, Matrix{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Matrix{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Matrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Matrix{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Matrix{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Null{T}, Constant{0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(+), Null{T}, ExaModels.AbstractNode} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Null{T}, Integer} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Null{T}, Null{S}} where {T<:Real, S<:Real}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Null{T}, Real} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Real, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractMatrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Vector{<:Real}, Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Vector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(+), Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}, Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), AbstractMatrix{<:Real}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), AbstractVector{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Constant{0.0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Constant{0}, Null{T}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), ExaModels.AbstractNode, Integer}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), ExaModels.AbstractNode, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Integer, ExaModels.AbstractNode}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Integer, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Null{T}, Constant{0.0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Null{T}, Constant{0}} where T<:Real" =>
            "Null op Constant: intersection with ExaModels' Constant folding rules (#364)",
        "Tuple{typeof(-), Null{T}, ExaModels.AbstractNode} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Null{T}, Integer} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Null{T}, Null{S}} where {T<:Real, S<:Real}" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Null{T}, Real} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), Real, Null{T}} where T<:Real" =>
            "zero/one elimination on scalar nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractMatrix{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(-), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(ExaModels._hdrpass_val), Type{<:SecondAdjointNull}, Type{<:SecondAdjointNull}}" =>
            "ExaModels' own ambiguity on two SecondAdjointNull, reached once structural zeros are folded",
        "Tuple{typeof(abs), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(abs2), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(adjoint), ExaModels.AbstractAdjointNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(adjoint), ExaModels.AbstractNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(adjoint), ExaModels.AbstractSecondAdjointNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(adjoint), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(cbrt), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(convert), Type{ExaModels.AbstractNode}, Real}" =>
            "numbers promote to nodes, so that [x, 0] is a vector of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(cos), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(cross), AbstractVector{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(cross), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(cross), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(det), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(diag), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(diagm), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(dot), AbstractVector{<:Real}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(dot), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, AbstractVector{<:Real}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(dot), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(exp), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(inv), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(log), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(norm), Union{Base.ReshapedArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 1, <:Array}, SubArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 1, <:Array}, Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}, Real}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(norm), Union{Base.ReshapedArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 1, <:Array}, SubArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 1, <:Array}, Vector{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(norm), Union{Base.ReshapedArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 2, <:Array}, SubArray{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, 2, <:Array}, Matrix{<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(one), Type{<:ExaModels.AbstractNode}}" =>
            "structural zero and one of a node, used by array code; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(one), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "structural zero and one of a node, used by array code; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(promote_rule), Type{<:ExaModels.AbstractNode}, Type{<:Real}}" =>
            "numbers promote to nodes, so that [x, 0] is a vector of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(sin), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(sqrt), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(sum), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "linear algebra on arrays of nodes; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(tan), Null{T}} where T<:Real" =>
            "a function of a constant Null is folded to a Null; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(transpose), ExaModels.AbstractAdjointNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(transpose), ExaModels.AbstractNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(transpose), ExaModels.AbstractSecondAdjointNode}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(transpose), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "nodes are real: adjoint and transpose; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(zero), Type{<:ExaModels.AbstractNode}}" =>
            "structural zero and one of a node, used by array code; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
        "Tuple{typeof(zero), Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "structural zero and one of a node, used by array code; port of ExaModels' linear-algebra extension, which ExaModels ships but does not register (#325)",
    ),
    "CTParserExaModelsSparseArrays" => Dict(
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, SparseArrays.AbstractSparseVector{<:Real}}" =>
            "sparse constant with an array of nodes: intersection with SparseArrays' own method (#365)",
        "Tuple{typeof(*), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Union{SubArray{T, 2, <:SparseArrays.AbstractSparseMatrixCSC{T, Ti}, Tuple{Base.Slice{Base.OneTo{Int64}}, I}} where {T<:Real, Ti, I<:AbstractUnitRange}, Adjoint{<:Real, <:SparseArrays.AbstractSparseMatrixCSC}, Transpose{<:Real, <:SparseArrays.AbstractSparseMatrixCSC}, SparseArrays.AbstractSparseMatrixCSC{T, Ti} where {T<:Real, Ti}}}" =>
            "sparse constant with an array of nodes: intersection with SparseArrays' own method (#365)",
        "Tuple{typeof(dot), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, Union{SparseArrays.AbstractCompressedVector{T, Ti}, SubArray{T, 1, <:SparseArrays.AbstractSparseMatrixCSC{T, Ti}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, false}, SubArray{T, 1, <:SparseArrays.AbstractSparseVector{T, Ti}, Tuple{Base.Slice{Base.OneTo{Int64}}}, false}} where {T<:Real, Ti}}" =>
            "sparse constant with an array of nodes: intersection with SparseArrays' own method (#365)",
        "Tuple{typeof(dot), Union{SparseArrays.AbstractCompressedVector{T, Ti}, SubArray{T, 1, <:SparseArrays.AbstractSparseMatrixCSC{T, Ti}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, false}, SubArray{T, 1, <:SparseArrays.AbstractSparseVector{T, Ti}, Tuple{Base.Slice{Base.OneTo{Int64}}}, false}} where {T<:Real, Ti}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "sparse constant with an array of nodes: intersection with SparseArrays' own method (#365)",
    ),
    "CTParserExaModelsStaticArrays" => Dict(
        "Tuple{typeof(*), StaticArraysCore.StaticArray{Tuple{N, M}, <:Real, 2} where {N, M}, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(+), StaticArraysCore.StaticArray{S, <:Real, 1} where S<:Tuple, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(+), StaticArraysCore.StaticArray{S, <:Real, 2} where S<:Tuple, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(+), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StaticArraysCore.StaticArray{S, <:Real, 2} where S<:Tuple}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(+), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StaticArraysCore.StaticArray{S, <:Real, 1} where S<:Tuple}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(-), StaticArraysCore.StaticArray{S, <:Real, 1} where S<:Tuple, StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(-), StaticArraysCore.StaticArray{S, <:Real, 2} where S<:Tuple, StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(-), StridedMatrix{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StaticArraysCore.StaticArray{S, <:Real, 2} where S<:Tuple}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
        "Tuple{typeof(-), StridedVector{T} where T<:Union{ExaModels.AbstractAdjointNode, ExaModels.AbstractNode, ExaModels.AbstractSecondAdjointNode}, StaticArraysCore.StaticArray{S, <:Real, 1} where S<:Tuple}" =>
            "static constant with an array of nodes: intersection with StaticArrays' own method (#365)",
    ),
)
