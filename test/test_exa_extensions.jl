module TestExaExtensions

# The two small extensions that complete `CTParserExaModels` for sparse and static numeric
# constants (`CTParserExaModelsSparseArrays`, `CTParserExaModelsStaticArrays`): they load
# with their triggers, agree with the main extension on what an array of nodes is, compute
# the right values, keep sparse graphs sparse, and raise typed errors.

using Test: Test
using CTBase: CTBase
using CTParser: CTParser
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

const MAIN = Base.get_extension(CTParser, :CTParserExaModels)
const SPARSE = Base.get_extension(CTParser, :CTParserExaModelsSparseArrays)
const STATIC = Base.get_extension(CTParser, :CTParserExaModelsStaticArrays)

const POINT = [1.5, -2.0, 0.5]

function variable_nodes(n)
    core = ExaModels.ExaCore()
    core, x = ExaModels.add_var(core, n)
    return [x[i] for i in 1:n]
end

function variable_matrix(m, n)
    core = ExaModels.ExaCore()
    core, x = ExaModels.add_var(core, m * n)
    return [x[(j - 1) * m + i] for i in 1:m, j in 1:n]
end

evaluate(n::Real, point=POINT) = n
evaluate(n::ExaModels.AbstractNode, point=POINT) = n(1, point, nothing)

# Number of `+` nodes in an expression: one fewer than the number of terms of a sum.
count_plus(::Any) = 0
count_plus(n::ExaModels.Node2{typeof(+)}) = 1 + count_plus(n.inner1) + count_plus(n.inner2)
count_plus(n::ExaModels.Node2) = count_plus(n.inner1) + count_plus(n.inner2)

function test_exa_extensions()
    Test.@testset "ExaModels extensions for sparse and static constants" verbose = VERBOSE showtiming =
        SHOWTIMING begin

        # ------------------------------------------------------------------
        # Contract
        # ------------------------------------------------------------------
        Test.@testset "the three extensions are loaded with their triggers" begin
            Test.@test MAIN isa Module
            Test.@test SPARSE isa Module
            Test.@test STATIC isa Module
        end

        Test.@testset "they agree on what an array of nodes is" begin
            for ext in (SPARSE, STATIC), name in (:ExaNode, :VecExaNode, :MatExaNode)
                Test.@test getfield(ext, name) == getfield(MAIN, name)
            end
        end

        # ------------------------------------------------------------------
        # Unit: sparse constants
        # ------------------------------------------------------------------
        Test.@testset "dot with a sparse vector, both orders" begin
            v = variable_nodes(3)
            q = SparseArrays.sparsevec([1, 3], [2.0, -1.0], 3)
            Test.@test evaluate(LinearAlgebra.dot(q, v)) ≈ 2.0 * POINT[1] - POINT[3]
            Test.@test evaluate(LinearAlgebra.dot(v, q)) ≈ 2.0 * POINT[1] - POINT[3]
            Test.@test evaluate(q' * v) ≈ 2.0 * POINT[1] - POINT[3]
        end

        Test.@testset "a sparse dot has one term per stored nonzero" begin
            v = variable_nodes(3)
            one_nonzero = SparseArrays.sparsevec([2], [2.0], 3)
            r = LinearAlgebra.dot(one_nonzero, v)
            Test.@test count_plus(r) == 0
            Test.@test evaluate(r) ≈ 2.0 * POINT[2]
            two_nonzeros = SparseArrays.sparsevec([1, 3], [2.0, -1.0], 3)
            Test.@test count_plus(LinearAlgebra.dot(two_nonzeros, v)) == 1
            # an explicitly stored zero adds no term
            with_stored_zero = SparseArrays.SparseVector(3, [1, 2], [0.0, 4.0])
            Test.@test count_plus(LinearAlgebra.dot(with_stored_zero, v)) == 0
            Test.@test evaluate(LinearAlgebra.dot(with_stored_zero, v)) ≈ 4.0 * POINT[2]
        end

        Test.@testset "dot with a column view of a sparse matrix" begin
            v = variable_nodes(3)
            S = SparseArrays.sparse([1.0 0.0; 0.0 2.0; 3.0 0.0])
            col = view(S, :, 1)
            Test.@test evaluate(LinearAlgebra.dot(v, col)) ≈ POINT[1] + 3.0 * POINT[3]
        end

        Test.@testset "a dot with an empty sparse vector is a structural zero" begin
            v = variable_nodes(3)
            r = LinearAlgebra.dot(SparseArrays.spzeros(3), v)
            Test.@test r isa ExaModels.Null
            Test.@test iszero(r.value)
        end

        Test.@testset "matrix of nodes times a sparse vector" begin
            X = variable_matrix(2, 3)
            point = collect(1.0:6.0)
            q = SparseArrays.sparsevec([2], [3.0], 3)
            r = X * q
            Test.@test length(r) == 2
            Test.@test [evaluate(ri, point) for ri in r] ≈ reshape(point, 2, 3) * Vector(q)
        end

        # ------------------------------------------------------------------
        # Unit: static constants
        # ------------------------------------------------------------------
        Test.@testset "vector of nodes ± a static vector, both orders" begin
            v = variable_nodes(3)
            q = StaticArrays.SVector(1.0, 2.0, 3.0)
            Test.@test (v + q) isa Vector
            Test.@test evaluate.(v + q) ≈ POINT + Vector(q)
            Test.@test evaluate.(q + v) ≈ Vector(q) + POINT
            Test.@test evaluate.(v - q) ≈ POINT - Vector(q)
            Test.@test evaluate.(q - v) ≈ Vector(q) - POINT
        end

        Test.@testset "matrix of nodes ± a static matrix, both orders" begin
            X = variable_matrix(2, 2)
            point = [1.0, 2.0, 3.0, 4.0]
            A = StaticArrays.SMatrix{2,2}(10.0, 20.0, 30.0, 40.0)
            ev(M) = [evaluate(m, point) for m in M]
            Test.@test (X + A) isa Matrix
            Test.@test ev(X + A) ≈ reshape(point, 2, 2) + Matrix(A)
            Test.@test ev(A - X) ≈ Matrix(A) - reshape(point, 2, 2)
        end

        Test.@testset "static matrix times a vector of nodes" begin
            v = variable_nodes(3)
            A = StaticArrays.SMatrix{2,3}(1.0, 0.0, 2.0, -1.0, 0.0, 3.0)
            r = A * v
            Test.@test r isa Vector
            Test.@test evaluate.(r) ≈ Matrix(A) * POINT
        end

        # ------------------------------------------------------------------
        # Error: dimensions that do not match
        # ------------------------------------------------------------------
        Test.@testset "incompatible dimensions raise a PreconditionError" begin
            v = variable_nodes(3)
            Test.@test_throws CTBase.PreconditionError LinearAlgebra.dot(SparseArrays.spzeros(2), v)
            Test.@test_throws CTBase.PreconditionError variable_matrix(2, 2) * SparseArrays.spzeros(3)
            Test.@test_throws CTBase.PreconditionError v + StaticArrays.SVector(1.0, 2.0)
            Test.@test_throws CTBase.PreconditionError StaticArrays.SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) * v
        end
    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_exa_extensions() = TestExaExtensions.test_exa_extensions()
