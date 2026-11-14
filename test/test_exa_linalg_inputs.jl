module TestExaLinalgInputs

# The linear algebra a user may write in a `:exa` `@def`, with the constants they may
# hold: dense, structured (`Diagonal`, `Symmetric`, triangular, `Tridiagonal`), sparse,
# static (`SMatrix`), lazy (`transpose`, `adjoint`, views) and ranges. Each one is checked
# against the same computation written with a dense `Matrix` / `Vector`:
#
# - unit: products of the constant with a vector of ExaModels variable nodes, evaluated
#   at a point and compared with the numbers;
# - integration: a `@def` whose dynamics and cost use the constant, built and solved,
#   compared with the dense version of the same problem.
#
# This pins what users can rely on, whatever the signatures of the extension are.

using Test: Test
using CTBase: CTBase
using CTParser: CTParser, @def
using CTModels: CTModels
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using MadNLP: MadNLP
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

# The `:exa` builders are generated when `@def` is expanded.
CTParser.activate_backend(:exa)

# ----------------------------------------------------------------------------
# Constants, each with its dense reference
# ----------------------------------------------------------------------------

const M3 = [1.0 2.0 0.0; -1.0 0.5 3.0; 4.0 0.0 -2.0]
const Q3 = [1.0, -2.0, 0.5]

# (name, constant, dense reference) for 3×3 matrices
function matrices3()
    S = M3 + M3'
    return [
        ("Matrix", M3, M3),
        (
            "Diagonal",
            LinearAlgebra.Diagonal([2.0, -1.0, 0.5]),
            [2.0 0 0; 0 -1.0 0; 0 0 0.5],
        ),
        ("Symmetric", LinearAlgebra.Symmetric(S), S),
        (
            "UpperTriangular",
            LinearAlgebra.UpperTriangular(M3),
            Matrix(LinearAlgebra.UpperTriangular(M3)),
        ),
        (
            "LowerTriangular",
            LinearAlgebra.LowerTriangular(M3),
            Matrix(LinearAlgebra.LowerTriangular(M3)),
        ),
        (
            "Tridiagonal",
            LinearAlgebra.Tridiagonal(M3),
            Matrix(LinearAlgebra.Tridiagonal(M3)),
        ),
        ("sparse", SparseArrays.sparse(M3), M3),
        ("SMatrix", StaticArrays.SMatrix{3,3}(M3), M3),
        ("transpose", transpose(M3), Matrix(transpose(M3))),
        ("adjoint", M3', Matrix(M3')),
        ("view", view(M3, :, :), M3),
    ]
end

# (name, constant, dense reference) for vectors of length 3
function vectors3()
    return [
        ("Vector", Q3, Q3),
        ("UnitRange", 1:3, [1.0, 2.0, 3.0]),
        ("StepRangeLen", 0.5:0.5:1.5, [0.5, 1.0, 1.5]),
        ("sparsevec", SparseArrays.sparsevec(Q3), Q3),
        ("SVector", StaticArrays.SVector{3}(Q3), Q3),
        ("view", view(Q3, 1:3), Q3),
    ]
end

# Same, for the 2-state problem of the integration tests
const M2 = [0.0 1.0; -2.0 -0.5]
const Q2 = [0.5, -1.0]

function matrices2()
    S = M2 + M2'
    return [
        ("Matrix", M2, M2),
        ("Diagonal", LinearAlgebra.Diagonal([-1.0, 0.5]), [-1.0 0; 0 0.5]),
        ("Symmetric", LinearAlgebra.Symmetric(S), S),
        (
            "UpperTriangular",
            LinearAlgebra.UpperTriangular(M2),
            Matrix(LinearAlgebra.UpperTriangular(M2)),
        ),
        (
            "LowerTriangular",
            LinearAlgebra.LowerTriangular(M2),
            Matrix(LinearAlgebra.LowerTriangular(M2)),
        ),
        ("sparse", SparseArrays.sparse(M2), M2),
        ("SMatrix", StaticArrays.SMatrix{2,2}(M2), M2),
        ("transpose", transpose(M2), Matrix(transpose(M2))),
        ("view", view(M2, :, :), M2),
    ]
end

function vectors2()
    return [
        ("Vector", Q2, Q2),
        ("UnitRange", 1:2, [1.0, 2.0]),
        ("sparsevec", SparseArrays.sparsevec(Q2), Q2),
        ("SVector", StaticArrays.SVector{2}(Q2), Q2),
        ("view", view(Q2, 1:2), Q2),
    ]
end

# ----------------------------------------------------------------------------
# Unit helpers: variable nodes and their evaluation at a point
# ----------------------------------------------------------------------------

const POINT = [1.5, -2.0, 0.5]

function variable_nodes(n)
    core = ExaModels.ExaCore()
    core, x = ExaModels.add_var(core, n)
    return [x[i] for i in 1:n]
end

# A 3×3 matrix of variable nodes, and the point it is evaluated at (column-major)
function variable_matrix3()
    core = ExaModels.ExaCore()
    core, x = ExaModels.add_var(core, 9)
    return [x[(j - 1) * 3 + i] for i in 1:3, j in 1:3]
end

const POINT9 = collect(1.0:9.0)

evaluate(n::Real) = n
evaluate(n::ExaModels.AbstractNode) = n(1, POINT, nothing)

# ----------------------------------------------------------------------------
# Integration helpers
# ----------------------------------------------------------------------------

# Linear-quadratic problem: well posed for any A, since nothing is asked at the final time.
function lq_problem(A, q)
    return @def begin
        t ∈ [0, 1], time
        x ∈ R², state
        u ∈ R, control
        x(0) == [1, 0]
        ∂(x)(t) == A * x(t) + [0, 1] * u(t)
        ∫(u(t)^2 + x₁(t)^2 + x₂(t)^2 + LinearAlgebra.dot(q, x(t))) → min
    end
end

function quadratic_cost_problem(Q)
    return @def begin
        t ∈ [0, 1], time
        x ∈ R², state
        u ∈ R, control
        x(0) == [1, 0]
        ∂(x)(t) == [x₂(t), u(t)]
        ∫(u(t)^2 + x(t)' * Q * x(t)) → min
    end
end

function build_exa(ocp; scheme)
    build = CTModels.get_build_examodel(ocp)
    model, _ = build(;
        scheme=scheme,
        grid_size=20,
        backend=CTParser.__default_backend_exa(),
        init=CTParser.__default_init_exa(),
        base_type=Float64,
    )
    return model
end

solve(model) = MadNLP.madnlp(model; print_level=MadNLP.ERROR, tol=1e-10)

# Build and solve `ocp` and its dense reference, and compare.
function same_solution(ocp, reference; scheme)
    m = build_exa(ocp; scheme=scheme)
    r = build_exa(reference; scheme=scheme)
    sm, sr = solve(m), solve(r)
    return m.meta.nvar == r.meta.nvar &&
           m.meta.ncon == r.meta.ncon &&
           sm.status == MadNLP.SOLVE_SUCCEEDED &&
           sr.status == MadNLP.SOLVE_SUCCEEDED &&
           isapprox(sm.objective, sr.objective; rtol=1e-8)
end

function test_exa_linalg_inputs()
    Test.@testset "Linear algebra inputs with :exa" verbose = VERBOSE showtiming =
        SHOWTIMING begin

        # ------------------------------------------------------------------
        # Unit: constants of every kind with a vector of variable nodes
        # ------------------------------------------------------------------
        Test.@testset "A * x for a $name matrix" for (name, A, D) in matrices3()
            v = variable_nodes(3)
            Test.@test evaluate.(A * v) ≈ D * POINT
        end

        # Products of two matrices, a matrix of nodes with a constant on either side. Julia 1.10
        # defines more of these for structured and sparse matrices than 1.12 does, each one
        # ambiguous with ours until covered (#365).
        Test.@testset "X * A and A * X for a $name matrix" for (name, A, D) in matrices3()
            X = variable_matrix3()
            Xd = reshape(POINT9, 3, 3)
            ev(M) = [n(1, POINT9, nothing) for n in M]
            Test.@test ev(X * A) ≈ Xd * D
            Test.@test ev(A * X) ≈ D * Xd
        end

        Test.@testset "x' * A for a $name matrix" for (name, A, D) in matrices3()
            v = variable_nodes(3)
            Test.@test evaluate.(v' * A) ≈ POINT' * D
        end

        Test.@testset "x' * A * x for a $name matrix" for (name, A, D) in matrices3()
            v = variable_nodes(3)
            Test.@test evaluate(v' * A * v) ≈ POINT' * D * POINT
        end

        Test.@testset "dot(q, x), dot(x, q) and q' * x for a $name vector" for (
            name, q, D
        ) in vectors3()
            v = variable_nodes(3)
            Test.@test evaluate(LinearAlgebra.dot(q, v)) ≈ LinearAlgebra.dot(D, POINT)
            Test.@test evaluate(LinearAlgebra.dot(v, q)) ≈ LinearAlgebra.dot(POINT, D)
            Test.@test evaluate(q' * v) ≈ D' * POINT
        end

        Test.@testset "x + q and q - x for a $name vector" for (name, q, D) in vectors3()
            v = variable_nodes(3)
            Test.@test evaluate.(v + q) ≈ POINT + D
            Test.@test evaluate.(q - v) ≈ D - POINT
        end

        # ------------------------------------------------------------------
        # Integration: the same constants in a `@def`, built and solved
        # ------------------------------------------------------------------
        Test.@testset "@def with a $name matrix in the dynamics ($scheme)" for (
                name, A, D
            ) in matrices2(),
            scheme in (:euler, :trapeze)

            Test.@test same_solution(lq_problem(A, Q2), lq_problem(D, Q2); scheme=scheme)
        end

        Test.@testset "@def with a $name vector in the cost ($scheme)" for (name, q, D) in
                                                                           vectors2(),
            scheme in (:euler, :trapeze)

            Test.@test same_solution(lq_problem(M2, q), lq_problem(M2, D); scheme=scheme)
        end

        Test.@testset "@def with x' * Q * x for a $name matrix" for (name, Q, D) in [
            ("Matrix", [2.0 0.5; 0.5 1.0], [2.0 0.5; 0.5 1.0]),
            ("Diagonal", LinearAlgebra.Diagonal([2.0, 1.0]), [2.0 0; 0 1.0]),
            ("Symmetric", LinearAlgebra.Symmetric([2.0 0.5; 0.5 1.0]), [2.0 0.5; 0.5 1.0]),
            ("SMatrix", StaticArrays.SMatrix{2,2}([2.0 0.5; 0.5 1.0]), [2.0 0.5; 0.5 1.0]),
        ]
            Test.@test same_solution(
                quadratic_cost_problem(Q), quadratic_cost_problem(D); scheme=:trapeze
            )
        end
    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_exa_linalg_inputs() = TestExaLinalgInputs.test_exa_linalg_inputs()
