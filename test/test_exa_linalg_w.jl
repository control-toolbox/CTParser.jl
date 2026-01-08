# test_exa_linalg_w - Tests for ExaLinAlg_w module

using .ExaLinAlg_w

# Import the module
using LinearAlgebra: norm_sqr

function test_exa_linalg_w()

    # Setup: Create symbolic variables for testing
    c = ExaModels.ExaCore()
    X = ExaModels.variable(c, 5, 4)

    # Wrap with new SymVector/SymMatrix types
    x = SymVector([X[i, 1] for i in 1:5])
    y = SymVector([X[i, 2] for i in 1:5])
    M = SymMatrix([X[i, j] for i in 1:3, j in 1:3])
    N = SymMatrix([X[i, j] for i in 1:2, j in 1:4])

    A = randn(5, 5)
    B = randn(3, 3)
    C = randn(2, 2)
    v = randn(5)
    w = randn(3)

    @testset "Basic SymNumber properties" begin
        println("  Testing basic SymNumber properties")

        # x[1] now returns SymNumber
        @test x[1] isa SymNumber

        # zero and one return SymNumber
        @test zero(SymNumber) isa SymNumber
        @test zero(x[1]) isa SymNumber
        @test one(SymNumber) isa SymNumber
        @test one(x[1]) isa SymNumber

        # Unwrapped values are Null
        @test unwrap_scalar(zero(SymNumber)) isa ExaModels.Null
        @test unwrap_scalar(one(SymNumber)) isa ExaModels.Null

        # Scalar properties
        @test length(x[1]) == 1
        @test size(x[1]) == ()
        @test ndims(x[1]) == 0
        @test ndims(typeof(x[1])) == 0

        # adjoint/transpose/conj for scalars
        @test adjoint(x[1]) === x[1]
        @test transpose(x[1]) === x[1]
        @test conj(x[1]) === x[1]

        # broadcastable
        @test Base.broadcastable(x[1]) isa Base.RefValue

        # iterate should return nothing (not iterable)
        @test iterate(x[1]) === nothing
    end

    @testset "Type promotion and conversion" begin
        println("  Testing type promotion and conversion")

        @test convert(SymNumber, 5.0) isa SymNumber
        @test convert(SymNumber, x[1]) === x[1]

        # Promotion with numbers
        @test promote_type(typeof(x[1]), Float64) == SymNumber
        @test promote_type(typeof(x[1]), Int) == SymNumber
    end

    @testset "Symbolic arithmetic helpers" begin
        println("  Testing sym_add and sym_mul")

        # sym_add with Null(nothing) - wrapping in SymNumber
        null_zero = SymNumber(ExaModels.Null(nothing))
        @test ExaLinAlg_w.sym_add(null_zero, x[1]) === x[1]
        @test ExaLinAlg_w.sym_add(x[1], null_zero) === x[1]
        @test ExaLinAlg_w.sym_add(x[1], y[1]) isa SymNumber

        # sym_mul
        @test ExaLinAlg_w.sym_mul(x[1], y[1]) isa SymNumber
        @test ExaLinAlg_w.sym_mul(2.0, x[1]) isa SymNumber
    end

    @testset "Matrix-vector products" begin
        println("  Testing matrix-vector products")

        # Numeric matrix × Symbolic vector
        @test A * x isa SymVector
        @test length(A * x) == size(A, 1)

        # Symbolic matrix × Numeric vector
        @test M * v[1:3] isa SymVector
        @test length(M * v[1:3]) == size(M, 1)

        # Symbolic matrix × Symbolic vector
        @test M * x[1:3] isa SymVector
        @test length(M * x[1:3]) == size(M, 1)

        # Different sizes
        @test B * x[1:3] isa SymVector
        @test length(B * x[1:3]) == 3
    end

    @testset "Row vector × Matrix (via adjoint)" begin
        println("  Testing row vector × matrix")

        # Symbolic row × Numeric matrix
        result = x' * A
        @test result isa LinearAlgebra.Adjoint
        @test size(parent(result)) == (size(A, 2),)

        # Numeric row × Symbolic matrix
        result = v[1:3]' * M
        @test result isa LinearAlgebra.Adjoint
        @test size(parent(result)) == (size(M, 2),)

        # Symbolic row × Symbolic matrix
        result = x[1:3]' * M
        @test result isa LinearAlgebra.Adjoint
        @test size(parent(result)) == (size(M, 2),)
    end

    @testset "Matrix × Matrix products" begin
        println("  Testing matrix × matrix products")

        # Numeric × Symbolic
        @test B * M isa SymMatrix
        @test size(B * M) == (size(B, 1), size(M, 2))

        # Symbolic × Numeric
        @test M * B isa SymMatrix
        @test size(M * B) == (size(M, 1), size(B, 2))

        # Symbolic × Symbolic
        @test M * M isa SymMatrix
        @test size(M * M) == (size(M, 1), size(M, 2))

        # Different dimensions
        M2x3 = SymMatrix([X[i, j] for i in 1:2, j in 1:3])
        M3x4 = SymMatrix([X[i, j] for i in 1:3, j in 1:4])
        @test M2x3 * M3x4 isa SymMatrix
        @test size(M2x3 * M3x4) == (2, 4)
    end

    @testset "Dot products" begin
        println("  Testing dot products")

        # Symbolic · Symbolic - now returns SymNumber
        @test dot(x, y) isa SymNumber
        @test dot(x[1:3], y[1:3]) isa SymNumber

        # Numeric · Symbolic
        @test dot(v, x) isa SymNumber
        @test dot(v[1:3], x[1:3]) isa SymNumber

        # Symbolic · Numeric
        @test dot(x, v) isa SymNumber
        @test dot(x[1:3], v[1:3]) isa SymNumber
    end

    @testset "Inner products via adjoint" begin
        println("  Testing inner products (x' * y)")

        # Symbolic' × Symbolic - now returns SymNumber
        @test x' * y isa SymNumber
        @test x' * x isa SymNumber

        # Symbolic' × Numeric
        @test x' * v isa SymNumber
        @test x[1:3]' * v[1:3] isa SymNumber

        # Numeric' × Symbolic
        @test v' * x isa SymNumber
        @test v[1:3]' * x[1:3] isa SymNumber
    end

    @testset "Quadratic forms" begin
        println("  Testing quadratic forms")

        # x' * A * x - now returns SymNumber
        @test x' * A * x isa SymNumber
        @test x[1:3]' * M * x[1:3] isa SymNumber
        @test x[1:3]' * B * x[1:3] isa SymNumber
    end

    @testset "Matrix transpose and adjoint" begin
        println("  Testing matrix transpose and adjoint")

        # Transpose
        @test M' isa SymMatrix
        @test size(M') == (size(M, 2), size(M, 1))
        @test transpose(M) isa SymMatrix
        @test size(transpose(M)) == (size(M, 2), size(M, 1))

        # Non-square matrix
        @test N' isa SymMatrix
        @test size(N') == (size(N, 2), size(N, 1))

        # Transpose should work in products
        @test M' * x[1:3] isa SymVector
        @test length(M' * x[1:3]) == size(M, 2)
    end

    @testset "Vector norms" begin
        println("  Testing vector norms")

        # Default norm (L2) - now returns SymNumber
        @test norm(x) isa SymNumber
        @test norm(y) isa SymNumber

        # norm_sqr
        @test norm_sqr(x) isa SymNumber
        @test norm_sqr(x[1:3]) isa SymNumber

        # Explicit L2 norm
        @test norm(x, 2) isa SymNumber

        # L1 norm
        @test norm(x, 1) isa SymNumber
        @test norm(x[1:3], 1) isa SymNumber

        # Lp norms
        @test norm(x, 3) isa SymNumber
        @test norm(x, 4) isa SymNumber
    end

    @testset "Matrix norms" begin
        println("  Testing matrix norms")

        # Frobenius norm (default) - now returns SymNumber
        @test norm(M) isa SymNumber
        @test norm(N) isa SymNumber
    end

    @testset "Broadcasting - unary operations" begin
        println("  Testing broadcasting with unary operations")

        # Unary functions on vectors
        @test sin.(x) isa SymVector
        @test length(sin.(x)) == length(x)
        @test cos.(x) isa SymVector
        @test exp.(x) isa SymVector
        @test log.(x) isa SymVector
        @test sqrt.(x) isa SymVector
        @test abs.(x) isa SymVector

        # Unary functions on matrices
        @test sin.(M) isa SymMatrix
        @test size(sin.(M)) == size(M)
        @test cos.(M) isa SymMatrix
        @test exp.(M) isa SymMatrix
        @test log.(M) isa SymMatrix
    end

    @testset "Broadcasting - binary operations" begin
        println("  Testing broadcasting with binary operations")

        # Element-wise arithmetic on vectors
        @test x .+ y isa SymVector
        @test length(x .+ y) == length(x)
        @test x .- y isa SymVector
        @test x .* y isa SymVector
        @test x ./ y isa SymVector
        @test x .^ 2 isa SymVector
        @test x .^ y isa SymVector

        # Element-wise with scalars
        @test 2.0 .* x isa SymVector
        @test x .+ 1.0 isa SymVector
        @test x .- 3.0 isa SymVector
        @test x ./ 2.0 isa SymVector

        # Element-wise on matrices
        @test M .+ M isa SymMatrix
        @test size(M .+ M) == size(M)
        @test M .- M isa SymMatrix
        @test M .* M isa SymMatrix
        @test 2.0 .* M isa SymMatrix
        @test M ./ 2.0 isa SymMatrix
    end

    @testset "Broadcasting - compound expressions" begin
        println("  Testing broadcasting with compound expressions")

        # Compound vector expressions
        @test exp.(x) .+ 1 isa SymVector
        @test sin.(x) .* cos.(y) isa SymVector
        @test (x .+ y) ./ 2 isa SymVector
        @test sqrt.(x.^2 .+ y.^2) isa SymVector

        # Compound matrix expressions
        @test sin.(M) .+ cos.(M) isa SymMatrix
        @test (M .+ M) ./ 2 isa SymMatrix
        @test exp.(M) .* 2.0 isa SymMatrix
    end

    @testset "abs and abs2" begin
        println("  Testing abs and abs2")

        # abs2 for scalar node - now returns SymNumber
        @test abs2(x[1]) isa SymNumber

        # abs2 via broadcasting
        @test abs2.(x) isa SymVector
        @test length(abs2.(x)) == length(x)
    end

    @testset "Edge cases and dimension mismatches" begin
        println("  Testing edge cases")

        # Test that dimension mismatches are caught
        A_wrong = randn(3, 4)
        @test_throws AssertionError A_wrong * x  # 4 != 5

        @test_throws AssertionError M * v  # 3 != 5

        # Different length dot products
        @test_throws AssertionError dot(x[1:3], y[1:4])

        # Matrix dimension mismatch
        M2x3 = SymMatrix([X[i, j] for i in 1:2, j in 1:3])
        M4x2 = SymMatrix([X[i, j] for i in 1:4, j in 1:2])
        @test_throws AssertionError M2x3 * M4x2  # 3 != 4
    end

    @testset "Mixed operations" begin
        println("  Testing mixed operations")

        # Combine different operations
        result = A * x + v
        @test result isa SymVector
        @test length(result) == length(v)

        # Matrix chain
        result = M * M * M
        @test result isa SymMatrix
        @test size(result) == size(M)

        # Complex expression - now returns SymNumber
        result = (x' * A * x) + dot(x, v) + norm(x)^2
        @test result isa SymNumber

        # Broadcasting mixed with products
        result = (A * x) .+ v
        @test result isa SymVector

        # Norm of matrix-vector product - now returns SymNumber
        result = norm(M * x[1:3])
        @test result isa SymNumber
    end

    println("  All exa_linalg_w tests passed!")
end
