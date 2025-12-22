# test_exa_linalg

function test_exa_linalg()

    # Setup: Create symbolic variables for testing
    c = ExaModels.ExaCore()
    X = ExaModels.variable(c, 5, 4)

    x = [X[i, 1] for i in 1:5]
    y = [X[i, 2] for i in 1:5]
    M = [X[i, j] for i in 1:3, j in 1:3]
    N = [X[i, j] for i in 1:2, j in 1:4]

    A = randn(5, 5)
    B = randn(3, 3)
    C = randn(2, 2)
    v = randn(5)
    w = randn(3)

    @testset "Basic AbstractNode properties" begin
        println("  Testing basic AbstractNode properties")

        # zero and one
        @test zero(typeof(x[1])) isa ExaModels.Null
        @test zero(x[1]) isa ExaModels.Null
        @test one(typeof(x[1])) isa ExaModels.Null
        @test one(x[1]) isa ExaModels.Null

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

        @test convert(AbstractNode, 5.0) isa ExaModels.Null
        @test convert(AbstractNode, x[1]) === x[1]

        # Promotion with numbers
        @test promote_type(typeof(x[1]), Float64) == AbstractNode
        @test promote_type(typeof(x[1]), Int) == AbstractNode
    end

    @testset "Symbolic arithmetic helpers" begin
        println("  Testing sym_add and sym_mul")

        # sym_add with Null(nothing)
        null_zero = ExaModels.Null(nothing)
        @test CTParser.sym_add(null_zero, x[1]) === x[1]
        @test CTParser.sym_add(x[1], null_zero) === x[1]
        @test CTParser.sym_add(x[1], y[1]) isa AbstractNode

        # sym_mul
        @test CTParser.sym_mul(x[1], y[1]) isa AbstractNode
        @test CTParser.sym_mul(2.0, x[1]) isa AbstractNode
    end

    @testset "Matrix-vector products" begin
        println("  Testing matrix-vector products")

        # Numeric matrix × Symbolic vector
        @test A * x isa Vector{<:AbstractNode}
        @test length(A * x) == size(A, 1)

        # Symbolic matrix × Numeric vector
        @test M * v[1:3] isa Vector{<:AbstractNode}
        @test length(M * v[1:3]) == size(M, 1)

        # Symbolic matrix × Symbolic vector
        @test M * x[1:3] isa Vector{<:AbstractNode}
        @test length(M * x[1:3]) == size(M, 1)

        # Different sizes
        @test B * x[1:3] isa Vector{<:AbstractNode}
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
        @test B * M isa Matrix{<:AbstractNode}
        @test size(B * M) == (size(B, 1), size(M, 2))

        # Symbolic × Numeric
        @test M * B isa Matrix{<:AbstractNode}
        @test size(M * B) == (size(M, 1), size(B, 2))

        # Symbolic × Symbolic
        @test M * M isa Matrix{<:AbstractNode}
        @test size(M * M) == (size(M, 1), size(M, 2))

        # Different dimensions
        M2x3 = [X[i, j] for i in 1:2, j in 1:3]
        M3x4 = [X[i, j] for i in 1:3, j in 1:4]
        @test M2x3 * M3x4 isa Matrix{<:AbstractNode}
        @test size(M2x3 * M3x4) == (2, 4)
    end

    @testset "Dot products" begin
        println("  Testing dot products")

        # Symbolic · Symbolic
        @test dot(x, y) isa AbstractNode
        @test dot(x[1:3], y[1:3]) isa AbstractNode

        # Numeric · Symbolic
        @test dot(v, x) isa AbstractNode
        @test dot(v[1:3], x[1:3]) isa AbstractNode

        # Symbolic · Numeric
        @test dot(x, v) isa AbstractNode
        @test dot(x[1:3], v[1:3]) isa AbstractNode
    end

    @testset "Inner products via adjoint" begin
        println("  Testing inner products (x' * y)")

        # Symbolic' × Symbolic
        @test x' * y isa AbstractNode
        @test x' * x isa AbstractNode

        # Symbolic' × Numeric
        @test x' * v isa AbstractNode
        @test x[1:3]' * v[1:3] isa AbstractNode

        # Numeric' × Symbolic
        @test v' * x isa AbstractNode
        @test v[1:3]' * x[1:3] isa AbstractNode
    end

    @testset "Quadratic forms" begin
        println("  Testing quadratic forms")

        # x' * A * x
        @test x' * A * x isa AbstractNode
        @test x[1:3]' * M * x[1:3] isa AbstractNode
        @test x[1:3]' * B * x[1:3] isa AbstractNode
    end

    @testset "Matrix transpose and adjoint" begin
        println("  Testing matrix transpose and adjoint")

        # Transpose
        @test M' isa Matrix{<:AbstractNode}
        @test size(M') == (size(M, 2), size(M, 1))
        @test transpose(M) isa Matrix{<:AbstractNode}
        @test size(transpose(M)) == (size(M, 2), size(M, 1))

        # Non-square matrix
        @test N' isa Matrix{<:AbstractNode}
        @test size(N') == (size(N, 2), size(N, 1))

        # Transpose should work in products
        @test M' * x[1:3] isa Vector{<:AbstractNode}
        @test length(M' * x[1:3]) == size(M, 2)
    end

    @testset "Vector norms" begin
        println("  Testing vector norms")

        # Default norm (L2)
        @test norm(x) isa AbstractNode
        @test norm(y) isa AbstractNode

        # norm_sqr
        @test norm_sqr(x) isa AbstractNode
        @test norm_sqr(x[1:3]) isa AbstractNode

        # Explicit L2 norm
        @test norm(x, 2) isa AbstractNode

        # L1 norm
        @test norm(x, 1) isa AbstractNode
        @test norm(x[1:3], 1) isa AbstractNode

        # Lp norms
        @test norm(x, 3) isa AbstractNode
        @test norm(x, 4) isa AbstractNode
    end

    @testset "Matrix norms" begin
        println("  Testing matrix norms")

        # Frobenius norm (default)
        @test norm(M) isa AbstractNode
        @test norm(N) isa AbstractNode
    end

    @testset "Broadcasting - unary operations" begin
        println("  Testing broadcasting with unary operations")

        # Unary functions on vectors
        @test sin.(x) isa Vector{<:AbstractNode}
        @test length(sin.(x)) == length(x)
        @test cos.(x) isa Vector{<:AbstractNode}
        @test exp.(x) isa Vector{<:AbstractNode}
        @test log.(x) isa Vector{<:AbstractNode}
        @test sqrt.(x) isa Vector{<:AbstractNode}
        @test abs.(x) isa Vector{<:AbstractNode}

        # Unary functions on matrices
        @test sin.(M) isa Matrix{<:AbstractNode}
        @test size(sin.(M)) == size(M)
        @test cos.(M) isa Matrix{<:AbstractNode}
        @test exp.(M) isa Matrix{<:AbstractNode}
        @test log.(M) isa Matrix{<:AbstractNode}
    end

    @testset "Broadcasting - binary operations" begin
        println("  Testing broadcasting with binary operations")

        # Element-wise arithmetic on vectors
        @test x .+ y isa Vector{<:AbstractNode}
        @test length(x .+ y) == length(x)
        @test x .- y isa Vector{<:AbstractNode}
        @test x .* y isa Vector{<:AbstractNode}
        @test x ./ y isa Vector{<:AbstractNode}
        @test x .^ 2 isa Vector{<:AbstractNode}
        @test x .^ y isa Vector{<:AbstractNode}

        # Element-wise with scalars
        @test 2.0 .* x isa Vector{<:AbstractNode}
        @test x .+ 1.0 isa Vector{<:AbstractNode}
        @test x .- 3.0 isa Vector{<:AbstractNode}
        @test x ./ 2.0 isa Vector{<:AbstractNode}

        # Element-wise on matrices
        @test M .+ M isa Matrix{<:AbstractNode}
        @test size(M .+ M) == size(M)
        @test M .- M isa Matrix{<:AbstractNode}
        @test M .* M isa Matrix{<:AbstractNode}
        @test 2.0 .* M isa Matrix{<:AbstractNode}
        @test M ./ 2.0 isa Matrix{<:AbstractNode}
    end

    @testset "Broadcasting - compound expressions" begin
        println("  Testing broadcasting with compound expressions")

        # Compound vector expressions
        @test exp.(x) .+ 1 isa Vector{<:AbstractNode}
        @test sin.(x) .* cos.(y) isa Vector{<:AbstractNode}
        @test (x .+ y) ./ 2 isa Vector{<:AbstractNode}
        @test sqrt.(x.^2 .+ y.^2) isa Vector{<:AbstractNode}

        # Compound matrix expressions
        @test sin.(M) .+ cos.(M) isa Matrix{<:AbstractNode}
        @test (M .+ M) ./ 2 isa Matrix{<:AbstractNode}
        @test exp.(M) .* 2.0 isa Matrix{<:AbstractNode}
    end

    @testset "abs and abs2" begin
        println("  Testing abs and abs2")

        # abs2 for scalar node
        @test abs2(x[1]) isa AbstractNode

        # abs2 via broadcasting
        @test abs2.(x) isa Vector{<:AbstractNode}
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
        M2x3 = [X[i, j] for i in 1:2, j in 1:3]
        M4x2 = [X[i, j] for i in 1:4, j in 1:2]
        @test_throws AssertionError M2x3 * M4x2  # 3 != 4
    end

    @testset "Mixed operations" begin
        println("  Testing mixed operations")

        # Combine different operations
        result = A * x + v
        @test result isa Vector{<:AbstractNode}
        @test length(result) == length(v)

        # Matrix chain
        result = M * M * M
        @test result isa Matrix{<:AbstractNode}
        @test size(result) == size(M)

        # Complex expression
        result = (x' * A * x) + dot(x, v) + norm(x)^2
        @test result isa AbstractNode

        # Broadcasting mixed with products
        result = (A * x) .+ v
        @test result isa Vector{<:AbstractNode}

        # Norm of matrix-vector product
        result = norm(M * x[1:3])
        @test result isa AbstractNode
    end

    @testset "Type aliases" begin
        println("  Testing type aliases")

        @test x isa CTParser.SymbolicVector
        @test y isa CTParser.SymbolicVector
        @test M isa CTParser.SymbolicMatrix
        @test N isa CTParser.SymbolicMatrix

        @test x isa CTParser.SymbolicVecOrMat
        @test M isa CTParser.SymbolicVecOrMat
    end

    println("  All exa_linalg tests passed!")
end
