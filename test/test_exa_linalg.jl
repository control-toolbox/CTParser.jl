# test_exa_linalg.jl
# Pure unit tests for ExaModels linear algebra extensions
# No dependencies on CTParser - only ExaModels and LinearAlgebra

using .ExaLinAlg

# Import internal functions for testing purposes
using .ExaLinAlg: opt_add, opt_sub, opt_mul, opt_sum

# Helper to create test AbstractNode instances
function create_nodes()
    # Create ExaCore and symbolic variables using ExaModels API
    c = ExaModels.ExaCore()
    _var = ExaModels.variable(c, 10)
    x = _var[1]
    y = _var[2]
    z = _var[3]
    w = _var[4]
    return x, y, z, w
end

function test_exa_linalg()
    @testset "ExaModels Linear Algebra Tests" begin
        x, y, z, w = create_nodes()


        @testset "Scalar × Vector multiplication" begin
            v_num = [1.0, 2.0, 3.0]

            # AbstractNode scalar × numeric vector
            result1 = x * v_num
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            # Numeric scalar × AbstractNode vector
            vec_nodes = [x, y, z]
            result2 = 2.0 * vec_nodes
            @test length(result2) == 3
            @test result2 isa Vector
            @test result2[1] isa ExaModels.AbstractNode
        end

        @testset "Vector × Scalar multiplication" begin
            v_num = [1.0, 2.0, 3.0]
            vec_nodes = [x, y, z]

            # Vector × AbstractNode scalar
            result1 = v_num * x
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            # AbstractNode vector × numeric scalar (uses broadcasting)
            result2 = vec_nodes .* 2.5
            @test length(result2) == 3
            @test result2 isa Vector
            @test result2[1] isa ExaModels.AbstractNode
        end

        @testset "Scalar × Matrix multiplication" begin
            A_num = [1.0 2.0; 3.0 4.0]
            mat_nodes = [x y; z w]

            # AbstractNode scalar × numeric matrix
            result1 = x * A_num
            @test size(result1) == (2, 2)
            @test result1 isa Matrix
            @test result1[1, 1] isa ExaModels.AbstractNode

            # Numeric scalar × AbstractNode matrix
            result2 = 3.0 * mat_nodes
            @test size(result2) == (2, 2)
            @test result2 isa Matrix
            @test result2[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Matrix × Scalar multiplication" begin
            A_num = [1.0 2.0; 3.0 4.0]
            mat_nodes = [x y; z w]

            # Matrix × AbstractNode scalar
            result1 = A_num * x
            @test size(result1) == (2, 2)
            @test result1 isa Matrix
            @test result1[1, 1] isa ExaModels.AbstractNode

            # AbstractNode matrix × numeric scalar (uses broadcasting)
            result2 = mat_nodes .* 1.5
            @test size(result2) == (2, 2)
            @test result2 isa Matrix
            @test result2[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Dot product" begin
            v_num = [1.0, 2.0, 3.0]
            vec_nodes = [x, y, z]

            # Numeric vector · AbstractNode vector
            result1 = dot(v_num, vec_nodes)
            @test result1 isa ExaModels.AbstractNode

            # AbstractNode vector · numeric vector
            result2 = dot(vec_nodes, v_num)
            @test result2 isa ExaModels.AbstractNode
        end

        @testset "Matrix × Vector product" begin
            A_num = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3
            vec_nodes = [x, y, z]

            # Numeric matrix × AbstractNode vector
            result = A_num * vec_nodes
            @test length(result) == 2
            @test result isa Vector
            @test result[1] isa ExaModels.AbstractNode
        end

        @testset "Matrix × Matrix product" begin
            A_num = [1.0 2.0; 3.0 4.0]  # 2×2
            B_nodes = [x y; z w]         # 2×2 with AbstractNodes

            # Numeric matrix × AbstractNode matrix
            result1 = A_num * B_nodes
            @test size(result1) == (2, 2)
            @test result1 isa Matrix
            @test result1[1, 1] isa ExaModels.AbstractNode

            # AbstractNode matrix × numeric matrix
            result2 = B_nodes * A_num
            @test size(result2) == (2, 2)
            @test result2 isa Matrix
            @test result2[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Adjoint Vector × Matrix product" begin
            vec_nodes = [x, y, z]
            A_num = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2

            # Adjoint vector × matrix
            result = vec_nodes' * A_num
            @test size(result) == (1, 2)
            @test result isa LinearAlgebra.Adjoint
        end

        @testset "Matrix adjoint" begin
            mat_nodes = [x y z; y z x]  # 2×3

            result = adjoint(mat_nodes)
            @test size(result) == (3, 2)
            @test result isa Matrix
        end

        @testset "Determinant" begin
            # Test 1×1 determinant
            A1 = reshape([x], 1, 1)
            result1 = det(A1)
            @test result1 isa ExaModels.AbstractNode

            # Test 2×2 determinant
            A2 = [x y; z w]
            result2 = det(A2)
            @test result2 isa ExaModels.AbstractNode

            # Test 3×3 determinant
            A3 = [x y z; y z x; z x y]
            result3 = det(A3)
            @test result3 isa ExaModels.AbstractNode
        end

        @testset "Broadcasting operations" begin
            vec_nodes = [x, y, z]
            mat_nodes = [x y; z w]

            # Test broadcasting unary functions on vectors
            result1 = cos.(vec_nodes)
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            result2 = sin.(vec_nodes)
            @test length(result2) == 3
            @test result2[1] isa ExaModels.AbstractNode

            result3 = exp.(vec_nodes)
            @test length(result3) == 3
            @test result3[1] isa ExaModels.AbstractNode

            # Test broadcasting on matrices
            result4 = cos.(mat_nodes)
            @test size(result4) == (2, 2)
            @test result4 isa Matrix
            @test result4[1, 1] isa ExaModels.AbstractNode

            # Test broadcasting binary operations
            result5 = vec_nodes .+ 1.0
            @test length(result5) == 3
            @test result5[1] isa ExaModels.AbstractNode

            result6 = vec_nodes .* 2.0
            @test length(result6) == 3
            @test result6[1] isa ExaModels.AbstractNode

            # Test element-wise operations between arrays
            v_num = [1.0, 2.0, 3.0]
            result7 = vec_nodes .+ v_num
            @test length(result7) == 3
            @test result7[1] isa ExaModels.AbstractNode

            result8 = vec_nodes .* v_num
            @test length(result8) == 3
            @test result8[1] isa ExaModels.AbstractNode
        end

        @testset "Trace" begin
            # Test trace on square matrices
            A2 = [x y; z w]
            result1 = tr(A2)
            @test result1 isa ExaModels.AbstractNode

            A3 = [x y z; y z w; z w x]
            result2 = tr(A3)
            @test result2 isa ExaModels.AbstractNode

            # Test error on non-square matrix
            A_rect = [x y z; y z w]
            @test_throws AssertionError tr(A_rect)
        end

        @testset "Norms" begin
            vec_nodes = [x, y, z]
            mat_nodes = [x y; z w]

            # Test Euclidean norm (default 2-norm) for vectors
            result1 = norm(vec_nodes)
            @test result1 isa ExaModels.AbstractNode

            # Test 1-norm
            result2 = norm(vec_nodes, 1)
            @test result2 isa ExaModels.AbstractNode

            # Test 2-norm (explicit)
            result3 = norm(vec_nodes, 2)
            @test result3 isa ExaModels.AbstractNode

            # Test p-norm with p=3
            result4 = norm(vec_nodes, 3)
            @test result4 isa ExaModels.AbstractNode

            # Test Frobenius norm for matrices
            result5 = norm(mat_nodes)
            @test result5 isa ExaModels.AbstractNode

            # Test that infinity norm raises error
            @test_throws ErrorException norm(vec_nodes, Inf)
        end

        @testset "Array addition" begin
            vec_nodes1 = [x, y, z]
            vec_nodes2 = [y, z, w]
            v_num = [1.0, 2.0, 3.0]

            # Vector + Vector (AbstractNode + Number)
            result1 = vec_nodes1 + v_num
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            # Vector + Vector (Number + AbstractNode)
            result2 = v_num + vec_nodes1
            @test length(result2) == 3
            @test result2[1] isa ExaModels.AbstractNode

            # Vector + Vector (AbstractNode + AbstractNode)
            result3 = vec_nodes1 + vec_nodes2
            @test length(result3) == 3
            @test result3[1] isa ExaModels.AbstractNode

            # Matrix + Matrix
            mat_nodes1 = [x y; z w]
            mat_nodes2 = [y z; w x]
            A_num = [1.0 2.0; 3.0 4.0]

            result4 = mat_nodes1 + A_num
            @test size(result4) == (2, 2)
            @test result4 isa Matrix
            @test result4[1, 1] isa ExaModels.AbstractNode

            result5 = A_num + mat_nodes1
            @test size(result5) == (2, 2)
            @test result5[1, 1] isa ExaModels.AbstractNode

            result6 = mat_nodes1 + mat_nodes2
            @test size(result6) == (2, 2)
            @test result6[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Array subtraction" begin
            vec_nodes1 = [x, y, z]
            vec_nodes2 = [y, z, w]
            v_num = [1.0, 2.0, 3.0]

            # Vector - Vector (AbstractNode - Number)
            result1 = vec_nodes1 - v_num
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            # Vector - Vector (Number - AbstractNode)
            result2 = v_num - vec_nodes1
            @test length(result2) == 3
            @test result2[1] isa ExaModels.AbstractNode

            # Vector - Vector (AbstractNode - AbstractNode)
            result3 = vec_nodes1 - vec_nodes2
            @test length(result3) == 3
            @test result3[1] isa ExaModels.AbstractNode

            # Matrix - Matrix
            mat_nodes1 = [x y; z w]
            mat_nodes2 = [y z; w x]
            A_num = [1.0 2.0; 3.0 4.0]

            result4 = mat_nodes1 - A_num
            @test size(result4) == (2, 2)
            @test result4 isa Matrix
            @test result4[1, 1] isa ExaModels.AbstractNode

            result5 = A_num - mat_nodes1
            @test size(result5) == (2, 2)
            @test result5[1, 1] isa ExaModels.AbstractNode

            result6 = mat_nodes1 - mat_nodes2
            @test size(result6) == (2, 2)
            @test result6[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Diagonal operations" begin
            # Test diag: extract diagonal
            mat_nodes = [x y z; w x y; z w x]
            result1 = diag(mat_nodes)
            @test length(result1) == 3
            @test result1 isa Vector
            @test result1[1] isa ExaModels.AbstractNode

            # Test diag on rectangular matrix
            mat_rect = [x y z; w x y]
            result2 = diag(mat_rect)
            @test length(result2) == 2
            @test result2[1] isa ExaModels.AbstractNode

            # Test diagm: create diagonal matrix
            vec_nodes = [x, y, z]
            result3 = diagm(vec_nodes)
            @test size(result3) == (3, 3)
            @test result3 isa Matrix
            @test result3[1, 1] isa ExaModels.AbstractNode
            @test iszero(result3[1, 2])  # Off-diagonal elements are 0
            @test iszero(result3[2, 1])

            # Test diagm with offset
            result4 = diagm(1 => vec_nodes)
            @test size(result4) == (4, 4)
            @test result4[1, 2] isa ExaModels.AbstractNode
            @test iszero(result4[1, 1])  # Off-diagonal elements are 0

            result5 = diagm(-1 => vec_nodes)
            @test size(result5) == (4, 4)
            @test result5[2, 1] isa ExaModels.AbstractNode
            @test iszero(result5[1, 1])
        end

        @testset "Transpose operations" begin
            # Test transpose for scalar
            result1 = transpose(x)
            @test result1 isa ExaModels.AbstractNode

            # Test transpose for matrix
            mat_nodes = [x y z; w x y]
            result2 = transpose(mat_nodes)
            @test size(result2) == (3, 2)
            @test result2 isa Matrix
            @test result2[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Dimension mismatch errors" begin
            # Test dot product dimension mismatch
            v1 = [1.0, 2.0]
            vec_nodes = [x, y, z]
            @test_throws AssertionError dot(v1, vec_nodes)

            # Test matrix-vector dimension mismatch
            A = [1.0 2.0; 3.0 4.0]  # 2×2
            v = [x, y, z]  # length 3
            @test_throws AssertionError A * v

            # Test matrix-matrix dimension mismatch
            A1 = [1.0 2.0; 3.0 4.0]  # 2×2
            A2_nodes = [x y; z w; y x]  # 3×2
            @test_throws AssertionError A1 * A2_nodes

            # Test determinant on non-square matrix
            A_rect = [x y z; y z x]  # 2×3
            @test_throws AssertionError det(A_rect)

            # Test addition dimension mismatch
            v1 = [x, y]
            v2 = [x, y, z]
            @test_throws AssertionError v1 + v2

            # Test subtraction dimension mismatch
            @test_throws AssertionError v1 - v2

            # Test matrix addition dimension mismatch
            A1 = [x y; z w]
            A2 = [x y z; w x y]
            @test_throws AssertionError A1 + A2

            # Test matrix subtraction dimension mismatch
            @test_throws AssertionError A1 - A2
        end

        @testset "ExaCore variable arrays" begin
            # Create a more realistic test using ExaModels.variable
            c = ExaModels.ExaCore()
            xvar = ExaModels.variable(c, 2, 0:10, lvar=0, uvar=1)

            # Create vector from variable
            v = [xvar[i, 1] for i in 1:2]
            @test length(v) == 2
            @test v isa Vector
            @test v[1] isa ExaModels.AbstractNode

            # Create matrix from variable
            A = [xvar[i, j] for (i, j) ∈ Base.product(1:2, 0:3)]
            @test size(A) == (2, 4)
            @test A isa Matrix
            @test A[1, 1] isa ExaModels.AbstractNode

            # Test operations with these arrays
            # Vector operations
            v_num = [1.0, 2.0]
            result1 = v + v_num
            @test length(result1) == 2
            @test result1[1] isa ExaModels.AbstractNode

            result2 = v - v_num
            @test length(result2) == 2
            @test result2[1] isa ExaModels.AbstractNode

            result3 = 2.0 * v
            @test length(result3) == 2
            @test result3[1] isa ExaModels.AbstractNode

            result4 = dot(v, v_num)
            @test result4 isa ExaModels.AbstractNode

            result5 = norm(v)
            @test result5 isa ExaModels.AbstractNode

            # Matrix operations
            v2 = [xvar[i, 2] for i in 1:2]
            result6 = A * [1.0, 2.0, 3.0, 4.0]
            @test length(result6) == 2
            @test result6[1] isa ExaModels.AbstractNode

            # Extract a 2×2 submatrix for square operations
            A_square = [xvar[i, j] for (i, j) ∈ Base.product(1:2, 1:2)]
            @test size(A_square) == (2, 2)

            result7 = det(A_square)
            @test result7 isa ExaModels.AbstractNode

            result8 = tr(A_square)
            @test result8 isa ExaModels.AbstractNode

            result9 = diag(A_square)
            @test length(result9) == 2
            @test result9[1] isa ExaModels.AbstractNode

            # Matrix addition/subtraction
            A_num = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
            result10 = A + A_num
            @test size(result10) == (2, 4)
            @test result10[1, 1] isa ExaModels.AbstractNode

            result11 = A - A_num
            @test size(result11) == (2, 4)
            @test result11[1, 1] isa ExaModels.AbstractNode

            # Broadcasting operations
            result12 = cos.(v)
            @test length(result12) == 2
            @test result12[1] isa ExaModels.AbstractNode

            result13 = sin.(A_square)
            @test size(result13) == (2, 2)
            @test result13[1, 1] isa ExaModels.AbstractNode

            # Adjoint/transpose
            result14 = v'
            @test size(result14) == (1, 2)
            @test result14 isa LinearAlgebra.Adjoint

            result15 = transpose(A)
            @test size(result15) == (4, 2)
            @test result15 isa Matrix

            # diagm from variable array
            result16 = diagm(v)
            @test size(result16) == (2, 2)
            @test result16[1, 1] isa ExaModels.AbstractNode
            @test iszero(result16[1, 2])  # Off-diagonal is 0
        end

        @testset "Method ambiguity fixes" begin
            # Tests for all previously ambiguous cases (both operands are AbstractNode)
            # These tests verify that the ambiguity fixes work correctly

            x, y, z, w = create_nodes()
            vec_nodes1 = [x, y, z]
            vec_nodes2 = [y, z, w]
            mat_nodes1 = [x y; z w]
            mat_nodes2 = [y z; w x]

            @testset "Scalar × Vector (both AbstractNode)" begin
                # AbstractNode × Vector{AbstractNode}
                result = x * vec_nodes1
                @test length(result) == 3
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Vector × Scalar (both AbstractNode)" begin
                # Vector{AbstractNode} × AbstractNode
                result = vec_nodes1 * x
                @test length(result) == 3
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Scalar × Matrix (both AbstractNode)" begin
                # AbstractNode × Matrix{AbstractNode}
                result = x * mat_nodes1
                @test size(result) == (2, 2)
                @test result isa Matrix
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Matrix × Scalar (both AbstractNode)" begin
                # Matrix{AbstractNode} × AbstractNode
                result = mat_nodes1 * x
                @test size(result) == (2, 2)
                @test result isa Matrix
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            @testset "dot product (both AbstractNode)" begin
                # dot(Vector{AbstractNode}, Vector{AbstractNode})
                result = dot(vec_nodes1, vec_nodes2)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Matrix × Vector (both AbstractNode)" begin
                # Matrix{AbstractNode} × Vector{AbstractNode}
                # This is the specific case mentioned in the issue!
                A = [x y z; w x y]  # 2×3 matrix
                v = [x, y, z]        # length 3 vector

                result = A * v
                @test length(result) == 2
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Matrix × Matrix (both AbstractNode)" begin
                # Matrix{AbstractNode} × Matrix{AbstractNode}
                result = mat_nodes1 * mat_nodes2
                @test size(result) == (2, 2)
                @test result isa Matrix
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Adjoint Vector × Matrix (both AbstractNode)" begin
                # Adjoint{AbstractNode} × Matrix{AbstractNode}
                A = [x y; z w; y x]  # 3×2 matrix
                v = [x, y, z]         # length 3 vector

                result = v' * A
                @test size(result) == (1, 2)
                @test result isa LinearAlgebra.Adjoint
            end

            @testset "Vector + Vector (both AbstractNode)" begin
                # Vector{AbstractNode} + Vector{AbstractNode}
                result = vec_nodes1 + vec_nodes2
                @test length(result) == 3
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Vector - Vector (both AbstractNode)" begin
                # Vector{AbstractNode} - Vector{AbstractNode}
                result = vec_nodes1 - vec_nodes2
                @test length(result) == 3
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Matrix + Matrix (both AbstractNode)" begin
                # Matrix{AbstractNode} + Matrix{AbstractNode}
                result = mat_nodes1 + mat_nodes2
                @test size(result) == (2, 2)
                @test result isa Matrix
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Matrix - Matrix (both AbstractNode)" begin
                # Matrix{AbstractNode} - Matrix{AbstractNode}
                result = mat_nodes1 - mat_nodes2
                @test size(result) == (2, 2)
                @test result isa Matrix
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Mixed operations (no standard library conflicts)" begin
                # These verify we don't have ambiguities with standard library types
                v_num = [1.0, 2.0, 3.0]
                v_num_2 = [1.0, 2.0]
                A_num = [1.0 2.0 3.0; 4.0 5.0 6.0]

                # Should work without ambiguity
                @test (vec_nodes1 * 2.0) isa Vector
                @test (2.0 * vec_nodes1) isa Vector
                @test (mat_nodes1 * 2.0) isa Matrix
                @test (2.0 * mat_nodes1) isa Matrix
                @test dot(v_num, vec_nodes1) isa ExaModels.AbstractNode
                @test dot(vec_nodes1, v_num) isa ExaModels.AbstractNode
                @test (A_num * vec_nodes1) isa Vector
                @test (mat_nodes1 * v_num_2) isa Vector
                @test (A_num * [x y; z w; y x]) isa Matrix
                @test (mat_nodes1 * mat_nodes2) isa Matrix  # Both AbstractNode
            end
        end

        @testset "Detection functions and canonical nodes" begin
            # Test is_zero with Numbers
            @testset "is_zero with Numbers" begin
                @test iszero(0) == true
                @test iszero(0.0) == true
                @test iszero(1) == false
                @test iszero(1.0) == false
                @test iszero(-1) == false
            end

            # Test is_zero with symbolic AbstractNode
            @testset "is_zero with symbolic nodes" begin
                c = ExaModels.ExaCore()
                _var = ExaModels.variable(c, 10)
                x = _var[1]
                y = _var[2]
                expr = x + y  # Creates a Node2
                @test iszero(expr) == false
            end


        end

        @testset "Zero multiplication optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Scalar × Vector with zero scalar" begin
                # 0 × numeric vector - returns numeric zeros
                result1 = 0 * [1.0, 2.0, 3.0]
                @test all(iszero.(result1))
                @test result1 isa Vector{<:Real}  # opt_mul returns numeric 0
                @test length(result1) == 3

                # 0 (Number) × AbstractNode vector - returns numeric zeros
                vec_nodes = [x, y, z]
                result2 = 0 * vec_nodes
                @test all(iszero.(result2))
                @test result2 isa Vector{<:Real}  # opt_mul returns numeric 0

                # 0.0 × AbstractNode vector - returns numeric zeros
                result3 = 0.0 * vec_nodes
                @test all(iszero.(result3))
                @test result3 isa Vector{<:Real}  # opt_mul returns numeric 0
            end

            @testset "Scalar × Vector with zero elements" begin
                # AbstractNode × mixed vector
                result = x * [0, 1.0, 0, 2.0]
                @test iszero(result[1])
                @test !iszero(result[2])
                @test iszero(result[3])
                @test !iszero(result[4])
            end

            @testset "Vector × Scalar with zero scalar" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector × 0 - returns numeric zeros
                result1 = vec_nodes * 0
                @test all(iszero.(result1))
                @test result1 isa Vector{<:Real}  # opt_mul returns numeric 0

                # AbstractNode vector × 0.0 - returns numeric zeros
                result2 = vec_nodes * 0.0
                @test all(iszero.(result2))
                @test result2 isa Vector{<:Real}  # opt_mul returns numeric 0

                # Numeric vector × 0 - returns numeric zeros
                result3 = [1.0, 2.0, 3.0] * 0
                @test all(iszero.(result3))
                @test result3 isa Vector{<:Real}  # opt_mul returns numeric 0
            end

            @testset "Vector × Scalar with zero elements" begin
                # Mixed vector × AbstractNode
                result = [0, 1.0, 0, 2.0] * x
                @test iszero(result[1])
                @test !iszero(result[2])
                @test iszero(result[3])
                @test !iszero(result[4])
            end

            @testset "Scalar × Matrix with zero scalar" begin
                mat_nodes = [x y; z w]

                # 0 × AbstractNode matrix - returns numeric zeros
                result1 = 0 * mat_nodes
                @test all(iszero.(result1))
                @test result1 isa Matrix{<:Real}  # opt_mul returns numeric 0
                @test size(result1) == (2, 2)

                # 0 × numeric matrix - returns numeric zeros
                result2 = 0 * [1.0 2.0; 3.0 4.0]
                @test all(iszero.(result2))
                @test result2 isa Matrix{<:Real}  # opt_mul returns numeric 0
            end

            @testset "Matrix × Scalar with zero scalar" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix × 0.0 - returns numeric zeros
                result = mat_nodes * 0.0
                @test all(iszero.(result))
                @test result isa Matrix{<:Real}  # opt_mul returns numeric 0
            end
        end

        @testset "Zero addition optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Vector + Vector with zero elements" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector + zeros
                result1 = vec_nodes + [0, 0, 0]
                @test result1[1] === x  # Identity check
                @test result1[2] === y
                @test result1[3] === z
                @test result1 isa Vector{<:ExaModels.AbstractNode}

                # Zeros + AbstractNode vector
                result2 = [0.0, 0.0, 0.0] + vec_nodes
                @test result2[1] === x
                @test result2[2] === y
                @test result2[3] === z

                # Mixed: some zeros
                result3 = vec_nodes + [0, 1.0, 0]
                @test result3[1] === x  # x + 0 = x
                @test !iszero(result3[2])  # x + 1.0 creates node
                @test result3[3] === z  # z + 0 = z

                # Both AbstractNode, with zeros
                zero_vec = [0, 0, 0]
                result4 = vec_nodes + zero_vec
                @test result4[1] === x
                @test result4[2] === y
                @test result4[3] === z
            end

            @testset "Matrix + Matrix with zero elements" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix + zeros
                result1 = mat_nodes + [0 0; 0 0]
                @test result1[1,1] === x
                @test result1[1,2] === y
                @test result1[2,1] === z
                @test result1[2,2] === w
                @test result1 isa Matrix{<:ExaModels.AbstractNode}

                # Zeros + AbstractNode matrix
                result2 = [0.0 0.0; 0.0 0.0] + mat_nodes
                @test result2[1,1] === x
                @test result2[1,2] === y

                # Mixed: some zeros
                result3 = mat_nodes + [0 1.0; 0 0]
                @test result3[1,1] === x  # x + 0 = x
                @test !iszero(result3[1,2])  # y + 1.0 creates node
                @test result3[2,1] === z  # z + 0 = z
                @test result3[2,2] === w  # w + 0 = w
            end
        end

        @testset "Zero subtraction optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Vector - Vector with zero elements" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector - zeros
                result1 = vec_nodes - [0, 0, 0]
                @test result1[1] === x  # Identity check
                @test result1[2] === y
                @test result1[3] === z
                @test result1 isa Vector{<:ExaModels.AbstractNode}

                # Mixed: some zeros
                result2 = vec_nodes - [0, 1.0, 0]
                @test result2[1] === x  # x - 0 = x
                @test !iszero(result2[2])  # x - 1.0 creates node
                @test result2[3] === z  # z - 0 = z

                # Both AbstractNode, with zeros
                zero_vec = [0, 0, 0]
                result3 = vec_nodes - zero_vec
                @test result3[1] === x
                @test result3[2] === y
                @test result3[3] === z
            end

            @testset "Matrix - Matrix with zero elements" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix - zeros
                result1 = mat_nodes - [0 0; 0 0]
                @test result1[1,1] === x
                @test result1[1,2] === y
                @test result1[2,1] === z
                @test result1[2,2] === w
                @test result1 isa Matrix{<:ExaModels.AbstractNode}

                # Mixed: some zeros
                result2 = mat_nodes - [0 1.0; 0 0]
                @test result2[1,1] === x  # x - 0 = x
                @test !iszero(result2[1,2])  # y - 1.0 creates node
                @test result2[2,1] === z  # z - 0 = z
                @test result2[2,2] === w  # w - 0 = w
            end
        end

        @testset "Optimized scalar operations (opt_add, opt_sub, opt_mul)" begin
            x, y, z, w = create_nodes()

            @testset "opt_add rules" begin
                # 0 + x = x
                @test opt_add(0, x) === x
                @test opt_add(0.0, x) === x
                @test opt_add(0, x) === x

                # x + 0 = x
                @test opt_add(x, 0) === x
                @test opt_add(x, 0.0) === x
                @test opt_add(x, 0) === x

                # Non-zero + non-zero should create expression
                result = opt_add(x, y)
                @test result isa ExaModels.AbstractNode
            end

            @testset "opt_sub rules" begin
                # x - 0 = x
                @test opt_sub(x, 0) === x
                @test opt_sub(x, 0.0) === x
                @test opt_sub(x, 0) === x

                # 0 - x = -x (unary minus, Node1)
                result1 = opt_sub(0, x)
                @test result1 isa ExaModels.Node1
                result2 = opt_sub(0, x)
                @test result2 isa ExaModels.Node1

                # Non-zero - non-zero should create expression
                result = opt_sub(x, y)
                @test result isa ExaModels.AbstractNode
            end

            @testset "opt_mul rules" begin
                # 0 * y = 0
                @test opt_mul(0, y) === 0
                @test opt_mul(0.0, y) === 0
                @test opt_mul(0, y) === 0

                # y * 0 = 0
                @test opt_mul(y, 0) === 0
                @test opt_mul(y, 0.0) === 0
                @test opt_mul(y, 0) === 0

                # Note: Tests for 1 * y = y with symbolic y are removed
                # ExaModels handles this natively, and isone() doesn't work on AbstractNode

                # Non-one, non-zero multiplication creates expression
                result = opt_mul(2, y)
                @test result isa ExaModels.AbstractNode
                @test !iszero(result)
            end
        end

        @testset "Optimized sum (opt_sum)" begin
            x, y, z, w = create_nodes()

            @testset "opt_sum skips zeros" begin
                # Sum with all zeros should return 0
                result1 = opt_sum([0, 0, 0])
                @test result1 === 0

                result2 = opt_sum([0, 0])
                @test result2 === 0

                # Sum with mixed zeros and non-zeros
                result3 = opt_sum([0, x, 0])
                @test result3 === x  # Only x remains

                result4 = opt_sum([x, 0, y, 0])
                @test result4 isa ExaModels.AbstractNode  # x + y
            end

            @testset "opt_sum with single element" begin
                result1 = opt_sum([x])
                @test result1 === x

                result2 = opt_sum([0])
                @test result2 === 0
            end
        end

        @testset "sum function (uses opt_sum)" begin
            x, y, z, w = create_nodes()

            @testset "sum with zeros" begin
                # All AbstractNode zeros should return 0
                result1 = sum([0, 0, 0])
                @test result1 === 0

                # sum([0, 0, x, 0]) should return x
                result2 = sum([0, 0, x, 0])
                @test result2 === x

                # sum with multiple non-zeros interspersed with zeros
                result3 = sum([0, x, 0, y, 0])
                @test result3 isa ExaModels.AbstractNode
            end

            @testset "sum with single element" begin
                # Single AbstractNode
                result1 = sum([x])
                @test result1 === x

                # Single 0
                result2 = sum([0])
                @test result2 === 0
            end

            @testset "sum with all non-zeros" begin
                # sum of all non-zeros should equal sequential addition
                result1 = sum([x, y, z])
                @test result1 === x + y + z

                # sum with interspersed zeros
                result2 = sum([x, 0, y, 0, z])
                @test result2 === x + y + z  # Zeros should be skipped
            end

            @testset "sum on matrices" begin
                # sum should work on matrices too
                mat = [x 0; 0 y]
                result = sum(mat)
                # Should sum all elements: x + 0 + 0 + y = opt_add(x, y)
                @test result isa ExaModels.AbstractNode
            end
        end

        @testset "Optimized dot product" begin
            x, y, z, t = create_nodes()

            @testset "dot([1, 0, 1, 0], [x, y, z, t]) = x + z" begin
                # This is the key test from the plan!
                result = dot([1, 0, 1, 0], [x, y, z, t])

                # The result should be x + z (with tree optimizations)
                expected = opt_add(x, z)  # Which is x + z
                @test result == expected
            end

            @testset "dot with all zeros = 0" begin
                result = dot([0, 0, 0], [x, y, z])
                @test result === 0
            end

            @testset "dot with all ones" begin
                result = dot([1, 1, 1], [x, y, z])
                # Should be x + y + z
                @test result isa ExaModels.AbstractNode
                @test !iszero(result)
            end

            @testset "dot with single non-zero" begin
                result = dot([0, 1, 0], [x, y, z])
                @test result === y  # Only y survives
            end
        end

        @testset "Matrix operations with optimization" begin
            x, y, z, w = create_nodes()

            @testset "Identity matrix multiplication" begin
                # [1 0; 0 1] * [x, y] should give [x, y]
                I2 = [1 0; 0 1]
                vec = [x, y]
                result = I2 * vec

                @test result[1] === x
                @test result[2] === y
            end

            @testset "Zero matrix multiplication" begin
                # [0 0; 0 0] * [x, y] should give [0, 0]
                Z2 = [0 0; 0 0]
                vec = [x, y]
                result = Z2 * vec

                @test iszero(result[1])
                @test iszero(result[2])
            end

            @testset "Sparse matrix multiplication" begin
                # [1 0 0; 0 0 1] * [x, y, z] should give [x, z]
                A = [1 0 0; 0 0 1]
                vec = [x, y, z]
                result = A * vec

                @test result[1] === x
                @test result[2] === z
            end
        end
    end
end
