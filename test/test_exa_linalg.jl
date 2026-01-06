# test_exa_linalg.jl
# Pure unit tests for ExaModels linear algebra extensions
# No dependencies on CTParser - only ExaModels and LinearAlgebra

using ExaModels: ExaModels
using LinearAlgebra
using Test

# Import the ExaLinAlg module
include("exa_linalg.jl")
using .ExaLinAlg

# Helper to create test AbstractNode instances
function create_nodes()
    # Use Null nodes which are simple AbstractNode instances
    x = ExaModels.Null(1.0)
    y = ExaModels.Null(2.0)
    z = ExaModels.Null(3.0)
    w = ExaModels.Null(4.0)
    return x, y, z, w
end

function test_exa_linalg()
    @testset "ExaModels Linear Algebra Tests" begin
        x, y, z, w = create_nodes()

        @testset "Type conversions and promotions" begin
            # Test convert
            node = convert(ExaModels.AbstractNode, 5)
            @test node isa ExaModels.Null
            @test node isa ExaModels.AbstractNode

            # Test promote_rule
            arr = [x, 2.0, 3.0]
            @test eltype(arr) == ExaModels.AbstractNode
        end

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
            @test result3[1, 2] == 0.0
            @test result3[2, 1] == 0.0

            # Test diagm with offset
            result4 = diagm(1 => vec_nodes)
            @test size(result4) == (4, 4)
            @test result4[1, 2] isa ExaModels.AbstractNode
            @test result4[1, 1] == 0.0

            result5 = diagm(-1 => vec_nodes)
            @test size(result5) == (4, 4)
            @test result5[2, 1] isa ExaModels.AbstractNode
            @test result5[1, 1] == 0.0
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
            @test result16[1, 2] == 0.0
        end
    end
end
