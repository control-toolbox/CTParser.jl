# test_exa_linalg.jl
# Pure unit tests for ExaModels linear algebra extensions
# No dependencies on CTParser - only ExaModels and LinearAlgebra

# Helper to check if a Null node represents zero
is_null_zero(x::ExaModels.Null) = iszero(x.value)
is_null_zero(x::ExaModels.AbstractNode) = false

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
            # Test convert with non-zero
            node = convert(ExaModels.AbstractNode, 5)
            @test node isa ExaModels.Null
            @test node isa ExaModels.AbstractNode
            @test node.value == 5

            # Test convert with zero (Int)
            zero_int = convert(ExaModels.AbstractNode, 0)
            @test zero_int isa ExaModels.Null
            @test iszero(zero_int.value)
            @test zero_int === zero(ExaModels.AbstractNode)  # Should be canonical zero

            # Test convert with zero (Float)
            zero_float = convert(ExaModels.AbstractNode, 0.0)
            @test zero_float isa ExaModels.Null
            @test iszero(zero_float.value)
            @test zero_float === zero(ExaModels.AbstractNode)  # Should be canonical zero

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
            @test result isa AbstractMatrix  # adjoint returns Adjoint wrapper, not plain Matrix
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
            @test_throws DimensionMismatch tr(A_rect)
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
            @test is_null_zero(result3[1, 2])  # Off-diagonal elements are zero(ExaModels.AbstractNode)
            @test is_null_zero(result3[2, 1])

            # Test diagm with offset
            result4 = diagm(1 => vec_nodes)
            @test size(result4) == (4, 4)
            @test result4[1, 2] isa ExaModels.AbstractNode
            @test is_null_zero(result4[1, 1])  # Off-diagonal elements are zero(ExaModels.AbstractNode)

            result5 = diagm(-1 => vec_nodes)
            @test size(result5) == (4, 4)
            @test result5[2, 1] isa ExaModels.AbstractNode
            @test is_null_zero(result5[1, 1])
        end

        @testset "Transpose operations" begin
            # Test transpose for scalar
            result1 = transpose(x)
            @test result1 isa ExaModels.AbstractNode

            # Test transpose for matrix
            mat_nodes = [x y z; w x y]
            result2 = transpose(mat_nodes)
            @test size(result2) == (3, 2)
            @test result2 isa AbstractMatrix  # transpose returns Transpose wrapper, not plain Matrix
            @test result2[1, 1] isa ExaModels.AbstractNode
        end

        @testset "Dimension mismatch errors" begin
            # Test dot product dimension mismatch
            v1 = [1.0, 2.0]
            vec_nodes = [x, y, z]
            @test_throws DimensionMismatch dot(v1, vec_nodes)

            # Test matrix-vector dimension mismatch
            A = [1.0 2.0; 3.0 4.0]  # 2×2
            v = [x, y, z]  # length 3
            @test_throws DimensionMismatch A * v

            # Test matrix-matrix dimension mismatch
            A1 = [1.0 2.0; 3.0 4.0]  # 2×2
            A2_nodes = [x y; z w; y x]  # 3×2
            @test_throws DimensionMismatch A1 * A2_nodes

            # Test determinant on non-square matrix
            A_rect = [x y z; y z x]  # 2×3
            @test_throws AssertionError det(A_rect)

            # Test addition dimension mismatch
            v1 = [x, y]
            v2 = [x, y, z]
            @test_throws DimensionMismatch v1 + v2

            # Test subtraction dimension mismatch
            @test_throws DimensionMismatch v1 - v2

            # Test matrix addition dimension mismatch
            A1 = [x y; z w]
            A2 = [x y z; w x y]
            @test_throws DimensionMismatch A1 + A2

            # Test matrix subtraction dimension mismatch
            @test_throws DimensionMismatch A1 - A2
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
            @test result15 isa AbstractMatrix  # transpose returns Transpose wrapper, not plain Matrix

            # diagm from variable array
            result16 = diagm(v)
            @test size(result16) == (2, 2)
            @test result16[1, 1] isa ExaModels.AbstractNode
            @test is_null_zero(result16[1, 2])  # Off-diagonal is zero(ExaModels.AbstractNode)
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

        @testset "Canonical nodes" begin
            # Test zero and one helpers
            @testset "zero and one helpers" begin
                z = zero(ExaModels.AbstractNode)
                @test z isa ExaModels.Null
                @test iszero(z.value)
                @test z.value == 0  # Canonical zero is Null(0)

                o = one(ExaModels.AbstractNode)
                @test o isa ExaModels.Null
                @test isone(o.value)
                @test o.value == 1  # Canonical one is Null(1)
            end

            @testset "zeros and ones array creation" begin
                # Test zeros with different dimensions
                z1 = zeros(ExaModels.AbstractNode, 3)
                @test length(z1) == 3
                @test z1 isa Vector{<:ExaModels.AbstractNode}
                @test all(is_null_zero.(z1))
                @test all(x -> x isa ExaModels.Null, z1)

                z2 = zeros(ExaModels.AbstractNode, 2, 3)
                @test size(z2) == (2, 3)
                @test z2 isa Matrix{<:ExaModels.AbstractNode}
                @test all(is_null_zero.(z2))

                z3 = zeros(ExaModels.AbstractNode, 2, 2, 2)
                @test size(z3) == (2, 2, 2)
                @test z3 isa Array{<:ExaModels.AbstractNode, 3}
                @test all(is_null_zero.(z3))

                # Test ones with different dimensions
                o1 = ones(ExaModels.AbstractNode, 3)
                @test length(o1) == 3
                @test o1 isa Vector{<:ExaModels.AbstractNode}
                @test all(x -> x isa ExaModels.Null && isone(x.value), o1)

                o2 = ones(ExaModels.AbstractNode, 2, 3)
                @test size(o2) == (2, 3)
                @test o2 isa Matrix{<:ExaModels.AbstractNode}
                @test all(x -> x isa ExaModels.Null && isone(x.value), o2)

                o3 = ones(ExaModels.AbstractNode, 2, 2, 2)
                @test size(o3) == (2, 2, 2)
                @test o3 isa Array{<:ExaModels.AbstractNode, 3}
                @test all(x -> x isa ExaModels.Null && isone(x.value), o3)

                # Test with specific ExaModels types (Variable <: AbstractNode)
                # Note: Variable is an alias/subtype, zeros/ones with AbstractNode works for all subtypes
                z_var = zeros(ExaModels.AbstractNode, 3)
                @test length(z_var) == 3
                @test eltype(z_var) <: ExaModels.AbstractNode
                @test all(is_null_zero.(z_var))

                o_var = ones(ExaModels.AbstractNode, 2, 2)
                @test size(o_var) == (2, 2)
                @test eltype(o_var) <: ExaModels.AbstractNode
                @test all(x -> x isa ExaModels.Null && isone(x.value), o_var)
            end

            @testset "zeros and ones in operations" begin
                x, y, z, w = create_nodes()

                # Test operations with zeros array
                z_vec = zeros(ExaModels.AbstractNode, 3)
                vec_nodes = [x, y, z]

                # Addition with zeros
                result1 = vec_nodes + z_vec
                @test result1[1].value == x.value
                @test result1[2].value == y.value
                @test result1[3].value == z.value

                # Multiplication with zeros (all zeros)
                result2 = [ExaModels.Null(2), ExaModels.Null(3), ExaModels.Null(4)] .* z_vec
                @test all(is_null_zero.(result2))

                # Test operations with ones array
                o_vec = ones(ExaModels.AbstractNode, 3)

                # Multiplication with ones
                result3 = vec_nodes .* o_vec
                @test result3[1].value == x.value
                @test result3[2].value == y.value
                @test result3[3].value == z.value

                # Matrix-vector with identity-like structure
                I_like = diagm(ones(ExaModels.AbstractNode, 2))
                vec2 = [x, y]
                result4 = I_like * vec2
                @test result4[1].value == x.value
                @test result4[2].value == y.value
            end
        end

        @testset "Zero multiplication optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Scalar × Vector with zero scalar" begin
                # Null(0) × numeric vector
                result1 = ExaModels.Null(0) * [1.0, 2.0, 3.0]
                @test all(is_null_zero.(result1))
                @test result1 isa Vector{<:ExaModels.AbstractNode}
                @test length(result1) == 3

                # 0 (Number) × AbstractNode vector
                vec_nodes = [x, y, z]
                result2 = 0 * vec_nodes
                @test all(is_null_zero.(result2))
                @test result2 isa Vector{<:ExaModels.AbstractNode}

                # 0.0 × AbstractNode vector
                result3 = 0.0 * vec_nodes
                @test all(is_null_zero.(result3))
            end

            @testset "Scalar × Vector with zero elements" begin
                # AbstractNode × mixed vector
                result = x * [0, 1.0, 0, 2.0]
                @test is_null_zero(result[1])
                @test !is_null_zero(result[2])
                @test is_null_zero(result[3])
                @test !is_null_zero(result[4])
            end

            @testset "Vector × Scalar with zero scalar" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector × 0
                result1 = vec_nodes * 0
                @test all(is_null_zero.(result1))
                @test result1 isa Vector{<:ExaModels.AbstractNode}

                # AbstractNode vector × 0.0
                result2 = vec_nodes * 0.0
                @test all(is_null_zero.(result2))

                # Numeric vector × Null(0)
                result3 = [1.0, 2.0, 3.0] * ExaModels.Null(0)
                @test all(is_null_zero.(result3))
            end

            @testset "Vector × Scalar with zero elements" begin
                # Mixed vector × AbstractNode
                result = [0, 1.0, ExaModels.Null(0), 2.0] * x
                @test is_null_zero(result[1])
                @test !is_null_zero(result[2])
                @test is_null_zero(result[3])
                @test !is_null_zero(result[4])
            end

            @testset "Scalar × Matrix with zero scalar" begin
                mat_nodes = [x y; z w]

                # 0 × AbstractNode matrix
                result1 = 0 * mat_nodes
                @test all(is_null_zero.(result1))
                @test result1 isa Matrix{<:ExaModels.AbstractNode}
                @test size(result1) == (2, 2)

                # Null(0) × numeric matrix
                result2 = ExaModels.Null(0) * [1.0 2.0; 3.0 4.0]
                @test all(is_null_zero.(result2))
            end

            @testset "Matrix × Scalar with zero scalar" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix × 0.0
                result = mat_nodes * 0.0
                @test all(is_null_zero.(result))
                @test result isa Matrix{<:ExaModels.AbstractNode}
            end
        end

        @testset "Zero addition optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Vector + Vector with zero elements" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector + zeros: Null(x) + Null(0) = Null(x + 0) = Null(x)
                result1 = vec_nodes + [0, 0, 0]
                @test result1[1].value == x.value  # x + 0 = x (same value)
                @test result1[2].value == y.value
                @test result1[3].value == z.value
                @test result1 isa Vector{<:ExaModels.AbstractNode}

                # Zeros + AbstractNode vector: Null(0) + Null(x) = Null(0 + x) = Null(x)
                result2 = [0.0, 0.0, 0.0] + vec_nodes
                @test result2[1].value == x.value
                @test result2[2].value == y.value
                @test result2[3].value == z.value

                # Mixed: some zeros
                result3 = vec_nodes + [0, 1.0, 0]
                @test result3[1].value == x.value  # x + 0 = x
                @test result3[2].value == y.value + 1.0  # y + 1.0
                @test result3[3].value == z.value  # z + 0 = z

                # Both AbstractNode, with zeros: Null(x) + Null(0) = Null(x + 0) = Null(x)
                zero_vec = [ExaModels.Null(0), ExaModels.Null(0), ExaModels.Null(0)]
                result4 = vec_nodes + zero_vec
                @test result4[1].value == x.value
                @test result4[2].value == y.value
                @test result4[3].value == z.value
            end

            @testset "Matrix + Matrix with zero elements" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix + zeros
                result1 = mat_nodes + [0 0; 0 0]
                @test result1[1,1].value == x.value
                @test result1[1,2].value == y.value
                @test result1[2,1].value == z.value
                @test result1[2,2].value == w.value
                @test result1 isa Matrix{<:ExaModels.AbstractNode}

                # Zeros + AbstractNode matrix
                result2 = [0.0 0.0; 0.0 0.0] + mat_nodes
                @test result2[1,1].value == x.value
                @test result2[1,2].value == y.value

                # Mixed: some zeros
                result3 = mat_nodes + [0 1.0; 0 0]
                @test result3[1,1].value == x.value  # x + 0 = x
                @test result3[1,2].value == y.value + 1.0  # y + 1.0
                @test result3[2,1].value == z.value  # z + 0 = z
                @test result3[2,2].value == w.value  # w + 0 = w
            end
        end

        @testset "Zero subtraction optimizations" begin
            x, y, z, w = create_nodes()

            @testset "Vector - Vector with zero elements" begin
                vec_nodes = [x, y, z]

                # AbstractNode vector - zeros: Null(x) - Null(0) = Null(x - 0) = Null(x)
                result1 = vec_nodes - [0, 0, 0]
                @test result1[1].value == x.value
                @test result1[2].value == y.value
                @test result1[3].value == z.value
                @test result1 isa Vector{<:ExaModels.AbstractNode}

                # Mixed: some zeros
                result2 = vec_nodes - [0, 1.0, 0]
                @test result2[1].value == x.value  # x - 0 = x
                @test result2[2].value == y.value - 1.0  # y - 1.0
                @test result2[3].value == z.value  # z - 0 = z

                # Both AbstractNode, with zeros: Null(x) - Null(0) = Null(x - 0) = Null(x)
                zero_vec = [ExaModels.Null(0), ExaModels.Null(0), ExaModels.Null(0)]
                result3 = vec_nodes - zero_vec
                @test result3[1].value == x.value
                @test result3[2].value == y.value
                @test result3[3].value == z.value
            end

            @testset "Matrix - Matrix with zero elements" begin
                mat_nodes = [x y; z w]

                # AbstractNode matrix - zeros
                result1 = mat_nodes - [0 0; 0 0]
                @test result1[1,1].value == x.value
                @test result1[1,2].value == y.value
                @test result1[2,1].value == z.value
                @test result1[2,2].value == w.value
                @test result1 isa Matrix{<:ExaModels.AbstractNode}

                # Mixed: some zeros
                result2 = mat_nodes - [0 1.0; 0 0]
                @test result2[1,1].value == x.value  # x - 0 = x
                @test result2[1,2].value == y.value - 1.0  # y - 1.0
                @test result2[2,1].value == z.value  # z - 0 = z
                @test result2[2,2].value == w.value  # w - 0 = w
            end
        end

        @testset "Scalar operations on Null nodes (+, -, *)" begin
            x, y, z, w = create_nodes()

            # Create a non-Null node for testing (Node2 from addition via ExaModels)
            e = x.value + y.value + 0.0  # Force ExaModels to create a non-Null expression
            e = ExaModels.Node2(+, x, y)  # Creates a Node2 directly
            f = ExaModels.Node2(+, z, w)  # Creates another Node2

            @testset "+ operator rules" begin
                # Null(x) + Null(y) = Null(x + y)
                result1 = ExaModels.Null(3) + ExaModels.Null(5)
                @test result1 isa ExaModels.Null
                @test result1.value == 8

                # Null(x) + e = x + e (unwraps Null, creates Node2)
                result2 = ExaModels.Null(3) + e
                @test result2 isa ExaModels.AbstractNode
                @test !(result2 isa ExaModels.Null)  # Should be Node2

                # e + Null(x) = e + x (unwraps Null, creates Node2)
                result3 = e + ExaModels.Null(3)
                @test result3 isa ExaModels.AbstractNode
                @test !(result3 isa ExaModels.Null)  # Should be Node2

                # e + f = e + f (native from ExaModels)
                result4 = e + f
                @test result4 isa ExaModels.AbstractNode
            end

            @testset "- operator rules" begin
                # Null(x) - Null(y) = Null(x - y)
                result1 = ExaModels.Null(5) - ExaModels.Null(3)
                @test result1 isa ExaModels.Null
                @test result1.value == 2

                # Null(0) - e = -e (unary minus)
                result2 = ExaModels.Null(0) - e
                @test result2 isa ExaModels.Node1  # Unary minus

                # Null(x) - e = x - e when !iszero(x)
                result3 = ExaModels.Null(3) - e
                @test result3 isa ExaModels.AbstractNode
                @test !(result3 isa ExaModels.Null)  # Should be Node2

                # e - Null(x) = e - x (unwraps Null)
                result4 = e - ExaModels.Null(3)
                @test result4 isa ExaModels.AbstractNode
                @test !(result4 isa ExaModels.Null)  # Should be Node2

                # e - f = e - f (native from ExaModels)
                result5 = e - f
                @test result5 isa ExaModels.AbstractNode
            end

            @testset "* operator rules" begin
                # Null(x) * Null(y) = Null(x * y)
                result1 = ExaModels.Null(3) * ExaModels.Null(5)
                @test result1 isa ExaModels.Null
                @test result1.value == 15

                # Null(0) * e = Null(0) (zero optimization)
                result2 = ExaModels.Null(0) * e
                @test result2 isa ExaModels.Null
                @test iszero(result2.value)

                # e * Null(0) = Null(0) (zero optimization)
                result3 = e * ExaModels.Null(0)
                @test result3 isa ExaModels.Null
                @test iszero(result3.value)

                # Null(x) * e = x * e when !iszero(x) (unwraps Null)
                result4 = ExaModels.Null(3) * e
                @test result4 isa ExaModels.AbstractNode
                @test !(result4 isa ExaModels.Null)  # Should be Node2

                # e * Null(x) = e * x when !iszero(x) (unwraps Null)
                result5 = e * ExaModels.Null(3)
                @test result5 isa ExaModels.AbstractNode
                @test !(result5 isa ExaModels.Null)  # Should be Node2

                # e * f = e * f (native from ExaModels)
                result6 = e * f
                @test result6 isa ExaModels.AbstractNode
            end
        end

        @testset "sum function" begin
            x, y, z, w = create_nodes()

            @testset "sum with zeros" begin
                # All AbstractNode zeros: 0 + 0 + 0 = 0
                result1 = sum([zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode)])
                @test result1 isa ExaModels.Null
                @test iszero(result1.value)

                # sum([zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode), x, zero(ExaModels.AbstractNode)]): 0 + 0 + x + 0 = x
                result2 = sum([zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode), x, zero(ExaModels.AbstractNode)])
                @test result2 isa ExaModels.Null
                @test result2.value == x.value

                # sum with multiple non-zeros interspersed with zeros
                result3 = sum([zero(ExaModels.AbstractNode), x, zero(ExaModels.AbstractNode), y, zero(ExaModels.AbstractNode)])
                @test result3 isa ExaModels.Null
                @test result3.value == x.value + y.value
            end

            @testset "sum with single element" begin
                # Single AbstractNode: 0 + x = x
                result1 = sum([x])
                @test result1 isa ExaModels.Null
                @test result1.value == x.value

                # Single zero(ExaModels.AbstractNode): 0 + 0 = 0
                result2 = sum([zero(ExaModels.AbstractNode)])
                @test result2 isa ExaModels.Null
                @test iszero(result2.value)
            end

            @testset "sum with all non-zeros" begin
                # sum of all non-zeros
                result1 = sum([x, y, z])
                @test result1 isa ExaModels.Null
                @test result1.value == x.value + y.value + z.value
            end

            @testset "sum on matrices" begin
                # sum should work on matrices too
                mat = [x zero(ExaModels.AbstractNode); zero(ExaModels.AbstractNode) y]
                result = sum(mat)
                # Should sum all elements: x + 0 + 0 + y
                @test result isa ExaModels.Null
                @test result.value == x.value + y.value
            end
        end

        @testset "Optimized dot product" begin
            x, y, z, t = create_nodes()

            @testset "dot([1, 0, 1, 0], [x, y, z, t]) = x + z" begin
                # dot = 1*x + 0*y + 1*z + 0*t = x + z
                result = dot([1, 0, 1, 0], [x, y, z, t])
                @test result isa ExaModels.Null
                @test result.value == x.value + z.value
            end

            @testset "dot with all zeros" begin
                # dot = 0*x + 0*y + 0*z = 0
                result = dot([0, 0, 0], [x, y, z])
                @test result isa ExaModels.Null
                @test iszero(result.value)
            end

            @testset "dot with all ones" begin
                # dot = 1*x + 1*y + 1*z = x + y + z
                result = dot([1, 1, 1], [x, y, z])
                @test result isa ExaModels.Null
                @test result.value == x.value + y.value + z.value
            end

            @testset "dot with single non-zero" begin
                # dot = 0*x + 1*y + 0*z = y
                result = dot([0, 1, 0], [x, y, z])
                @test result isa ExaModels.Null
                @test result.value == y.value
            end
        end

        @testset "Matrix operations with optimization" begin
            x, y, z, w = create_nodes()

            @testset "Identity matrix multiplication" begin
                # [1 0; 0 1] * [x, y] = [1*x + 0*y, 0*x + 1*y] = [x, y]
                I2 = [1 0; 0 1]
                vec = [x, y]
                result = I2 * vec

                @test result[1] isa ExaModels.Null
                @test result[1].value == x.value
                @test result[2] isa ExaModels.Null
                @test result[2].value == y.value
            end

            @testset "Zero matrix multiplication" begin
                # [0 0; 0 0] * [x, y] = [0, 0]
                Z2 = [0 0; 0 0]
                vec = [x, y]
                result = Z2 * vec

                @test result[1] isa ExaModels.Null
                @test iszero(result[1].value)
                @test result[2] isa ExaModels.Null
                @test iszero(result[2].value)
            end

            @testset "Sparse matrix multiplication" begin
                # [1 0 0; 0 0 1] * [x, y, z] = [x, z]
                A = [1 0 0; 0 0 1]
                vec = [x, y, z]
                result = A * vec

                @test result[1] isa ExaModels.Null
                @test result[1].value == x.value
                @test result[2] isa ExaModels.Null
                @test result[2].value == z.value
            end
        end

        @testset "Iterate protocol for scalar behavior" begin
            x, y, z, w = create_nodes()

            @testset "Basic iterate protocol" begin
                # Test first iteration
                result = iterate(x)
                @test result !== nothing
                @test result[1] === x
                @test result[2] === nothing

                # Test second iteration (should be done)
                result2 = iterate(x, nothing)
                @test result2 === nothing

                # Test collect
                collected = collect(x)
                @test length(collected) == 1
                @test collected[1] === x
            end

            @testset "Splatting behavior" begin
                # Splatting a scalar should pass it as single argument
                f(a) = a * ExaModels.Null(2)
                @test f(x...) isa ExaModels.AbstractNode
            end
        end

        @testset "SubArray (view) support" begin
            vec_nodes = [x, y, z, w]
            v_num = [1.0, 2.0, 3.0, 4.0]

            @testset "dot with SubArray" begin
                # View of AbstractNode vector with numeric vector
                view_nodes = view(vec_nodes, 1:2)
                @test view_nodes isa SubArray
                result = dot([1.0, 2.0], view_nodes)
                @test result isa ExaModels.AbstractNode

                # View of numeric vector with AbstractNode vector
                view_num = view(v_num, 2:3)
                result2 = dot(view_num, [y, z])
                @test result2 isa ExaModels.AbstractNode

                # Both views
                view_nodes2 = view(vec_nodes, 3:4)
                view_num2 = view(v_num, 3:4)
                result3 = dot(view_num2, view_nodes2)
                @test result3 isa ExaModels.AbstractNode
            end

            # Matrix-vector with SubArray removed - not supported with current implementation
            # (would require AbstractMatrix/AbstractVector which creates ambiguities)

            @testset "Vector addition with SubArray" begin
                view_nodes = view(vec_nodes, 1:3)
                result = view_nodes + [1.0, 2.0, 3.0]
                @test length(result) == 3
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Scalar multiplication with SubArray" begin
                view_nodes = view(vec_nodes, 2:4)
                result = 2.0 * view_nodes
                @test length(result) == 3
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "norm with SubArray" begin
                view_nodes = view(vec_nodes, 1:3)
                result = norm(view_nodes)
                @test result isa ExaModels.AbstractNode
            end
        end

        @testset "ReshapedArray support" begin
            # Create a vector and reshape to matrix
            vec_nodes = [x, y, z, w]
            mat_reshaped = reshape(vec_nodes, 2, 2)

            @testset "Reshape type check" begin
                @test mat_reshaped isa AbstractMatrix  # reshape may return Matrix or ReshapedArray
                @test size(mat_reshaped) == (2, 2)
            end

            # Matrix-vector with ReshapedArray removed - not supported with current implementation
            # (would require AbstractMatrix/AbstractVector which creates ambiguities)

            @testset "det with ReshapedArray" begin
                result = det(mat_reshaped)
                @test result isa ExaModels.AbstractNode
            end

            @testset "tr with ReshapedArray" begin
                result = tr(mat_reshaped)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Matrix multiplication with ReshapedArray" begin
                A_num = [1.0 2.0; 3.0 4.0]
                result = A_num * mat_reshaped
                @test size(result) == (2, 2)
                @test result[1, 1] isa ExaModels.AbstractNode
            end

            # Reshape numeric to multiply with nodes removed - matrix-vector not supported
            # with ReshapedArray (would require AbstractMatrix which creates ambiguities)

            @testset "norm with ReshapedArray" begin
                result = norm(mat_reshaped)
                @test result isa ExaModels.AbstractNode
            end
        end

        @testset "ReinterpretArray support" begin
            @testset "Reinterpret Complex to Real (numeric)" begin
                # Create complex vector and reinterpret to real
                c = [1.0 + 2.0im, 3.0 + 4.0im]
                r = reinterpret(Float64, c)
                @test r isa Base.ReinterpretArray
                @test length(r) == 4  # 2 complex numbers = 4 reals

                # dot with reinterpreted real and AbstractNode
                vec_nodes = [x, y, z, w]
                result = dot(r, vec_nodes)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Reinterpret with different element sizes" begin
                # Int64 to Int32 (works on little-endian)
                if Base.ENDIAN_BOM == 0x04030201  # Little-endian
                    i64 = Int64[1, 2]
                    i32 = reinterpret(Int32, i64)
                    @test i32 isa Base.ReinterpretArray

                    # Create AbstractNode vector of matching length
                    nodes = [ExaModels.Null(Float64(i)) for i in 1:length(i32)]
                    # Convert Int32 to Float64 for dot product
                    result = dot(Float64.(collect(i32)), nodes)
                    @test result isa ExaModels.AbstractNode
                end
            end

            # Scalar multiplication with ReinterpretArray removed - not supported
            # (would require AbstractVector which creates ambiguities)
        end

        @testset "Combined wrapper types" begin
            # Test combinations of view, reshape, etc.
            vec_nodes = [x, y, z, w, x, y]  # 6 elements

            @testset "View then reshape" begin
                v = view(vec_nodes, 1:4)
                m = reshape(v, 2, 2)
                @test m isa Base.ReshapedArray

                result = det(m)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Reshape then view" begin
                mat = reshape(vec_nodes, 2, 3)
                v = view(mat, :, 1)  # First column
                @test v isa SubArray
                @test length(v) == 2

                result = dot([1.0, 2.0], v)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Complex operations with views" begin
                A = reshape(vec_nodes, 2, 3)
                v1 = view(A, :, 1)  # Column 1
                v2 = view(A, :, 2)  # Column 2

                # Both are SubArrays
                result = dot(v1, v2)
                @test result isa ExaModels.AbstractNode

                # Addition
                result2 = v1 + v2
                @test length(result2) == 2
                @test result2[1] isa ExaModels.AbstractNode
            end
        end

        @testset "Compatibility with existing specialized methods" begin
            # Ensure our iterate doesn't break the optimized Vector/Matrix paths
            vec_nodes = [x, y, z]
            mat_nodes = [x y; z w]
            v_num = [1.0, 2.0, 3.0]
            A_num = [1.0 2.0; 3.0 4.0]

            @testset "Vector methods still work" begin
                # These should use the specialized ExaLinAlg methods
                result1 = dot(v_num, vec_nodes)
                @test result1 isa ExaModels.AbstractNode

                result2 = v_num + vec_nodes
                @test result2 isa Vector
                @test length(result2) == 3

                result3 = 2.0 * vec_nodes
                @test result3 isa Vector
            end

            @testset "Matrix methods still work" begin
                result1 = A_num * [x, y]
                @test result1 isa Vector
                @test length(result1) == 2

                result2 = det(mat_nodes)
                @test result2 isa ExaModels.AbstractNode

                result3 = tr(mat_nodes)
                @test result3 isa ExaModels.AbstractNode
            end
        end

        @testset "Edge cases with wrapper types" begin
            @testset "Empty views" begin
                vec_nodes = [x, y, z]
                empty_view = view(vec_nodes, 1:0)
                @test length(empty_view) == 0

                # dot with empty arrays should work
                result = dot(Float64[], empty_view)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Single element views" begin
                vec_nodes = [x, y, z]
                single_view = view(vec_nodes, 2:2)
                @test length(single_view) == 1

                result = dot([1.0], single_view)
                @test result isa ExaModels.AbstractNode
            end

            @testset "Strided views" begin
                vec_nodes = [x, y, z, w]
                strided_view = view(vec_nodes, 1:2:4)  # Elements 1 and 3
                @test length(strided_view) == 2

                result = dot([1.0, 2.0], strided_view)
                @test result isa ExaModels.AbstractNode
            end
        end
    end
end
