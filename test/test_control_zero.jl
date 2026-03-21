module TestControlZero

using Test: Test
import CTParser
import CTBase.Exceptions
import CTModels.OCP
import CTModels.Init

# for the @def and @init macros
import CTBase
import CTModels

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

function test_control_zero()
    Test.@testset "Control Zero Dimension Tests" verbose=VERBOSE showtiming=SHOWTIMING begin

        # Build a Model without control
        function get_model(; variable=false)
            if variable
                return CTParser.@def begin
                    v ∈ R, variable
                    t ∈ [0, 1], time
                    x ∈ R², state
                    ẋ(t) == [x₂(t), -x₁(t)]
                    x₁(1)^2 + v → min
                end
            else
                return CTParser.@def begin
                    t ∈ [0, 1], time
                    x ∈ R², state
                    ẋ(t) == [x₂(t), -x₁(t)]
                    x₁(1)^2 → min
                end
            end
        end

        # ====================================================================
        # UNIT TESTS - Building without control
        # ====================================================================

        Test.@testset "build() - Model without control" begin
            o = get_model()
            Test.@test o isa OCP.Model
            Test.@test OCP.control_dimension(o) == 0
            Test.@test OCP.control_name(o) == ""
            Test.@test OCP.control_components(o) == String[]
        end

        Test.@testset "build() - Model without control but with variable" begin
            # build a model with a variable
            ov = get_model(variable=true)
            Test.@test OCP.control_dimension(ov) == 0
            Test.@test OCP.variable_dimension(ov) == 1
            Test.@test OCP.state_dimension(ov) == 2
        end

        # ====================================================================
        # UNIT TESTS - Declaration Order Validation
        # ====================================================================

        Test.@testset "Control declaration order validation" begin
            # Control after dynamics should fail
            Test.@test_throws Exceptions.ParsingError begin
                CTParser.@def begin
                    t ∈ [0, 1], time
                    x ∈ R², state
                    ẋ(t) == [x₂(t), -x₁(t)]
                    u ∈ R, control  # ❌ After dynamics
                    x₁(1)^2 → min
                end
            end
            
            # Control after cost should fail
            Test.@test_throws Exceptions.ParsingError begin
                CTParser.@def begin
                    t ∈ [0, 1], time
                    x ∈ R², state
                    x₁(1)^2 → min
                    u ∈ R, control  # ❌ After cost
                end
            end
        end

        # ====================================================================
        # UNIT TESTS - Coordinate Dynamics Without Control
        # ====================================================================

        Test.@testset "Coordinate dynamics without control" begin
            o = CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R², state
                ∂(x₁)(t) == x₂(t)
                ∂(x₂)(t) == -x₁(t)
                x₁(1)^2 → min
            end
            Test.@test OCP.control_dimension(o) == 0
            Test.@test OCP.state_dimension(o) == 2
        end

        # ====================================================================
        # UNIT TESTS - Advanced Cost Criteria Without Control
        # ====================================================================

        Test.@testset "Advanced cost criteria without control" begin
            # Lagrange cost
            o1 = CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R², state
                ẋ(t) == [x₂(t), -x₁(t)]
                ∫(x₁(t)^2 + x₂(t)^2) → min
            end
            Test.@test OCP.control_dimension(o1) == 0
            
            # Bolza cost
            o2 = CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R², state
                ẋ(t) == [x₂(t), -x₁(t)]
                x₁(0)^2 + ∫(x₂(t)^2) → min
            end
            Test.@test OCP.control_dimension(o2) == 0
        end

        # ====================================================================
        # UNIT TESTS - Constraints Without Control
        # ====================================================================

        Test.@testset "Constraints without control" begin
            o = CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R², state
                ẋ(t) == [x₂(t), -x₁(t)]
                x₁(0) == 1
                x₂(0) == 0
                x₁(1) + x₂(1) ≤ 1
                x₁(1)^2 → min
            end
            Test.@test OCP.control_dimension(o) == 0
            Test.@test OCP.state_dimension(o) == 2
        end

        # ====================================================================
        # UNIT TESTS - Initialization without control
        # ====================================================================

        Test.@testset "Init - initial_control with scalar throws error" begin
            o = get_model()
            Test.@test_throws Exceptions.IncorrectArgument begin
                ig = CTParser.@init o begin 
                    u(t) := 0.5
                end
            end
        end

        Test.@testset "Init - initial_control with non-empty vector throws error" begin
            o = get_model()
            Test.@test_throws Exceptions.IncorrectArgument begin
                ig = CTParser.@init o begin 
                    u(t) := [0.5]
                end
            end
        end

        Test.@testset "Init - initial_guess without control" begin
            o = get_model()
            ig = CTParser.@init o begin end
            u_init = OCP.control(ig)
            Test.@test ig isa Init.InitialGuess
            Test.@test u_init isa Function
            Test.@test u_init(0.5) == Float64[]
        end

        Test.@testset "Advanced initialization without control" begin
            # Test with state initialization only
            o = get_model()
            ig = CTParser.@init o begin
                x(t) := [sin(t), cos(t)]
            end
            Test.@test ig isa Init.InitialGuess
            
            # Test with variable initialization
            o = get_model(; variable=true)
            ig2 = CTParser.@init o begin
                x(t) := [sin(t), cos(t)]
                v := 1.0
            end
            Test.@test ig2 isa Init.InitialGuess
            Test.@test OCP.variable(ig2) == 1.0
        end

        # ====================================================================
        # INTEGRATION TESTS - Serialization without control
        # ====================================================================

        Test.@testset "Serialization - Solution building without control" begin
            o = get_model()
            # Create a solution without control
            T = collect(range(0, 1, length=10))
            x_data = hcat(sin.(T), cos.(T))  # (10, 2) matrix
            u_data = Matrix{Float64}(undef, 10, 0)  # Empty control matrix (10×0)
            p_data = hcat(cos.(T), -sin.(T))  # (10, 2) matrix
            v_data = Float64[]

            sol = OCP.build_solution(
                o,
                T,
                T,
                T,
                T,
                x_data,
                u_data,
                v_data,
                p_data;
                objective=1.0,
                iterations=10,
                constraints_violation=0.0,
                message="Test solution",
                status=:success,
                successful=true,
            )

            # Test that control_dimension is 0
            Test.@test OCP.control_dimension(sol) == 0

            # Test that control function returns empty vector
            u_func = OCP.control(sol)
            Test.@test u_func(0.5) == Float64[]

            # Test that solution properties are correct
            Test.@test OCP.state_dimension(sol) == 2
            Test.@test OCP.objective(sol) == 1.0
        end

    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_control_zero() = TestControlZero.test_control_zero()