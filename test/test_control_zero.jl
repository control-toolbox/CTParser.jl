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
        function get_model()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R², state
                ẋ(t) == [x₂(t), -x₁(t)]
                x₁(1)^2 → min
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
            ov = CTParser.@def begin
                v ∈ R, variable
                t ∈ [0, 1], time
                x ∈ R², state
                ẋ(t) == [x₂(t), -x₁(t)]
                x₁(1)^2 + v → min
            end
            Test.@test OCP.control_dimension(ov) == 0
            Test.@test OCP.variable_dimension(ov) == 1
            Test.@test OCP.state_dimension(ov) == 2
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