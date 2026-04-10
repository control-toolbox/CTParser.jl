module TestSlices

using Test: Test
import CTParser
import CTBase.Exceptions
import CTModels.OCP

# for the @def macro
import CTBase
import CTModels
import ExaModels
import MadNLP

include(joinpath(@__DIR__, "utils.jl"))

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

function test_slices()
    Test.@testset "Slice with end Tests" verbose=VERBOSE showtiming=SHOWTIMING begin

        # ====================================================================
        # MODEL BUILDERS - Slices in constraints
        # ====================================================================
        # These models test slice syntax in initial conditions: x[...](0) == x0
        # - o1: x[1:3](0)   → literal integer range (works)
        # - o2: x[1:end](0) → end keyword (broken - not yet supported)
        # - o3: x[1:n](0)   → variable n (broken - not yet supported)

        function get_model_o1()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_o2()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:end](0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_o3()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:n](0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        # ====================================================================
        # UNIT TESTS - Slices in constraints
        # ====================================================================

        Test.@testset "build() - Model with x[1:3](0) == x0 constraint" begin
            o1 = get_model_o1()
            Test.@test o1 isa OCP.Model
            Test.@test OCP.state_dimension(o1) == 3
            Test.@test OCP.control_dimension(o1) == 1
        end

        Test.@testset "build() - Model with x[1:end](0) == x0 constraint (broken)" begin
            Test.@test_broken get_model_o2() isa OCP.Model
        end

        Test.@testset "build() - Model with x[1:n](0) == x0 constraint (broken)" begin
            Test.@test_broken get_model_o3() isa OCP.Model
        end

        # ====================================================================
        # INTEGRATION TESTS - Slices in constraints (discretisation + solve)
        # ====================================================================

        Test.@testset "discretise_exa with x[1:3](0) == x0" begin
            o1 = get_model_o1()
            m1 = discretise_exa(o1)
            Test.@test m1 isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with x[1:end](0) == x0 (broken)" begin
            Test.@test_broken discretise_exa(get_model_o2()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with x[1:n](0) == x0 (broken)" begin
            Test.@test_broken discretise_exa(get_model_o3()) isa ExaModels.ExaModel
        end

        Test.@testset "solve with x[1:3](0) == x0" begin
            o1 = get_model_o1()
            m1 = discretise_exa(o1)
            sol1 = MadNLP.madnlp(m1; tol=1e-6, print_level=MadNLP.ERROR)
            Test.@test sol1.status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with x[1:end](0) == x0 (broken)" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_o2()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with x[1:n](0) == x0 (broken)" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_o3()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # MODEL BUILDERS - Slices in dynamics
        # ====================================================================
        # These models test slice syntax in dynamics: ∂(x[...])(t) == ...
        # - d1: ∂(x[1:3])(t)   → literal integer range (works for parsing, broken for discretisation)
        # - d2: ∂(x[1:end])(t) → end keyword (broken - not yet supported)
        # - d3: ∂(x[1:n])(t)   → variable n (works for parsing, broken for discretisation)

        function get_model_d1()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ∂(x[1:3])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_d2()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ∂(x[1:end])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_d3()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ∂(x[1:n])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        # ====================================================================
        # UNIT TESTS - Slices in dynamics
        # ====================================================================

        Test.@testset "build() - Model with ∂(x[1:3])(t) dynamics" begin
            o = get_model_d1()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.control_dimension(o) == 1
        end

        Test.@testset "build() - Model with ∂(x[1:end])(t) dynamics (broken)" begin
            Test.@test_broken get_model_d2() isa OCP.Model
        end

        Test.@testset "build() - Model with ∂(x[1:n])(t) dynamics (broken)" begin
            o = get_model_d3()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.control_dimension(o) == 1
        end

        # ====================================================================
        # INTEGRATION TESTS - Slices in dynamics (discretisation + solve)
        # ====================================================================

        Test.@testset "discretise_exa with ∂(x[1:3])(t) dynamics" begin
            Test.@test_broken discretise_exa(get_model_d1()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with ∂(x[1:end])(t) dynamics (broken)" begin
            Test.@test_broken discretise_exa(get_model_d2()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with ∂(x[1:n])(t) dynamics (broken)" begin
            Test.@test_broken discretise_exa(get_model_d3()) isa ExaModels.ExaModel
        end

        Test.@testset "solve with ∂(x[1:3])(t) dynamics" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_d1()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with ∂(x[1:end])(t) dynamics (broken)" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_d2()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with ∂(x[1:n])(t) dynamics (broken)" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_d3()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # MODEL BUILDERS - Slices in Mayer cost
        # ====================================================================
        # These models test slice syntax in Mayer cost: x[...](tf)
        # - m1: x[1:3](tf)   → literal integer range (works for parsing, broken for discretisation)
        # - m2: x[1:end](tf) → end keyword (works for parsing, broken for discretisation)
        # - m3: x[1:n](tf)   → variable n (works for parsing, broken for discretisation)
        function get_model_m1()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x)(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:3](tf).^2) + ∫(u(t)^2) → min
            end
        end

        function get_model_m2()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x)(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:end](tf).^2) + ∫(u(t)^2) → min
            end
        end

        function get_model_m3()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x)(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:n](tf).^2) + ∫(u(t)^2) → min
            end
        end

        # ====================================================================
        # UNIT TESTS - Slices in Mayer cost
        # ====================================================================

        Test.@testset "build() - Model with x[1:3](tf) in Mayer cost" begin
            o = get_model_m1()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.control_dimension(o) == 1
        end

        Test.@testset "build() - Model with x[1:end](tf) in Mayer cost" begin
            o = get_model_m2()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.control_dimension(o) == 1
        end

        Test.@testset "build() - Model with x[1:n](tf) in Mayer cost" begin
            o = get_model_m3()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.control_dimension(o) == 1
        end

        # ====================================================================
        # INTEGRATION TESTS - Slices in Mayer cost (discretisation + solve)
        # ====================================================================

        Test.@testset "discretise_exa with x[1:3](tf) in Mayer cost" begin
            Test.@test_broken discretise_exa(get_model_m1()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with x[1:end](tf) in Mayer cost" begin
            Test.@test_broken discretise_exa(get_model_m2()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa with x[1:n](tf) in Mayer cost" begin
            Test.@test_broken discretise_exa(get_model_m3()) isa ExaModels.ExaModel
        end

        Test.@testset "solve with x[1:3](tf) in Mayer cost" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_m1()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with x[1:end](tf) in Mayer cost" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_m2()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve with x[1:n](tf) in Mayer cost" begin
            Test.@test_broken MadNLP.madnlp(discretise_exa(get_model_m3()); tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_slices() = TestSlices.test_slices()