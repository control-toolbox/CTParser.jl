module TestSlices

using Test: Test
import CTParser
import CTBase.Exceptions
import CTModels.OCP
import CTBase
import CTModels
import ExaModels
import MadNLP

include(joinpath(@__DIR__, "utils.jl"))

# Must run before any @def is first compiled so the exa closure is attached.
CTParser.activate_backend(:exa)

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

# ---------------------------------------------------------------------------
# Introspection helpers (functional backend)
# ---------------------------------------------------------------------------
function _oop(f!, n; T=Float64)
    isnothing(f!) && return nothing
    function f(args...)
        r = zeros(T, n)
        f!(r, args...)
        return r
    end
    return f
end
_dyn(ocp) = _oop(CTModels.dynamics(ocp), CTModels.state_dimension(ocp))

# Sample evaluation point used throughout
const _T, _X, _U = 0.5, [1.0, 2.0, 3.0], [4.0]

function test_slices()
    Test.@testset "Slice/Index Notation Tests" verbose=VERBOSE showtiming=SHOWTIMING begin

        # ====================================================================
        # SECTION 1 – SCALAR INDEX NOTATION  x[i], xᵢ, xi
        # ====================================================================

        function get_model_scalar_bracket_ic()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1](0) == 1.0
                x[2](0) == 0.0
                x[3](0) == 0.0
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_scalar_subscript_ic()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x₁(0) == 1.0
                x₂(0) == 0.0
                x₃(0) == 0.0
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_scalar_ascii_ic()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x1(0) == 1.0
                x2(0) == 0.0
                x3(0) == 0.0
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_scalar_bracket_fc()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1](1) == 0.0
                x[2](1) == 0.0
                ∫(u(t)^2) → min
            end
        end

        function get_model_scalar_coord_dyn()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x[1])(t) == x[2](t)
                ∂(x[2])(t) == x[3](t)
                ∂(x[3])(t) == u(t)
                ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – scalar bracket x[i] in initial constraint" begin
            o = get_model_scalar_bracket_ic()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
        end

        Test.@testset "build – scalar subscript xᵢ in initial constraint" begin
            o = get_model_scalar_subscript_ic()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
        end

        Test.@testset "build – scalar ascii xi in initial constraint" begin
            o = get_model_scalar_ascii_ic()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
        end

        Test.@testset "build – scalar x[i] in final constraint" begin
            Test.@test get_model_scalar_bracket_fc() isa OCP.Model
        end

        Test.@testset "build – coordinate dynamics ∂(x[i])(t)" begin
            o = get_model_scalar_coord_dyn()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test _dyn(o)(_T, _X, _U, nothing) == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "discretise_exa – scalar x[i] initial constraint" begin
            Test.@test discretise_exa(get_model_scalar_bracket_ic()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – scalar x[i] final constraint" begin
            Test.@test discretise_exa(get_model_scalar_bracket_fc()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – coordinate dynamics ∂(x[i])(t)" begin
            Test.@test discretise_exa(get_model_scalar_coord_dyn()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – scalar x[i] initial constraint" begin
            m = discretise_exa(get_model_scalar_bracket_ic())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – coordinate dynamics ∂(x[i])(t)" begin
            m = discretise_exa(get_model_scalar_coord_dyn())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 2 – INITIAL CONSTRAINT RANGE SLICES  x[a:b](0) == c
        # Labels are forbidden by the exa parser for sliced constraints.
        # ====================================================================

        function get_model_ic_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_ic_offset()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[2:3](0) == [0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_ic_step()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:2:3](0) == [1, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_ic_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:end](0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_ic_var()
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

        Test.@testset "build – x[1:3](0)" begin
            Test.@test get_model_ic_1_3() isa OCP.Model
        end

        Test.@testset "build – x[2:3](0) offset" begin
            Test.@test get_model_ic_offset() isa OCP.Model
        end

        Test.@testset "build – x[1:2:3](0) step" begin
            Test.@test get_model_ic_step() isa OCP.Model
        end

        Test.@testset "build – x[1:end](0) (broken)" begin
            Test.@test_broken get_model_ic_end() isa OCP.Model
        end

        Test.@testset "build – x[1:n](0) variable (broken)" begin
            Test.@test_broken get_model_ic_var() isa OCP.Model
        end

        Test.@testset "discretise_exa – x[1:3](0)" begin
            Test.@test discretise_exa(get_model_ic_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[2:3](0) offset" begin
            Test.@test discretise_exa(get_model_ic_offset()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:2:3](0) step" begin
            Test.@test discretise_exa(get_model_ic_step()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:end](0) (broken)" begin
            Test.@test_broken discretise_exa(get_model_ic_end()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:n](0) variable (broken)" begin
            Test.@test_broken discretise_exa(get_model_ic_var()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – x[1:3](0)" begin
            m = discretise_exa(get_model_ic_1_3())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[2:3](0) offset" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_ic_offset()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:2:3](0) step" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_ic_step()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:end](0) (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_ic_end()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:n](0) variable (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_ic_var()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 3 – FINAL CONSTRAINT RANGE SLICES  x[a:b](tf) == c
        # ====================================================================

        function get_model_fc_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:3](1) == [0, 0, 0]
                ∫(u(t)^2) → min
            end
        end

        function get_model_fc_offset()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[2:3](1) == [0, 0]
                ∫(u(t)^2) → min
            end
        end

        function get_model_fc_step()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:2:3](1) == [0, 0]
                ∫(u(t)^2) → min
            end
        end

        function get_model_fc_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:end](1) == [0, 0, 0]
                ∫(u(t)^2) → min
            end
        end

        function get_model_fc_var()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:n](1) == [0, 0, 0]
                ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – x[1:3](tf)" begin
            Test.@test get_model_fc_1_3() isa OCP.Model
        end

        Test.@testset "build – x[2:3](tf) offset" begin
            Test.@test get_model_fc_offset() isa OCP.Model
        end

        Test.@testset "build – x[1:2:3](tf) step" begin
            Test.@test get_model_fc_step() isa OCP.Model
        end

        Test.@testset "build – x[1:end](tf) (broken)" begin
            Test.@test_broken get_model_fc_end() isa OCP.Model
        end

        Test.@testset "build – x[1:n](tf) variable (broken)" begin
            Test.@test_broken get_model_fc_var() isa OCP.Model
        end

        Test.@testset "discretise_exa – x[1:3](tf)" begin
            Test.@test discretise_exa(get_model_fc_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[2:3](tf) offset" begin
            Test.@test discretise_exa(get_model_fc_offset()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:2:3](tf) step" begin
            Test.@test discretise_exa(get_model_fc_step()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:end](tf) (broken)" begin
            Test.@test_broken discretise_exa(get_model_fc_end()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:n](tf) variable (broken)" begin
            Test.@test_broken discretise_exa(get_model_fc_var()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – x[1:3](tf)" begin
            m = discretise_exa(get_model_fc_1_3())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[2:3](tf) offset" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_fc_offset()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:2:3](tf) step" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_fc_step()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 4 – PATH / BOX CONSTRAINT SLICES  x[a:b](t) ≤ c
        # The exa parser rejects labeled indexed path constraints; labels are
        # omitted below. Even so, range path constraints are broken at exa level.
        # ====================================================================

        function get_model_path_scalar()
            # x₂(t) alias works as a scalar path constraint
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x₂(t) <= 2
                ∫(u(t)^2) → min
            end
        end

        function get_model_path_range_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:3](t) <= [2, 2, 2]
                ∫(u(t)^2) → min
            end
        end

        function get_model_path_range_offset()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[2:3](t) <= [2, 2]
                ∫(u(t)^2) → min
            end
        end

        function get_model_path_range_step()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:2:3](t) <= [2, 2]
                ∫(u(t)^2) → min
            end
        end

        function get_model_path_range_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:end](t) <= [2, 2, 2]
                ∫(u(t)^2) → min
            end
        end

        function get_model_path_range_var()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                x[1:n](t) <= [2, 2, 2]
                ∫(u(t)^2) → min
            end
        end

        function get_model_box_range()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                -2 <= x[1:3](t) <= 2
                ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – x₂(t) scalar path constraint" begin
            Test.@test get_model_path_scalar() isa OCP.Model
        end

        Test.@testset "build – x[1:3](t) range path constraint" begin
            Test.@test get_model_path_range_1_3() isa OCP.Model
        end

        Test.@testset "build – x[2:3](t) offset path constraint" begin
            Test.@test get_model_path_range_offset() isa OCP.Model
        end

        Test.@testset "build – x[1:2:3](t) step path constraint" begin
            Test.@test get_model_path_range_step() isa OCP.Model
        end

        Test.@testset "build – x[1:end](t) path constraint (broken)" begin
            Test.@test_broken get_model_path_range_end() isa OCP.Model
        end

        Test.@testset "build – x[1:n](t) variable path constraint (broken)" begin
            Test.@test_broken get_model_path_range_var() isa OCP.Model
        end

        Test.@testset "build – -c <= x[1:3](t) <= c box constraint (broken)" begin
            Test.@test_broken get_model_box_range() isa OCP.Model
        end

        Test.@testset "discretise_exa – x₂(t) scalar path constraint" begin
            Test.@test discretise_exa(get_model_path_scalar()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:3](t) range path" begin
            Test.@test discretise_exa(get_model_path_range_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – x₂(t) scalar path constraint" begin
            m = discretise_exa(get_model_path_scalar())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:3](t) range path" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_path_range_1_3()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 5 – DYNAMICS VIA ∂ NOTATION: RANGE SLICES
        # ====================================================================

        function get_model_dyn_range_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x[1:3](0) == [1, 0, 0]
                ∂(x[1:3])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_dyn_range_offset()
            # coordinate 1 by scalar, 2:3 by slice
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x[1])(t) == x₂(t)
                ∂(x[2:3])(t) == [x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_dyn_range_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x[1:end])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_dyn_range_var()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ∂(x[1:n])(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – ∂(x[1:3])(t) range dynamics" begin
            o = get_model_dyn_range_1_3()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test _dyn(o)(_T, _X, _U, nothing) == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "build – ∂(x[2:3])(t) offset range dynamics" begin
            o = get_model_dyn_range_offset()
            Test.@test o isa OCP.Model
            Test.@test _dyn(o)(_T, _X, _U, nothing) == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "build – ∂(x[1:end])(t) (broken)" begin
            Test.@test_broken get_model_dyn_range_end() isa OCP.Model
        end

        Test.@testset "build – ∂(x[1:n])(t) variable range" begin
            o = get_model_dyn_range_var()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test _dyn(o)(_T, _X, _U, nothing) == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "discretise_exa – ∂(x[1:3])(t) (broken)" begin
            Test.@test_broken discretise_exa(get_model_dyn_range_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∂(x[2:3])(t) offset (broken)" begin
            Test.@test_broken discretise_exa(get_model_dyn_range_offset()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∂(x[1:end])(t) (broken)" begin
            Test.@test_broken discretise_exa(get_model_dyn_range_end()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∂(x[1:n])(t) variable (broken)" begin
            Test.@test_broken discretise_exa(get_model_dyn_range_var()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – ∂(x[1:3])(t) (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_dyn_range_1_3()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – ∂(x[1:n])(t) variable (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_dyn_range_var()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 6 – DOTTED DERIVATIVE NOTATION  ẋ, ẋᵢ, ẋ[1:3]
        # ẋ aliases to ∂(x) and works; ẋᵢ and ẋ[a:b] are not yet aliased.
        # ====================================================================

        function get_model_dot_full()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_dot_scalar_subscript()
            # ẋ₁(t) as LHS – dotted scalar component (not yet aliased, see onepass.jl:519)
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ₁(t) == x₂(t)
                ẋ₂(t) == x₃(t)
                ẋ₃(t) == u(t)
                ∫(u(t)^2) → min
            end
        end

        function get_model_dot_scalar_ascii()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ1(t) == x₂(t)
                ẋ2(t) == x₃(t)
                ẋ3(t) == u(t)
                ∫(u(t)^2) → min
            end
        end

        function get_model_dot_range()
            # ẋ[1:3](t) – not yet supported
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ[1:3](t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – ẋ(t) full dotted dynamics" begin
            o = get_model_dot_full()
            Test.@test o isa OCP.Model
            Test.@test _dyn(o)(_T, _X, _U, nothing) == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "discretise_exa – ẋ(t) full dotted dynamics" begin
            Test.@test discretise_exa(get_model_dot_full()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – ẋ(t) full dotted dynamics" begin
            m = discretise_exa(get_model_dot_full())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "build – ẋ₁(t) scalar subscript (broken)" begin
            Test.@test_broken get_model_dot_scalar_subscript() isa OCP.Model
        end

        Test.@testset "build – ẋ1(t) scalar ascii (broken)" begin
            Test.@test_broken get_model_dot_scalar_ascii() isa OCP.Model
        end

        Test.@testset "build – ẋ[1:3](t) range (broken)" begin
            Test.@test_broken get_model_dot_range() isa OCP.Model
        end

        Test.@testset "solve – ẋ₁(t) scalar subscript (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_dot_scalar_subscript()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – ẋ[1:3](t) range (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_dot_range()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 7 – LAGRANGE COST SLICES  ∫( f(x[a:b](t)) ) → min
        # x[1:end] and x[1:n] in the integrand BUILD successfully (surprise).
        # ====================================================================

        function get_model_lag_scalar()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(x₁(t)^2 + x₂(t)^2 + x₃(t)^2 + u(t)^2) → min
            end
        end

        function get_model_lag_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:3](t).^2) + u(t)^2 → min
            end
        end

        function get_model_lag_offset()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[2:3](t).^2) + u(t)^2 → min
            end
        end

        function get_model_lag_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:end](t).^2) + u(t)^2 → min
            end
        end

        function get_model_lag_var()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:n](t).^2) + u(t)^2 → min
            end
        end

        Test.@testset "build – scalar Lagrange ∫(xᵢ(t)²)" begin
            o = get_model_lag_scalar()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – ∫(sum(x[1:3](t).^2))" begin
            o = get_model_lag_1_3()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – ∫(sum(x[2:3](t).^2)) offset" begin
            o = get_model_lag_offset()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – ∫(sum(x[1:end](t).^2))" begin
            o = get_model_lag_end()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – ∫(sum(x[1:n](t).^2)) variable" begin
            o = get_model_lag_var()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "discretise_exa – scalar Lagrange" begin
            Test.@test discretise_exa(get_model_lag_scalar()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∫(x[1:3](t)) (broken)" begin
            Test.@test_broken discretise_exa(get_model_lag_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∫(x[2:3](t)) offset (broken)" begin
            Test.@test_broken discretise_exa(get_model_lag_offset()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∫(x[1:end](t)) (broken)" begin
            Test.@test_broken discretise_exa(get_model_lag_end()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – ∫(x[1:n](t)) variable (broken)" begin
            Test.@test_broken discretise_exa(get_model_lag_var()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – scalar Lagrange" begin
            m = discretise_exa(get_model_lag_scalar())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – ∫(x[1:3](t)) (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_lag_1_3()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 8 – MAYER COST SLICES  x[a:b](tf) in terminal cost
        # All forms including end and variable BUILD successfully.
        # ====================================================================

        function get_model_mayer_1_3()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:3](tf).^2) + ∫(u(t)^2) → min
            end
        end

        function get_model_mayer_offset()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[2:3](tf).^2) + ∫(u(t)^2) → min
            end
        end

        function get_model_mayer_end()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:end](tf).^2) + ∫(u(t)^2) → min
            end
        end

        function get_model_mayer_var()
            n = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                sum(x[1:n](tf).^2) + ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – sum(x[1:3](tf).^2) Mayer" begin
            o = get_model_mayer_1_3()
            Test.@test o isa OCP.Model
            Test.@test OCP.state_dimension(o) == 3
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – sum(x[2:3](tf).^2) offset Mayer" begin
            o = get_model_mayer_offset()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – sum(x[1:end](tf).^2) Mayer" begin
            o = get_model_mayer_end()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "build – sum(x[1:n](tf).^2) variable Mayer" begin
            o = get_model_mayer_var()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "discretise_exa – x[1:3](tf) Mayer (broken)" begin
            Test.@test_broken discretise_exa(get_model_mayer_1_3()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[2:3](tf) offset Mayer (broken)" begin
            Test.@test_broken discretise_exa(get_model_mayer_offset()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:end](tf) Mayer (broken)" begin
            Test.@test_broken discretise_exa(get_model_mayer_end()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – x[1:n](tf) variable Mayer (broken)" begin
            Test.@test_broken discretise_exa(get_model_mayer_var()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – x[1:3](tf) Mayer (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_mayer_1_3()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[2:3](tf) offset Mayer (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_mayer_offset()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:end](tf) Mayer (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_mayer_end()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – x[1:n](tf) variable Mayer (broken)" begin
            Test.@test_broken MadNLP.madnlp(
                discretise_exa(get_model_mayer_var()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 9 – CONTROL u SLICES  (u ∈ R^3)
        # ====================================================================

        _X3, _U3 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]

        function get_model_ctrl_scalar()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_scalar_path()
            # scalar bracket u[2] as path constraint (no label)
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                u[2](t) <= 10
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_range_path()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                u[1:3](t) <= [10, 10, 10]
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_offset_path()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                u[2:3](t) <= [10, 10]
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_end_path()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                u[1:end](t) <= [10, 10, 10]
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_var_path()
            m = 3
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                u[1:m](t) <= [10, 10, 10]
                ∫(u(t)'*u(t)) → min
            end
        end

        function get_model_ctrl_range_lag()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R^3, control
                x(0) == [1, 0, 0]
                ẋ(t) == [u₁(t), u₂(t), u₃(t)]
                sum(u[1:3](t).^2) → min
            end
        end

        Test.@testset "build – u₁(t),u₂(t),u₃(t) scalar subscripts in dynamics" begin
            o = get_model_ctrl_scalar()
            Test.@test o isa OCP.Model
            Test.@test OCP.control_dimension(o) == 3
            Test.@test _dyn(o)(_T, _X3, _U3, nothing) == _U3
        end

        Test.@testset "build – u[2](t) scalar bracket path constraint" begin
            Test.@test get_model_ctrl_scalar_path() isa OCP.Model
        end

        Test.@testset "build – u[1:3](t) range path constraint" begin
            Test.@test get_model_ctrl_range_path() isa OCP.Model
        end

        Test.@testset "build – u[2:3](t) offset path constraint" begin
            Test.@test get_model_ctrl_offset_path() isa OCP.Model
        end

        Test.@testset "build – u[1:end](t) path constraint (broken)" begin
            Test.@test_broken get_model_ctrl_end_path() isa OCP.Model
        end

        Test.@testset "build – u[1:m](t) variable path constraint (broken)" begin
            Test.@test_broken get_model_ctrl_var_path() isa OCP.Model
        end

        Test.@testset "build – sum(u[1:3](t).^2) Lagrange" begin
            o = get_model_ctrl_range_lag()
            Test.@test o isa OCP.Model
            Test.@test OCP.criterion(o) == :min
        end

        Test.@testset "discretise_exa – u₁,u₂,u₃ subscripts in dynamics" begin
            Test.@test discretise_exa(get_model_ctrl_scalar()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – u[2](t) scalar path" begin
            Test.@test discretise_exa(get_model_ctrl_scalar_path()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – u[1:3](t) range path" begin
            Test.@test discretise_exa(get_model_ctrl_range_path()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – sum(u[1:3](t).^2) Lagrange (broken)" begin
            Test.@test_broken discretise_exa(get_model_ctrl_range_lag()) isa ExaModels.ExaModel
        end

        Test.@testset "solve – u subscripts in dynamics" begin
            m = discretise_exa(get_model_ctrl_scalar())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – u[2](t) scalar path" begin
            m = discretise_exa(get_model_ctrl_scalar_path())
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "solve – u[1:3](t) range path" begin
            Test.@test MadNLP.madnlp(
                discretise_exa(get_model_ctrl_range_path()); tol=1e-6, print_level=MadNLP.ERROR
            ).status == MadNLP.SOLVE_SUCCEEDED
        end

        # ====================================================================
        # SECTION 10 – VARIABLE v SLICES  (v ∈ R^2)
        # ====================================================================

        function get_model_var_subscript()
            # v₁, v₂ subscript access in cost
            return CTParser.@def begin
                tf ∈ R, variable
                t ∈ [0, tf], time
                x ∈ R^2, state
                u ∈ R, control
                x(0) == [0, 0]
                ẋ(t) == [x₂(t), u(t)]
                tf → min
            end
        end

        function get_model_var_bracket_cost()
            # v[1] bracket access in Bolza cost; parentheses required around the Mayer part
            return CTParser.@def begin
                v ∈ R^2, variable
                t ∈ [0, 1], time
                x ∈ R^2, state
                u ∈ R, control
                x(0) == [0, 0]
                ẋ(t) == [x₂(t), u(t)]
                (v[1] + v[2]) + ∫(u(t)^2) → min
            end
        end

        function get_model_var_range_cost()
            # v[1:2] range access in cost
            return CTParser.@def begin
                v ∈ R^2, variable
                t ∈ [0, 1], time
                x ∈ R^2, state
                u ∈ R, control
                x(0) == [0, 0]
                ẋ(t) == [x₂(t), u(t)]
                sum(v[1:2].^2) + ∫(u(t)^2) → min
            end
        end

        Test.@testset "build – scalar variable tf ∈ R" begin
            o = get_model_var_subscript()
            Test.@test o isa OCP.Model
            Test.@test OCP.variable_dimension(o) == 1
        end

        Test.@testset "build – v[1], v[2] bracket in cost" begin
            o = get_model_var_bracket_cost()
            Test.@test o isa OCP.Model
            Test.@test OCP.variable_dimension(o) == 2
        end

        Test.@testset "build – sum(v[1:2].^2) range in cost" begin
            o = get_model_var_range_cost()
            Test.@test o isa OCP.Model
            Test.@test OCP.variable_dimension(o) == 2
        end

        Test.@testset "discretise_exa – (v[1]+v[2])+∫ Bolza bracket" begin
            Test.@test discretise_exa(get_model_var_bracket_cost()) isa ExaModels.ExaModel
        end

        Test.@testset "discretise_exa – sum(v[1:2].^2) range in cost (broken)" begin
            Test.@test_broken discretise_exa(get_model_var_range_cost()) isa ExaModels.ExaModel
        end

        # ====================================================================
        # SECTION 11 – NOTATION EQUIVALENCE
        # x[i](t), xᵢ(t), xi(t) must produce identical dynamics.
        # Guard the recommended idiomatic forms (JB: scalar u(t), index u₂(t)).
        # ====================================================================

        function get_model_equiv_bracket()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x[2](t), x[3](t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_equiv_subscript()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x₂(t), x₃(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_equiv_ascii()
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^3, state
                u ∈ R, control
                x(0) == [1, 0, 0]
                ẋ(t) == [x2(t), x3(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_idiom_scalar_u()
            # idiomatic scalar control: u(t) not u[1](t)
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^2, state
                u ∈ R, control
                x(0) == [1, 0]
                ẋ(t) == [x₂(t), u(t)]
                ∫(u(t)^2) → min
            end
        end

        function get_model_idiom_index_u()
            # idiomatic index notation: u₁(t), u₂(t) for vector control
            return CTParser.@def begin
                t ∈ [0, 1], time
                x ∈ R^2, state
                u ∈ R^2, control
                x(0) == [1, 0]
                ẋ(t) == [u₁(t), u₂(t)]
                ∫(u(t)'*u(t)) → min
            end
        end

        Test.@testset "equivalence – x[i](t) == xᵢ(t) == xi(t) produce same dynamics" begin
            ob = get_model_equiv_bracket()
            os = get_model_equiv_subscript()
            oa = get_model_equiv_ascii()
            Test.@test ob isa OCP.Model
            Test.@test os isa OCP.Model
            Test.@test oa isa OCP.Model
            db = _dyn(ob)(_T, _X, _U, nothing)
            ds = _dyn(os)(_T, _X, _U, nothing)
            da = _dyn(oa)(_T, _X, _U, nothing)
            Test.@test db == ds
            Test.@test db == da
            Test.@test db == [_X[2], _X[3], _U[1]]
        end

        Test.@testset "idiom – scalar u(t) control parses and solves" begin
            o = get_model_idiom_scalar_u()
            Test.@test o isa OCP.Model
            Test.@test OCP.control_dimension(o) == 1
            m = discretise_exa(o)
            Test.@test m isa ExaModels.ExaModel
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "idiom – u₁(t),u₂(t) index notation for vector control" begin
            o = get_model_idiom_index_u()
            Test.@test o isa OCP.Model
            Test.@test OCP.control_dimension(o) == 2
            m = discretise_exa(o)
            Test.@test m isa ExaModels.ExaModel
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

        Test.@testset "idiom – x[i] form discretises and solves" begin
            m = discretise_exa(get_model_equiv_bracket())
            Test.@test m isa ExaModels.ExaModel
            Test.@test MadNLP.madnlp(m; tol=1e-6, print_level=MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
        end

    end # outer testset
end # test_slices

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_slices() = TestSlices.test_slices()
