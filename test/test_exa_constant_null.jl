module TestExaConstantNull

# `ext/CTParserExaModels/` folds structural zeros and ones on `ExaModels.Null`;
# ExaModels folds them on `ExaModels.Constant{V}` (src/specialization.jl, every 0.12.x).
# A call mixing the two, such as `Null(2) + Constant(0)`, matched a rule of each side and
# was ambiguous (#364). These tests pin the intersection methods that resolve it, their
# scope, and the `@def` path that reaches them.

using Test: Test
using CTBase: CTBase
using CTParser: CTParser, @def
using CTModels: CTModels
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using MadNLP: MadNLP

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

# The `:exa` builders are generated when `@def` is expanded, so the backend must be
# active before the module body below is lowered (another test file may have
# deactivated it).
CTParser.activate_backend(:exa)

const EXT = Base.get_extension(CTParser, :CTParserExaModels)

# Values for which ExaModels defines a `Constant` folding rule, by operator.
const ZEROS = (0, 0.0, Float32(0), Float16(0))
const ONES = (1, 1.0, Float32(1), Float16(1))

# A node type the extension knows nothing about: the new methods must not capture it.
struct FakeNode <: ExaModels.AbstractNode end

# The pairs (op, a, b) that were ambiguous: + and - with a zero Constant, * with a zero
# or a one Constant, the Constant on either side.
function ambiguous_cases(a::ExaModels.Null)
    cases = Tuple{Function,Any,Any}[]
    for op in (+, -, *), z in ZEROS
        push!(cases, (op, a, ExaModels.Constant(z)), (op, ExaModels.Constant(z), a))
    end
    for o in ONES
        push!(cases, (*, a, ExaModels.Constant(o)), (*, ExaModels.Constant(o), a))
    end
    return cases
end

value_of(n::ExaModels.Null) = n.value
value_of(::ExaModels.Constant{V}) where {V} = V

# In an ambiguous pair (a, b), the method of the extension and the other one.
ext_method(a::Method, b::Method) = a.module === EXT ? a : b
other_method(a::Method, b::Method) = a.module === EXT ? b : a

# Whether a method takes an array argument (a vector, a matrix, an adjoint of a vector):
# the linear-algebra part of the extension, as opposed to scalar node arithmetic.
function has_array_arg(m::Method)
    sig = Base.unwrap_unionall(m.sig)
    return any(sig.parameters[2:end]) do p
        q = Base.unwrap_unionall(p)
        q isa TypeVar && (q = q.ub)
        return q isa Type && q <: AbstractArray
    end
end

# Build the `:exa` model of `ocp` on the CPU, as the getter of CTModels does.
function build_exa(ocp; scheme=:trapeze, grid_size=20)
    build = CTModels.get_build_examodel(ocp)
    model, _ = build(;
        scheme=scheme,
        grid_size=grid_size,
        backend=CTParser.__default_backend_exa(),
        init=CTParser.__default_init_exa(),
        base_type=Float64,
    )
    return model
end

solve(model) = MadNLP.madnlp(model; print_level=MadNLP.ERROR, tol=1e-10)

function test_exa_constant_null()
    Test.@testset "Null op Constant (#364)" verbose = VERBOSE showtiming = SHOWTIMING begin

        # ------------------------------------------------------------------
        # Unit: every formerly ambiguous call returns the folded Null
        # ------------------------------------------------------------------
        Test.@testset "the 32 formerly ambiguous calls fold to a Null" begin
            for a in (ExaModels.Null(2), ExaModels.Null(-3.5), ExaModels.Null(0))
                cases = ambiguous_cases(a)
                Test.@test length(cases) == 32
                for (op, x, y) in cases
                    r = op(x, y)
                    Test.@test r isa ExaModels.Null
                    Test.@test value_of(r) == op(value_of(x), value_of(y))
                end
            end
        end

        Test.@testset "values agree with ExaModels' own folding" begin
            # ExaModels: 0 + x → x, x - 0 → x, 0 - x → -x, 0 * x → 0, 1 * x → x.
            n = ExaModels.Null(2.5)
            Test.@test value_of(n + ExaModels.Constant(0)) == 2.5
            Test.@test value_of(ExaModels.Constant(0.0) + n) == 2.5
            Test.@test value_of(n - ExaModels.Constant(0)) == 2.5
            Test.@test value_of(ExaModels.Constant(0) - n) == -2.5
            Test.@test value_of(n * ExaModels.Constant(0)) == 0
            Test.@test value_of(ExaModels.Constant(Float32(0)) * n) == 0
            Test.@test value_of(n * ExaModels.Constant(1)) == 2.5
            Test.@test value_of(ExaModels.Constant(Float16(1)) * n) == 2.5
        end

        # ------------------------------------------------------------------
        # Contract: the new methods are exactly the intersections, nothing wider
        # ------------------------------------------------------------------
        Test.@testset "a Constant that is not zero or one is left to the Null rules" begin
            # Not ambiguous before, not folded now: `Null + AbstractNode` builds a Node2.
            r = ExaModels.Null(2) + ExaModels.Constant(5)
            Test.@test r isa ExaModels.Node2
            r = ExaModels.Constant(5) * ExaModels.Null(2)
            Test.@test r isa ExaModels.Node2
        end

        Test.@testset "an unrelated node type is not captured" begin
            Test.@test (ExaModels.Null(2) + FakeNode()) isa ExaModels.Node2
            Test.@test (FakeNode() * ExaModels.Null(3)) isa ExaModels.Node2
            # the extension's own zero/one elimination still applies
            Test.@test (ExaModels.Null(0) + FakeNode()) isa FakeNode
            Test.@test (ExaModels.Null(1) * FakeNode()) isa FakeNode
        end

        Test.@testset "ExaModels keeps its own rules for Constant with other nodes" begin
            # Not the extension's business: ExaModels' rules answer, as before.
            Test.@test (FakeNode() + ExaModels.Constant(0)) isa FakeNode
            Test.@test (ExaModels.Constant(0) * FakeNode()) isa ExaModels.Constant{0}
        end

        # ------------------------------------------------------------------
        # Non-regression: no ambiguity left between the extension and ExaModels
        # ------------------------------------------------------------------
        Test.@testset "no ambiguity left on scalar node arithmetic" begin
            Test.@test EXT isa Module
            pairs = Test.detect_ambiguities(EXT; recursive=true)
            with_exa = [(a, b) for (a, b) in pairs if other_method(a, b).module === ExaModels]
            Test.@test isempty(with_exa)
            # What is left pairs a method of the extension on *arrays* of nodes with a
            # method on another array type (LinearAlgebra's structured matrices,
            # SparseArrays, and, depending on what the session has loaded, StaticArrays or
            # GPUArrays): tracked in #365. A scalar method of the extension must not be
            # ambiguous with anything.
            scalar = [(a, b) for (a, b) in pairs if !has_array_arg(ext_method(a, b))]
            Test.@test isempty(scalar) || begin
                for (a, b) in scalar
                    println(stderr, "  ambiguous: ", ext_method(a, b), "  vs  ", other_method(a, b))
                end
                false
            end
        end

        # ------------------------------------------------------------------
        # Integration: the `@def` path that reached the ambiguity
        # ------------------------------------------------------------------
        Test.@testset "@def mixing ExaModels.Constant and a structural zero builds" begin
            # `0 * x₂(t)` is folded to `Null(0)` by the extension; adding a `Constant`
            # (which ExaModels documents for value-in-type folding) used to raise
            # `MethodError: +(::Null{Int64}, ::Constant{0}) is ambiguous`.
            with_constant = @def begin
                t ∈ [0, 1], time
                x ∈ R², state
                u ∈ R, control
                x(0) == [0, 0]
                x(1) == [1, 0]
                ∂(x₁)(t) == x₂(t) + 0 * x₁(t) + ExaModels.Constant(0)
                ∂(x₂)(t) == ExaModels.Constant(1) * u(t) + 0 * x₂(t) * ExaModels.Constant(1)
                ∫(0.5u(t)^2) → min
            end
            reference = @def begin
                t ∈ [0, 1], time
                x ∈ R², state
                u ∈ R, control
                x(0) == [0, 0]
                x(1) == [1, 0]
                ∂(x₁)(t) == x₂(t)
                ∂(x₂)(t) == u(t)
                ∫(0.5u(t)^2) → min
            end
            for scheme in (:euler, :euler_implicit, :midpoint, :trapeze)
                m1 = build_exa(with_constant; scheme=scheme)
                m2 = build_exa(reference; scheme=scheme)
                Test.@test m1 isa ExaModels.ExaModel
                Test.@test m1.meta.nvar == m2.meta.nvar
                Test.@test m1.meta.ncon == m2.meta.ncon
                s1 = solve(m1)
                s2 = solve(m2)
                Test.@test s1.status == MadNLP.SOLVE_SUCCEEDED
                Test.@test s2.status == MadNLP.SOLVE_SUCCEEDED
                Test.@test s1.objective ≈ s2.objective rtol = 1e-8
            end
        end

        Test.@testset "@def with literal zeros and ones only is unchanged" begin
            # The ordinary path never builds a Constant: it must keep building.
            ocp = @def begin
                t ∈ [0, 1], time
                x ∈ R², state
                u ∈ R, control
                x(0) == [0, 0]
                x(1) == [1, 0]
                ∂(x)(t) == [0 1; 0 0] * x(t) + [0, 1] * u(t)
                ∫(0.5u(t)^2) → min
            end
            m = build_exa(ocp)
            Test.@test m isa ExaModels.ExaModel
            Test.@test solve(m).status == MadNLP.SOLVE_SUCCEEDED
        end
    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_exa_constant_null() = TestExaConstantNull.test_exa_constant_null()
