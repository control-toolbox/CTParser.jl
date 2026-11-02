# test_hygiene.jl
# Regression tests for #356: the expressions emitted by @def, @def_exa and @init
# interpolate the FUN / EXA / ERR module *objects* (CTModels, ExaModels, CTBase),
# so they resolve as GlobalRefs where the macros are defined. They must not look
# up the names CTModels / ExaModels / CTBase in the caller's module.
#
# A nested module is a faithful reproduction of `import OptimalControl`: the
# packages are loaded (CTParser brings them in as dependencies), but none of
# their names is bound in the module's own namespace. Before the fix the same
# code raised UndefVarError: `CTModels` not defined in <module>.
module TestHygiene

using Test: Test
using CTParser: CTParser
using CTModels: CTModels
using CTBase: CTBase
using ExaModels: ExaModels

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

# Bare namespace: only the macros are bound. `import OptimalControl` leaves a
# user with exactly this. The OCP, the initial guess and the ExaModels builder
# must all be constructible.
module Bare
using CTParser: @def, @init, @def_exa

const ocp = @def begin
    t ∈ [0, 1], time
    x = (q, v) ∈ R², state
    u ∈ R, control
    x(0) == [-1, 0]
    ẋ(t) == [v(t), u(t)]
    ∫(0.5u(t)^2) → min
end

const ig = @init ocp begin
    u(t) := t
end

# @def_exa skips the :fun build, so a block without dynamics yields the :exa
# builder itself; its dyn_check emits ERR.ParsingError when called.
const build_nodyn = @def_exa begin
    t ∈ [0, 1], time
    x ∈ R, state
    u ∈ R, control
    ∫(u(t)^2) → min
end

# __throw emits ERR.ParsingError at macro-expansion time; the throw expression
# runs in this module when the function is called.
broken_init(ocp) = @init ocp begin
    (q + v) := 1.0
end
end # module Bare

# Shadowed namespace: the same names are bound to unrelated values. If the
# generated code still resolved them by name, `CTModels.PreModel()` would mean
# `42.PreModel()`. The interpolated module objects must win over any binding.
module Shadowed
const CTModels = 42
const ExaModels = "nope"
const CTBase = nothing

using CTParser: @def, @init

const ocp = @def begin
    t ∈ [0, 1], time
    x ∈ R, state
    u ∈ R, control
    ẋ(t) == u(t)
    ∫(u(t)^2) → min
end

const ig = @init ocp begin
    u(t) := 0.0
end
end # module Shadowed

function test_hygiene()
    Test.@testset "Generated code carries the modules, not their names (#356)" verbose =
        VERBOSE showtiming = SHOWTIMING begin
        # --- :fun backend in a bare module ---
        Test.@test Bare.ocp isa CTModels.Model
        Test.@test CTModels.initial_time(Bare.ocp) == 0
        Test.@test CTModels.final_time(Bare.ocp) == 1
        Test.@test CTModels.state_dimension(Bare.ocp) == 2

        # --- @init in the same bare module ---
        Test.@test Bare.ig isa CTModels.AbstractInitialGuess

        # --- :exa builder stored on the model emits ExaModels.* calls ---
        exa, getter = CTModels.get_build_examodel(Bare.ocp)(; grid_size=10)
        Test.@test exa isa ExaModels.ExaModel
        Test.@test getter isa Function

        # --- ERR interpolation: errors raised *from generated code* in a module
        # where CTBase is not bound must still be the typed CTBase exceptions ---
        Test.@test_throws CTBase.IncorrectArgument getter(nothing; val=:bogus)
        Test.@test_throws CTBase.ParsingError Bare.build_nodyn()
        Test.@test_throws CTBase.ParsingError Bare.broken_init(Bare.ocp)

        # --- shadowed bindings must not be picked up ---
        Test.@test Shadowed.ocp isa CTModels.Model
        Test.@test CTModels.initial_time(Shadowed.ocp) == 0
        Test.@test Shadowed.ig isa CTModels.AbstractInitialGuess
        exa2, getter2 = CTModels.get_build_examodel(Shadowed.ocp)(; grid_size=5)
        Test.@test exa2 isa ExaModels.ExaModel
        Test.@test_throws CTBase.IncorrectArgument getter2(nothing; val=:bogus)
    end
end

end # module TestHygiene

# CRITICAL: Redefine in outer scope for TestRunner
test_hygiene() = TestHygiene.test_hygiene()
