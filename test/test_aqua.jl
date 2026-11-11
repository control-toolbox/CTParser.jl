module TestAqua

# Code quality of the package (Aqua) and of its extensions (ct-conventions TST8). The
# triggers of every extension are loaded here, since the TST8 check requires each one to be
# loaded and this file may run before the tests that load them.

using Test: Test
using Aqua: Aqua
using CTBase: CTBase
using CTParser: CTParser
using ExaModels: ExaModels
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

include(joinpath(@__DIR__, "extension_piracies.jl"))

function test_aqua()
    Test.@testset "Aqua.jl" verbose = VERBOSE showtiming = SHOWTIMING begin
        Aqua.test_all(
            CTParser;
            ambiguities=true, # also tests submodules
            deps_compat=(ignore=[:LinearAlgebra, :Unicode],),
            piracies=true,
        )
    end
    # Aqua.test_all does not inspect extensions: no ambiguity, and every pirated method
    # listed with its reason (extension_piracies.jl).
    CTBase.DevTools.test_extensions_quality(CTParser; allowed_piracies=EXTENSION_PIRACIES)
    return nothing
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_aqua() = TestAqua.test_aqua()
