# ==============================================================================
# CTParser Test Runner
# ==============================================================================

# Test dependencies
using Test
using Aqua
using OrderedCollections: OrderedDict
import CTParser:
    CTParser,
    subs,
    subs2,
    subs2m,
    subs3,
    replace_call,
    has,
    concat,
    constraint_type,
    @def,
    @init,
    prefix_fun,
    prefix_fun!,
    prefix_exa,
    prefix_exa!,
    e_prefix,
    e_prefix!,
    activate_backend,
    deactivate_backend,
    is_active_backend,
    @def_exa # todo: remove __default... (as soon as discretise_exa has been moved to CTDirect)
import CTBase: CTBase, ParsingError, PreconditionError
import CTModels:
    CTModels,
    initial_time,
    final_time,
    time_name,
    variable_dimension,
    variable_components,
    variable_name,
    state_dimension,
    state_components,
    state_name,
    control_dimension,
    control_components,
    control_name,
    constraint,
    dynamics,
    mayer,
    lagrange,
    criterion,
    Model,
    get_build_examodel
using ExaModels: ExaModels
using MadNLP
using MadNLPGPU
using CUDA
using BenchmarkTools
using Interpolations
using NLPModels
using LinearAlgebra

include("utils.jl")

const VERBOSE = true
const SHOWTIMING = true

# Run tests using the TestRunner extension
CTBase.run_tests(;
    args=String.(ARGS),
    testset_name="CTParser tests",
    available_tests=(
        "suite/test_*",
        ),
    filename_builder=name -> "test_$(name).jl",
    funcname_builder=name -> "test_$(name)",
    verbose=VERBOSE,
    showtiming=SHOWTIMING,
    test_dir=@__DIR__,
)

# If running with coverage enabled, remind the user to run the post-processing script
# because .cov files are flushed at process exit and cannot be cleaned up by this script.
if Base.JLOptions().code_coverage != 0
    println(
        """
        ================================================================================
        Coverage files generated. To process them, please run:

            julia --project -e 'using Pkg; Pkg.test("CTParser"; coverage=true); include("test/coverage.jl")'

        ================================================================================
        """
    )
end