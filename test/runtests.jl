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
# Qualified: a bare `using ExaModels` brings its exported `constraint` into Main, where
# it clashes with the `constraint` imported from CTModels above and every :exa test run
# opens with "WARNING: using ExaModels.constraint in module Main conflicts with an
# existing identifier". Test files already write ExaModels.x throughout. Fixes #230.
using ExaModels: ExaModels
using LinearAlgebra
using MadNLP
using MadNLPGPU
using CUDA
# MadNLPGPU 0.10 moved CUDSS from [deps] to [weakdeps]: its CUDA extension now triggers
# on ["CUDACore", "CUDSS", "cuBLAS", "cuSOLVER", "cuSPARSE"], so CUDSS no longer arrives
# transitively and the consumer must load it. Without this, every GPU solve dies with
# "MadNLPGPU: cannot build a GPU sparse KKT system because the GPU backend extension is
# not loaded". Invisible on a CPU-only machine, where CUDA.functional() is false and the
# GPU paths never run.
using CUDSS: CUDSS
using BenchmarkTools
using Interpolations
using NLPModels

# Capability constants, computed once, here. `CUDA_FUNCTIONAL` is the suite's single
# CUDA-device predicate — never write a bare `CUDA.functional()` guard in a test file
# (duplicated copies drift; Handbook philosophy/testing.md §"Capability-gated tests").
# `ON_GPU_RUNNER` turns the device tier from *skipped* into *required* on the self-hosted
# GPU runners: `RUNNER_NAME` is set by the GitHub Actions runner agent itself (no CI.yml
# or CTActions change needed) to the runner's *registered* name. Ours are registered as
# `kkt-runner` / `occidata-runner` — the CI.yml `runs_on` label is the bare
# `kkt`/`occidata`, a different string — so match on the substring to survive the
# `-runner` suffix. Enforcement lives centrally in test/test_environment_contract.jl.
module TestCapabilities
using CUDA: CUDA
using CUDSS: CUDSS          # with CUDA, arms MadNLPGPUCUDAExt
using MadNLPGPU: MadNLPGPU

const CUDA_FUNCTIONAL = CUDA.functional()
const ON_GPU_RUNNER = any(
    gpu -> occursin(gpu, get(ENV, "RUNNER_NAME", "")), ("kkt", "occidata")
)
# `isdefined`, not CTSolvers' `MadNLPGPU.CUDSSSolver isa Type`: the symbol only exists
# once MadNLPGPUCUDAExt loads, and an UndefVarError at module load would abort the whole
# run instead of failing one assertion.
const GPU_SOLVER_ARMED = isdefined(MadNLPGPU, :CUDSSSolver)
end

if TestCapabilities.CUDA_FUNCTIONAL
    println("✓ CUDA functional, GPU tests enabled")
else
    println("⚠️  CUDA not functional, GPU device tests will be skipped (Test.@test_skip)")
end

include("utils.jl")

# Controls nested testset output formatting (used by individual test files)
module TestData
const VERBOSE = true
const SHOWTIMING = true
end
using .TestData: VERBOSE, SHOWTIMING

# Run tests using the TestRunner extension
CTBase.run_tests(;
    args=String.(ARGS),
    testset_name="CTParser tests",
    available_tests=("test_*",),
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
        """,
    )
end
