module TestEnvironmentContract

using Test: Test

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

# Central enforcement of the Handbook's capability-gated-test contract (philosophy/testing.md
# §"Capability-gated tests"), the CTParser side of control-toolbox/CTParser.jl#339. CTSolvers
# carries the same file at test/suite/environment/test_environment_contract.jl; two
# deliberate differences:
#
#  - its companion `isdefined(Main, ...)` audit is not ported. That audit exists because
#    every CTSolvers suite file is wrapped in its own module, which makes such a check
#    always false. CTParser's test files are a mix of module-wrapped (this one,
#    test_control_zero.jl) and flat files included straight into `Main`, where the idiom is
#    legitimate, so the audit would flag correct code.
#  - the silent-guard audit below is stricter than CTSolvers' `if`-anchored regex, because
#    the anti-pattern CTParser actually had was the short-circuit `CUDA.functional() && ...`,
#    which an `if`-anchored pattern does not see.

"""
    _silent_cuda_guard_offenders()

Find, among the test files next to this one (`@__DIR__`, not `pwd()`, so the caller's
working directory does not matter), every mention of a raw CUDA-device predicate — a local
`is_cuda_on()` or a bare `CUDA.functional()` call.

The suite has exactly one such predicate, `Main.TestCapabilities.CUDA_FUNCTIONAL`, computed
in `test/runtests.jl`. Any other copy is the anti-pattern this file exists to catch: it
makes a correctly-skipped run (no device, as expected on a CPU/developer machine) and a
silently-broken one (device *should* be present but isn't) produce the same output — a
green testset with zero assertions. The fix is
`if Main.TestCapabilities.CUDA_FUNCTIONAL ... else Test.@test_skip ... end`, with the
device tier made *required* on the GPU runners by the testset below.

Two files are excluded from the walk: `runtests.jl`, which legitimately holds the single
definition and the comments naming it, and this file, which necessarily spells out the very
pattern it searches for.
"""
function _silent_cuda_guard_offenders()
    test_dir = @__DIR__
    excluded = ("runtests.jl", basename(@__FILE__))
    offenders = Tuple{String,Int,String}[]
    # Assembled from two literals so this line does not match itself.
    pattern = Regex("(is_cuda_on\\(\\)|CUDA" * "\\.functional\\(\\))")
    for f in sort(readdir(test_dir))
        (endswith(f, ".jl") && f ∉ excluded) || continue
        for (lineno, line) in enumerate(eachline(joinpath(test_dir, f)))
            if match(pattern, line) !== nothing
                push!(offenders, (f, lineno, strip(line)))
            end
        end
    end
    return offenders
end

function test_environment_contract()
    Test.@testset "Test-environment contract" verbose = VERBOSE showtiming = SHOWTIMING begin
        Test.@testset "GPU solver extension is armed" begin
            # Runs on every runner, CPU laptops included: "armed" comes from packages being
            # loaded (test/Project.toml + the `using`s in runtests.jl), not from a driver
            # being present. This is the assertion that catches the CUDSS wiring regression
            # — MadNLPGPU 0.10 moved CUDSS to [weakdeps], and without it every GPU solve
            # dies with "cannot build a GPU sparse KKT system because the GPU backend
            # extension is not loaded", invisibly on a CPU-only machine.
            Test.@test Main.TestCapabilities.GPU_SOLVER_ARMED
        end

        Test.@testset "GPU driver required on the GPU runner" begin
            # On a machine that is supposed to have a GPU, a missing or broken device fails
            # loudly here rather than being silently skipped everywhere else.
            #
            # `RUNNER_NAME` is set automatically by the GitHub Actions runner agent itself
            # (no .github/workflows/CI.yml or CTActions change needed) to the runner's
            # *registered* name — `kkt-runner` / `occidata-runner` for our self-hosted GPU
            # runners, where the CI.yml `runs_on` label is the bare `kkt` / `occidata`.
            # `ON_GPU_RUNNER` (test/runtests.jl) matches on the substring, so it survives
            # the `-runner` suffix; if a runner is renamed past that, this check stops
            # firing silently rather than failing loudly.
            if Main.TestCapabilities.ON_GPU_RUNNER
                Test.@test Main.TestCapabilities.CUDA_FUNCTIONAL
            else
                Test.@test_skip "a CUDA device is only required on the kkt/occidata runners"
            end
        end

        Test.@testset "silent CUDA-guard anti-pattern has not returned" begin
            offenders = _silent_cuda_guard_offenders()
            Test.@test isempty(offenders)
            for (file, lineno, text) in offenders
                @warn "silent CUDA guard at $file:$lineno — read Main.TestCapabilities.CUDA_FUNCTIONAL and give it a Test.@test_skip else-branch" text
            end
        end
    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_environment_contract() = TestEnvironmentContract.test_environment_contract()
