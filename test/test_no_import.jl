module TestNoImport

# Regression guard for the ct-conventions rules "using, never import" (`MOD3`) and
# "qualified external imports" (`MOD4`): flags any tracked `.jl` or `.md` file that uses
# the `import` keyword, or the forbidden dotted form `using Pkg.Sub` (the canonical
# spelling is `using Pkg: Sub`). Modelled on CTLie's `test_no_import.jl`.

using Test: Test

const VERBOSE = isdefined(Main, :TestData) ? Main.TestData.VERBOSE : true
const SHOWTIMING = isdefined(Main, :TestData) ? Main.TestData.SHOWTIMING : true

const IMPORT_RE = r"^\s*import\b"
const DOTTED_USING_RE = r"^\s*using\s+[A-Za-z_]\w*\.[A-Za-z_]"

# These two files quote past text verbatim (a warning message, a "Before" snippet): it is
# history, not current usage, so it is exempt from the dotted-`using` check.
const DOTTED_USING_EXEMPT = ("BREAKING.md", "CHANGELOG.md")

function _report_violations(violations)
    println(stderr, "Found forbidden import syntax:")
    for violation in violations
        println(stderr, "  ", violation)
    end
    return false
end

function test_no_import()
    Test.@testset "No `import` / no dotted `using Pkg.Sub` (MOD3, MOD4)" verbose = VERBOSE showtiming =
        SHOWTIMING begin
        repo_root = joinpath(@__DIR__, "..")
        tracked = filter(
            f -> endswith(f, ".jl") || endswith(f, ".md"),
            readlines(Cmd(`git ls-files`; dir=repo_root)),
        )
        violations = String[]
        for relpath in tracked
            path = joinpath(repo_root, relpath)
            check_dotted = !(basename(relpath) in DOTTED_USING_EXEMPT)
            for (line_number, line) in enumerate(eachline(path))
                if occursin(IMPORT_RE, line)
                    push!(violations, "$relpath:$line_number: `import`: $(strip(line))")
                end
                if check_dotted && occursin(DOTTED_USING_RE, line)
                    push!(
                        violations, "$relpath:$line_number: dotted `using`: $(strip(line))"
                    )
                end
            end
        end
        Test.@test isempty(violations) || _report_violations(violations)
    end
end

end # module

# CRITICAL: Redefine in outer scope for TestRunner
test_no_import() = TestNoImport.test_no_import()
