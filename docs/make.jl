using Documenter
using CTParser
using CTBase

repo_url = "github.com/control-toolbox/CTParser.jl"

# Paths to source files
src_dir = abspath(joinpath(@__DIR__, "..", "src"))

src(files...) = [abspath(joinpath(src_dir, f)) for f in files]

# Symbols to exclude from automatic reference docs (generated helpers, etc.)
const EXCLUDE_SYMBOLS = Symbol[
    :include,
    :eval,
]

makedocs(
    draft=false,
    remotes=nothing,
    warnonly=true,
    sitename="CTParser.jl",
    format=Documenter.HTML(
        repolink="https://" * repo_url,
        prettyurls=false,
        assets=[
            asset("https://control-toolbox.org/assets/css/documentation.css"),
            asset("https://control-toolbox.org/assets/js/documentation.js"),
        ],
    ),
    checkdocs=:none,
    pages=[
        "Introduction" => "index.md",
        "API" => [
            CTBase.automatic_reference_documentation(
                subdirectory=".",
                primary_modules=[
                    CTParser => src(
                        "CTParser.jl",
                        "defaults.jl",
                        "utils.jl",
                        "onepass.jl",
                        "initial_guess.jl",
                    ),
                ],
                exclude=EXCLUDE_SYMBOLS,
                public=false,
                private=true,
                title="CTParser",
                title_in_menu="CTParser",
                filename="ctparser",
            ),
        ],
    ],
)

deploydocs(; repo=repo_url * ".git", devbranch="main")
