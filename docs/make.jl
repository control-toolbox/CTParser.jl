using Documenter
using CTParser
using CTBase
using Markdown
using MarkdownAST: MarkdownAST

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
    format=Documenter.HTML(;
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
        "API Reference" => [
            CTBase.automatic_reference_documentation(
                subdirectory=".",
                primary_modules=[
                    CTParser => src(
                        "defaults.jl",
                    ),
                ],
                exclude=EXCLUDE_SYMBOLS,
                public=false,
                private=true,
                title="Defaults",
                title_in_menu="Defaults",
                filename="defaults",
            ),
            CTBase.automatic_reference_documentation(
                subdirectory=".",
                primary_modules=[
                    CTParser => src(
                        "utils.jl",
                    ),
                ],
                exclude=EXCLUDE_SYMBOLS,
                public=false,
                private=true,
                title="Utils",
                title_in_menu="Utils",
                filename="utils",
            ),
            CTBase.automatic_reference_documentation(
                subdirectory=".",
                primary_modules=[
                    CTParser => src(
                        "onepass.jl",
                    ),
                ],
                exclude=EXCLUDE_SYMBOLS,
                public=false,
                private=true,
                title="Onepass",
                title_in_menu="Onepass",
                filename="onepass",
            ),
            CTBase.automatic_reference_documentation(
                subdirectory=".",
                primary_modules=[
                    CTParser => src(
                        "initial_guess.jl",
                    ),
                ],
                exclude=EXCLUDE_SYMBOLS,
                public=false,
                private=true,
                title="Initial Guess",
                title_in_menu="Initial Guess",
                filename="initial_guess",
            ),
        ],
    ],
)

deploydocs(; repo=repo_url * ".git", devbranch="main")
