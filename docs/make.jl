using Documenter
using CTParser

# to add docstrings from external packages
#Modules = [Plots]
#for Module in Modules
#    isnothing(DocMeta.getdocmeta(Module, :DocTestSetup)) &&
#        DocMeta.setdocmeta!(Module, :DocTestSetup, :(using $Module); recursive=true)
#end

repo_url = "github.com/control-toolbox/CTParser.jl"

makedocs(;
    remotes=nothing,
    warnonly=[:cross_references, :autodocs_block],
    sitename="CTParser.jl",
    format=Documenter.HTML(;
        repolink="https://" * repo_url,
        prettyurls=false,
        size_threshold_ignore=["dev.md"],
        assets=[
            asset("https://control-toolbox.org/assets/css/documentation.css"),
            asset("https://control-toolbox.org/assets/js/documentation.js"),
        ],
    ),
    pages=["Introduction" => "index.md", "API" => "dev.md"],
    checkdocs=:none,
)

deploydocs(; repo=repo_url * ".git", devbranch="main")
