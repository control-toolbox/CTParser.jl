"""
[`CTParser`](@ref) module.

Lists all the imported modules and packages:

$(IMPORTS)

List of all the exported names:

$(EXPORTS)
"""
module CTParser

# imports
using CTBase: CTBase # indices utilities and ParsingError
using DocStringExtensions
using MLStyle
using OrderedCollections
using Parameters # @with_kw: to have default values in struct
using Unicode

# sources
include("defaults.jl")
include("utils.jl")
include("onepass.jl")

end
