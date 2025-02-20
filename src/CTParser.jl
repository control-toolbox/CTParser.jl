"""
[`CTParser`](@ref) module.

Lists all the imported modules and packages:

$(IMPORTS)

List of all the exported names:

$(EXPORTS)
"""
module CTParser

# imports
import CTBase
using DocStringExtensions
using MLStyle
using OrderedCollections
using Parameters # @with_kw: to have default values in struct
using Unicode

# exports
export @def

# sources
include("utils.jl")
include("onepass.jl")

end