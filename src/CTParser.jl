module CTParser

# imports
import CTBase
using DocStringExtensions
using MLStyle
using OrderedCollections
using Parameters # @with_kw: to have default values in struct

# sources
include("utils.jl")
include("onepass.jl")

end