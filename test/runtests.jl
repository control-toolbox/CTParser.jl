using CTParser: CTParser, subs, replace_call, has, constraint_type, @def 
using Test
using Aqua
using Pkg # debug: also remove Pkg from test/Project.toml
Pkg.add(url="https://github.com/control-toolbox/CTModels.jl") # debug
import CTModels # debug: should eventually be OptimalControl (that will reexport CTModels func. primitives)

#
@testset verbose = true showtiming = true "CTParser tests" begin
	for name in (:aqua, :utils)
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end

Pkg.rm("CTModels") # debug