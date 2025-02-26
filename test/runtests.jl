using CTParser: CTParser, subs, replace_call, has, constraint_type, @def # some utils are named for testing
using Test
using Aqua

include("temporary1.jl") # todo: remove

#
@testset verbose = true showtiming = true "CTParser tests" begin
	for name in (:aqua, :utils, :onepass)
	#for name in (:utils,)
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end

include("temporary2.jl") # todo: remove