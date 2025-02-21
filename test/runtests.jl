using CTParser: CTParser, subs, replace_call, has, constraint_type, @def 
using Test
using Aqua

include("temporary1.jl") # debug

#
@testset verbose = true showtiming = true "CTParser tests" begin
	for name in (:aqua, :utils)
	#for name in (:utils,)
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end

include("temporary2.jl") # debug