using Test
using Aqua
using CTParser

#
@testset verbose = true showtiming = true "CTParser tests" begin
	for name in (:default, ) #(:aqua, :default)
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end
