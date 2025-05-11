using Test
using Aqua
import CTParser: CTParser, subs, subs2, subs3, subs4, replace_call, has, constraint_type, parsing_backend!, prefix!, e_prefix!, @def
import CTBase: CTBase, ParsingError
import CTModels: CTModels, initial_time, final_time, time_name, variable_dimension, variable_components, variable_name, state_dimension, state_components, state_name, control_dimension, control_components, control_name, constraint, dynamics, mayer, lagrange, criterion, Model
import ExaModels
using MadNLP
using MadNLPGPU
using CUDA
using KernelAbstractions
using BenchmarkTools
using Interpolations

#
@testset verbose = true showtiming = true "CTParser tests" begin
	for name in (:aqua, :utils, :onepass_fun, :onepass_exa)
	#for name in (:aqua, :utils, :onepass_exa) # debug
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end