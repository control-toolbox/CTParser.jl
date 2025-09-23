using Test
using Aqua
import CTParser: CTParser, subs, subs2, subs3, subs4, replace_call, has, concat, constraint_type, @def, prefix_fun, prefix_fun!, prefix_exa, prefix_exa!, e_prefix, e_prefix!, activate_backend, deactivate_backend, is_active_backend, @def_exa # todo: remove __default... (as soon as discretise_exa has been moved to CTDirect)
import CTBase: CTBase, ParsingError, UnauthorizedCall
import CTModels: CTModels, initial_time, final_time, time_name, variable_dimension, variable_components, variable_name, state_dimension, state_components, state_name, control_dimension, control_components, control_name, constraint, dynamics, mayer, lagrange, criterion, Model, get_build_examodel
import ExaModels
using MadNLP
using MadNLPGPU
using CUDA
using BenchmarkTools
using Interpolations

macro ignore(e) return :() end

@testset verbose = true showtiming = true "CTParser tests" begin
    #for name ∈ (:aqua, :utils, :prefix, :onepass_fun, :onepass_exa)
    for name ∈ (:onepass_exa,) # debug
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end
