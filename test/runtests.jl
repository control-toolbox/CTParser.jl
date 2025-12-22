using Test
using Aqua
using OrderedCollections: OrderedDict
import CTParser:
    CTParser,
    subs,
    subs2,
    subs2m,
    subs3,
    replace_call,
    has,
    concat,
    constraint_type,
    @def,
    @init,
    prefix_fun,
    prefix_fun!,
    prefix_exa,
    prefix_exa!,
    e_prefix,
    e_prefix!,
    activate_backend,
    deactivate_backend,
    is_active_backend,
    @def_exa # todo: remove __default... (as soon as discretise_exa has been moved to CTDirect)
import CTBase: CTBase, ParsingError, UnauthorizedCall
import CTModels:
    CTModels,
    initial_time,
    final_time,
    time_name,
    variable_dimension,
    variable_components,
    variable_name,
    state_dimension,
    state_components,
    state_name,
    control_dimension,
    control_components,
    control_name,
    constraint,
    dynamics,
    mayer,
    lagrange,
    criterion,
    Model,
    get_build_examodel
using ExaModels: ExaModels, AbstractNode
using MadNLP
using MadNLP
using CUDA
using BenchmarkTools
using Interpolations
using NLPModels
using LinearAlgebra: LinearAlgebra, dot, norm, norm_sqr 

macro ignore(e)
    return :()
end

const VERBOSE = true
const SHOWTIMING = true

"""Return the default set of tests enabled for CTParser."""
function default_tests()
    return OrderedDict(
        :aqua => true,
        :initial_guess => true,
        :utils => true,
        :utils_bis => true,
        :prefix => true,
        :prefix_bis => true,
        :onepass_fun => true,
        :onepass_fun_bis => true,
        :onepass_exa => true,
        :onepass_exa_bis => true,
        :exa_linalg => true,
    )
end

const TEST_SELECTIONS = isempty(ARGS) ? Symbol[] : Symbol.(ARGS)

function selected_tests()
    tests = default_tests()
    sels = TEST_SELECTIONS

    # No selection: use defaults
    if isempty(sels)
        return tests
    end

    # Single :all selection: enable everything
    if length(sels) == 1 && sels[1] == :all
        for k in keys(tests)
            tests[k] = true
        end
        return tests
    end

    # Otherwise, start with everything disabled
    for k in keys(tests)
        tests[k] = false
    end

    # Enable explicit selections
    for sel in sels
        if sel == :all
            for k in keys(tests)
                tests[k] = true
            end
            break
        end
        if haskey(tests, sel)
            tests[sel] = true
        end
    end

    return tests
end

const SELECTED_TESTS = selected_tests()

@testset verbose = VERBOSE showtiming = SHOWTIMING "CTParser tests" begin
    for (name, enabled) in SELECTED_TESTS
        enabled || continue
        @testset "$(name)" begin
            test_name = Symbol(:test_, name)
            include("$(test_name).jl")
            @eval $test_name()
        end
    end
end
