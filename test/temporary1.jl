# temporary.jl

using Pkg
Pkg.add(url="https://github.com/control-toolbox/CTModels.jl", rev="main") # debug: replace (still temporarily) by CTModels from gen. registry

module OptimalControl

import CTModels

struct ParsingError <: Exception
    var::String
end
Base.showerror(io::IO, e::ParsingError) = print(io, "ParsingError: ", e.var)

# setters

variable! = CTModels.variable!
time! = CTModels.time!
state! = CTModels.state!
control! = CTModels.control!
dynamics! = CTModels.dynamics!
constraint! = CTModels.constraint!
objective! = CTModels.objective!
definition! = CTModels.definition!
PreModel = CTModels.PreModel
build_model = CTModels.build_model

# getters

initial_time = CTModels.initial_time
final_time = CTModels.final_time
variable_dimension = CTModels.variable_dimension
state_dimension = CTModels.state_dimension
control_dimension = CTModels.control_dimension
constraint = CTModels.constraint
dynamics = CTModels.dynamics
mayer = CTModels.mayer
lagrange = CTModels.lagrange

function to_out_of_place(f!, n; T=Float64)
    function f(args...; kwargs...)
        r = zeros(T, n)
        f!(r, args...; kwargs...)
        #return n == 1 ? r[1] : r
        return r # everything is now a vector
    end
    return isnothing(f!) ? nothing : f
end

__dynamics(ocp) = to_out_of_place(dynamics(ocp), state_dimension(ocp))

function __constraint(ocp, label)
    type = constraint(ocp,label)[1]
    c = constraint(ocp,label)[2]
    n = length(constraint(ocp,label)[3]) # Size of lb 
    m = length(constraint(ocp,label)[4]) # Size of ub 
    @assert(n == m)
    if type in [:boundary, :path]
        f = to_out_of_place(c, n)
    elseif type == :state
        f = (t, x, u, v) -> x[c]
    elseif type == :control
        f = (t, x, u, v) -> u[c]
    elseif type == :variable
        f = (t, x, u, v) -> v[c]
    else
        throw("Unknow constraint type")
    end
    return f
end

end
