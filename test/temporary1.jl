# temporary.jl

using Pkg
Pkg.add(url="https://github.com/control-toolbox/CTModels.jl", rev="main")

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
        return n == 1 ? r[1] : r
    end
    return isnothing(f!) ? nothing : f
end

__dynamics(ocp) = to_out_of_place(dynamics(ocp), state_dimension(ocp))

function __constraint(ocp, label)
    n = length(constraint(ocp,label)[3]) # Size of lb 
    return to_out_of_place(constraint(ocp, label)[2], n)
end

end