# temporary1.jl

module OptimalControl

import CTModels

# types

PreModel = CTModels.PreModel
Model = CTModels.Model

# exceptions

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
build_model = CTModels.build_model

# getters

initial_time = CTModels.initial_time
final_time = CTModels.final_time
time_name = CTModels.time_name
variable_dimension = CTModels.variable_dimension
variable_components = CTModels.variable_components
variable_name = CTModels.variable_name
state_dimension = CTModels.state_dimension
state_components = CTModels.state_components
state_name = CTModels.state_name
control_dimension = CTModels.control_dimension
control_components = CTModels.control_components
control_name = CTModels.control_name
constraint = CTModels.constraint
dynamics = CTModels.dynamics
mayer = CTModels.mayer
lagrange = CTModels.lagrange
criterion = CTModels.criterion

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
        f = (x0, xf, v) -> v[c] # debug: :variable_range gives a :boundary, not a :path (check on CTModels)
    else
        throw("Unknow constraint type")
    end
    return f
end

end