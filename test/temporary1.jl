# temporary.jl

using Pkg
Pkg.add(url="https://github.com/control-toolbox/CTModels.jl")

module OptimalControl

import CTModels

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

end