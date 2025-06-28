using Revise
using Pkg
Pkg.activate(".")

#using CTParser

function standardise_declaration(e::Expr)
    @match e begin
        :(PRAGMA($a))   => push!(data[:misc], e)
        :($a = $b)      => push!(data[:declaration], e)
        :($a, variable) => push!(data[:variable], e)
        :($a, time)     => push!(data[:declaration], e)
        :($a, state)    => push!(data[:declaration], e)
        :($a, control)  => push!(data[:declaration], e)
        :($a → max)     => push!(data[:objective], e)
        :($a → min)     => push!(data[:objective], e)
        _ => begin
            if e isa LineNumberNode
                nothing
            elseif e isa Expr && e.head == :block
                Expr(:block, map(e -> store!(data, e), e.args)...)
                # !!! assumes that map is done sequentially for side effects on data
            else
                push!(data[:misc], e)
            end
        end
    end
    return nothing
end