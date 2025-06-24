function store!(data, e)
    # assume data is a dict with keys:
    # variable, declaration, misc, objective
    # for each key, you have a vector of Expr already initialised
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

function reorder(data::Dict)
    # assume data is a dict with keys:
    # variable, declaration, misc, objective
    # for each key, you have a vector of Expr already initialised
    code = Expr(:block)
    keys = [:variable, :declaration, :misc, :objective]
    for key ∈ keys
        for e ∈ data[key]
            code = code==Expr(:block) ? e : concat(code, e)
        end
    end
    return code
end

function reorder(e::Expr)
    data = Dict(
        :variable    => Vector{Expr}(),
        :declaration => Vector{Expr}(),
        :misc        => Vector{Expr}(),
        :objective   => Vector{Expr}(),
    )
    store!(data, e)
    return reorder(data)
end