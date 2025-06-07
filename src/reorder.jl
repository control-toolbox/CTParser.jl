function store!(data, e)
    # we assume data is a dict with keys:
    # variable, time, state, control, alias, misc, objective
    # for each key, you have a vector of Expr already initialised
    @match e begin
        :(PRAGMA($a)) => push!(data[:misc], e)
        :($a = $b) => push!(data[:alias], e)
        :($a, variable) => push!(data[:variable], e)
        :($a, time) => push!(data[:time], e)
        :($a, state) => push!(data[:state], e)
        :($a, control) => push!(data[:control], e)
        :($a → max) => push!(data[:objective], e)
        :($a → min) => push!(data[:objective], e)
        _ => begin
            if e isa LineNumberNode
                nothing
            elseif e isa Expr && e.head == :block
                Expr(:block, map(e -> store!(data, e), e.args)...)
                # !!! assumes that map is done sequentially for side effects on p
            else
                push!(data[:misc], e)
            end
        end
    end
    return nothing
end

function order(data::Dict)
    # we assume data is a dict with keys:
    # variable, time, state, control, alias, misc, objective
    # for each key, you have a vector of Expr already initialised
    code = :()
    keys = [:variable, :time, :state, :control, :alias, :misc, :objective]
    for key ∈ keys
        for e ∈ data[key]
            code = code==:() ? e : concat(code, e)
        end
    end
    return code
end

function order(e::Expr)
    data = Dict(
        :variable => Vector{Expr}(),
        :time => Vector{Expr}(),
        :state => Vector{Expr}(),
        :control => Vector{Expr}(),
        :objective => Vector{Expr}(),
        :alias => Vector{Expr}(),
        :misc => Vector{Expr}(),
    )
    store!(data, e)
    return order(data)
end