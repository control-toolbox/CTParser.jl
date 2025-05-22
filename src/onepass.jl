# onepass
# todo: check todo's
# - as_range / as_vector for rg / lb, ub could be done here in p_constraint when calling PREFIX.constraint!
# - cannot call solve if problem not fully defined (dynamics not defined...)
# - doc: explain projections wrt to t0, tf, t; (...x1...x2...)(t) -> ...gensym1...gensym2... (most internal first)
# - additional checks: when generating functions (constraints, dynamics, costs), there should not be any x or u left
#   (but the user might indeed do so); meaning that has(ee, x/u/t) must be false (postcondition)
# - tests exceptions (parsing and semantics/runtime)
# - add assert for pre/post conditions and invariants
# - add tests on ParsingError + run time errors (wrapped in try ... catch's - use string to be precise)
# - currently "t ∈ [ 0+0, 1 ], time" is allowed, and compels to declare "x(0+0) == ..."
# - exa: generated function forbids to use kwarg names (grid_size...) in expressions; could either (i) replace such names in the user code (e.g. prefixing by a gensym...), but not so readable; (ii) throw an explicit error ("grid_size etc. are reserved names")
# - exa: x = ExaModels.variable($p_ocp, $n / 1:$n...) ?
# - exa: what about expressions with x(t), not indexed but used as a scalar? should be x[:, ...] in ExaModels? does it occur (sum(x(t)...))
# - iterators i and j (cf. dyn / lagrange): gensym's!
# todo: test all constraints bounds (== 1, and 5 ranges)

# Defaults

__default_parsing_backend() = 1 # :fun
__default_scheme_exa() = :trapezoidal
__default_grid_size_exa() = 200
__default_backend_exa() = nothing
__default_init_exa() = (0.1, 0.1, 0.1) # default init for v, x, u
__default_base_type_exa() = Float64 

# Modules vars 

const PARSING_BACKENDS = (:fun, :exa) # known parsing backends
const PARSING_BACKEND = Ref(__default_parsing_backend())

parsing_backend() = PARSING_BACKENDS[PARSING_BACKEND[]]

function parsing_backend!(b)
    b ∈ PARSING_BACKENDS || throw("unknown parsing backend")
    PARSING_BACKEND[] = findall(x -> x == b, PARSING_BACKENDS)[1] 
    return nothing
end

const PREFIX = Ref(:OptimalControl) # prefix for generated code, assumed to be evaluated within OptimalControl.jl; can be CTModel for tests

prefix() = PREFIX[]

function prefix!(p)
    PREFIX[] = p
    return nothing
end

const E_PREFIX = Ref(:OptimalControl) # prefix for exceptions in generated code, assumed to be evaluated within OptimalControl.jl; can be CTBase for tests

e_prefix() = E_PREFIX[]

function e_prefix!(p)
    E_PREFIX[] = p
    return nothing
end

# Utils

__symgen(s...) = Symbol(s..., gensym())

"""
$(TYPEDEF)

**Fields**

"""
@with_kw mutable struct ParsingInfo
    v::Symbol = __symgen(:unset) # not nothing as, if unset, this name is still used in function args
    t::Union{Symbol,Nothing} = nothing
    t0::Union{Real,Symbol,Expr,Nothing} = nothing
    tf::Union{Real,Symbol,Expr,Nothing} = nothing
    x::Union{Symbol,Nothing} = nothing
    u::Union{Symbol,Nothing} = nothing
    dim_v::Union{Integer, Symbol, Expr, Nothing} = nothing
    dim_x::Union{Integer, Symbol, Expr, Nothing} = nothing
    dim_u::Union{Integer, Symbol, Expr, Nothing} = nothing
    aliases::OrderedDict{Union{Symbol,Expr},Union{Real,Symbol,Expr}} = __init_aliases() # Dict ordered by Symbols *and Expr* just for scalar variable / state / control
    lnum::Int = 0
    line::String = ""
    dt::Symbol = __symgen(:dt)
    dyn_coords::Vector{Int64} = Int64[]
    l_v::Symbol = __symgen(:l_v)
    u_v::Symbol = __symgen(:u_v)
    box_v::Expr = :(LineNumberNode(0, "box constraints: variable"))
    l_x::Symbol = __symgen(:l_x)
    u_x::Symbol = __symgen(:u_x)
    box_x::Expr = :(LineNumberNode(0, "box constraints: state"))
    l_u::Symbol = __symgen(:l_u)
    u_u::Symbol = __symgen(:u_u)
    box_u::Expr = :(LineNumberNode(0, "box constraints: control"))
end

__init_aliases(; max_dim=20) = begin
    al = OrderedDict{Symbol,Union{Real,Symbol,Expr}}()
    for i in 1:max_dim
        al[Symbol(:R, CTBase.ctupperscripts(i))] = :(R^$i)
    end
    al[:<=] = :≤
    al[:>=] = :≥
    al[:derivative] = :∂
    al[:integral] = :∫
    al[:(=>)] = :→
    al[:in] = :∈
    return al
end

# for static (= not in generated code evaluated at run time) errors; should only appear in syntactic analysis
__throw(mess, n, line) = begin
    e_pref = e_prefix()
    info = string("\nLine ", n, ": ", line, "\n", mess)
    return :(throw($e_pref.ParsingError($info))) 
end

__wrap(e, n, line) = quote
    local ex
    try
        $e
    catch ex
        println("Line ", $n, ": ", $line)
        throw(ex)
    end
end

is_range(x) = false 
is_range(x::T) where {T <: AbstractRange} = true 
is_range(x::Expr) = (x.head == :call) && (x.args[1] == :(:))
as_range(x) = is_range(x) ? x : [x] 

# Main code

"""
$(TYPEDSIGNATURES)

Parse the expression `e` and update the `ParsingInfo` structure `p`.

# Example
```@example
parse!(p, :p_ocp, :(v ∈ R, variable))
```
"""
parse!(p, p_ocp, e; log=false) = begin
    #
    p.lnum = p.lnum + 1
    p.line = string(e)
    for a in keys(p.aliases)
        e = subs(e, a, p.aliases[a])
    end
    #
    @match e begin
        # PRAGMA
        :(PRAGMA($e)) => p_pragma!(p, p_ocp, e; log)
        # aliases
        :($a = $e1) => @match e1 begin
            :(($names) ∈ R^$q, variable) =>
                p_variable!(p, p_ocp, a, q; components_names=names, log)
            :([$names] ∈ R^$q, variable) =>
                p_variable!(p, p_ocp, a, q; components_names=names, log)
            :(($names) ∈ R^$n, state) =>
                p_state!(p, p_ocp, a, n; components_names=names, log)
            :([$names] ∈ R^$n, state) =>
                p_state!(p, p_ocp, a, n; components_names=names, log)
            :(($names) ∈ R^$m, control) =>
                p_control!(p, p_ocp, a, m; components_names=names, log)
            :([$names] ∈ R^$m, control) =>
                p_control!(p, p_ocp, a, m; components_names=names, log)
            _ => p_alias!(p, p_ocp, a, e1; log) # alias
        end
        # variable                    
        :($v ∈ R^$q, variable) => p_variable!(p, p_ocp, v, q; log)
        :($v ∈ R, variable) => p_variable!(p, p_ocp, v, 1; log)
        # time                        
        :($t ∈ [$t0, $tf], time) => p_time!(p, p_ocp, t, t0, tf; log)
        # state                       
        :($x ∈ R^$n, state) => p_state!(p, p_ocp, x, n; log)
        :($x ∈ R, state) => p_state!(p, p_ocp, x, 1; log)
        # control                     
        :($u ∈ R^$m, control) => p_control!(p, p_ocp, u, m; log)
        :($u ∈ R, control) => p_control!(p, p_ocp, u, 1; log)
        # dynamics                    
        :(∂($x[$i])($t) == $e1) => p_dynamics_coord!(p, p_ocp, x, i, t, e1; log) # must be filtered before ∂($x)($t) pattern
        :(∂($x[$i])($t) == $e1, $label) => p_dynamics_coord!(p, p_ocp, x, i, t, e1, label; log)
        :(∂($x)($t) == $e1) => p_dynamics!(p, p_ocp, x, t, e1; log)
        :(∂($x)($t) == $e1, $label) => p_dynamics!(p, p_ocp, x, t, e1, label; log)
        # constraints                 
        :($e1 == $e2) => p_constraint!(p, p_ocp, e2, e1, e2; log)
        :($e1 == $e2, $label) => p_constraint!(p, p_ocp, e2, e1, e2, label; log)
        :($e1 ≤ $e2 ≤ $e3) => p_constraint!(p, p_ocp, e1, e2, e3; log)
        :($e1 ≤ $e2 ≤ $e3, $label) => p_constraint!(p, p_ocp, e1, e2, e3, label; log)
        :($e2 ≤ $e3) => p_constraint!(p, p_ocp, nothing, e2, e3; log)
        :($e2 ≤ $e3, $label) => p_constraint!(p, p_ocp, nothing, e2, e3, label; log)
        :($e3 ≥ $e2 ≥ $e1) => p_constraint!(p, p_ocp, e1, e2, e3; log)
        :($e3 ≥ $e2 ≥ $e1, $label) => p_constraint!(p, p_ocp, e1, e2, e3, label; log)
        :($e2 ≥ $e1) => p_constraint!(p, p_ocp, e1, e2, nothing; log)
        :($e2 ≥ $e1, $label) => p_constraint!(p, p_ocp, e1, e2, nothing, label; log)
        # lagrange cost
        :(∫($e1) → min) => p_lagrange!(p, p_ocp, e1, :min; log)
        :(-∫($e1) → min) => p_lagrange!(p, p_ocp, :(-$e1), :min; log)
        :($e1 * ∫($e2) → min) => if has(e1, p.t) # this test (and those similar below) is here to allow reduction to p_lagrange! standard call
            (return __throw("time $(p.t) must not appear in $e1", p.lnum, p.line)) 
        else
            p_lagrange!(p, p_ocp, :($e1 * $e2), :min; log)
        end
        :(∫($e1) → max) => p_lagrange!(p, p_ocp, e1, :max; log)
        :(-∫($e1) → max) => p_lagrange!(p, p_ocp, :(-$e1), :max; log)
        :($e1 * ∫($e2) → max) => if has(e1, p.t)
            (return __throw("time $(p.t) must not appear in $e1", p.lnum, p.line))
        else
            p_lagrange!(p, p_ocp, :($e1 * $e2), :max; log)
        end
        # bolza cost
        :($e1 + ∫($e2) → min) => p_bolza!(p, p_ocp, e1, e2, :min; log)
        :($e1 + $e2 * ∫($e3) → min) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :($e2 * $e3), :min; log)
        end
        :($e1 - ∫($e2) → min) => p_bolza!(p, p_ocp, e1, :(-$e2), :min; log)
        :($e1 - $e2 * ∫($e3) → min) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :(-$e2 * $e3), :min; log)
        end
        :($e1 + ∫($e2) → max) => p_bolza!(p, p_ocp, e1, e2, :max; log)
        :($e1 + $e2 * ∫($e3) → max) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :($e2 * $e3), :max; log)
        end
        :($e1 - ∫($e2) → max) => p_bolza!(p, p_ocp, e1, :(-$e2), :max; log)
        :($e1 - $e2 * ∫($e3) → max) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :(-$e2 * $e3), :max; log)
        end
        :(∫($e2) + $e1 → min) => p_bolza!(p, p_ocp, e1, e2, :min; log)
        :($e2 * ∫($e3) + $e1 → min) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :($e2 * $e3), :min; log)
        end
        :(∫($e2) - $e1 → min) => p_bolza!(p, p_ocp, :(-$e1), e2, :min; log)
        :($e2 * ∫($e3) - $e1 → min) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, :(-$e1), :($e2 * $e3), :min; log)
        end
        :(∫($e2) + $e1 → max) => p_bolza!(p, p_ocp, e1, e2, :max; log)
        :($e2 * ∫($e3) + $e1 → max) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, e1, :($e2 * $e3), :max; log)
        end
        :(∫($e2) - $e1 → max) => p_bolza!(p, p_ocp, :(-$e1), e2, :max; log)
        :($e2 * ∫($e3) - $e1 → max) => if has(e2, p.t)
            (return __throw("time $(p.t) must not appear in $e2", p.lnum, p.line))
        else
            p_bolza!(p, p_ocp, :(-$e1), :($e2 * $e3), :max; log)
        end
        # mayer cost
        :($e1 → min) => p_mayer!(p, p_ocp, e1, :min; log)
        :($e1 → max) => p_mayer!(p, p_ocp, e1, :max; log)
        #
        _ => begin
            if e isa LineNumberNode
                p.lnum = p.lnum - 1
                e
            elseif e isa Expr && e.head == :block
                p.lnum = p.lnum - 1
                Expr(:block, map(e -> parse!(p, p_ocp, e; log), e.args)...)
                # !!! assumes that map is done sequentially for side effects on p
            else
                return __throw("unknown syntax", p.lnum, p.line)
            end
        end
    end
end

function p_pragma!(p, p_ocp, e; log=false)
    log && println("PRAGMA: $e")
    return parsing(:pragma)(p, p_ocp, e)
end

function p_pragma_fun!(p, p_ocp, e)
    return __throw("PRAGMA not allowed", p.lnum, p.line)
end

function p_pragma_exa!(p, p_ocp, e)
    code = e
    return __wrap(code, p.lnum, p.line)
end

function p_alias!(p, p_ocp, a, e; log=false)
    log && println("alias: $a = $e")
    a isa Symbol || return __throw("forbidden alias name: $a", p.lnum, p.line)
    p.aliases[a] = e
    return parsing(:alias)(p, p_ocp, a, e)
end
    
function p_alias_fun!(p, p_ocp, a, e)
    aa = QuoteNode(a)
    ee = QuoteNode(e)
    code = :(LineNumberNode(0, "alias: " * string($aa) * " = " * string($ee)))
    return __wrap(code, p.lnum, p.line)
end

p_alias_exa! = p_alias_fun!

function p_variable!(p, p_ocp, v, q; components_names=nothing, log=false)
    log && println("variable: $v, dim: $q")
    v isa Symbol || return __throw("forbidden variable name: $v", p.lnum, p.line)
    vv = QuoteNode(v)
    if q == 1
        vg = __symgen(v)
        p.aliases[:($v[1])] = :($vg[1]) # case for which the Dict of aliases can be indexed by Expr, not only Symbol; avoids (otherwise harmless) vg[1][1] that will not be accepted for t0 or tf (same for state and control)
        p.aliases[v] = :($vg[1])
        p.aliases[Symbol(v, CTBase.ctindices(1))] = :($vg[1])
        p.aliases[Symbol(v, 1)] = :($vg[1])
        v = vg 
    end
    p.v = v
    p.dim_v = q
    qq = q isa Int ? q : 9
    for i in 1:qq
        p.aliases[Symbol(v, CTBase.ctindices(i))] = :($v[$i])
    end # make v₁, v₂... if the variable is named v
    for i in 1:qq
        p.aliases[Symbol(v, i)] = :($v[$i])
    end # make v1, v2... if the variable is named v
    if !isnothing(components_names)
        qq == length(components_names.args) ||
            return __throw("the number of variable components must be $qq", p.lnum, p.line)
        for i in 1:qq
            p.aliases[components_names.args[i]] = :($v[$i])
        end # aliases from names given by the user
    end
    return parsing(:variable)(p, p_ocp, v, q, vv; components_names=components_names)
end

function p_variable_fun!(p, p_ocp, v, q, vv; components_names=nothing)
    pref = prefix()
    if isnothing(components_names)
        code = :($pref.variable!($p_ocp, $q, $vv))
    else
        ss = QuoteNode(string.(components_names.args))
        code = :($pref.variable!($p_ocp, $q, $vv, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_variable_exa!(p, p_ocp, v, q, vv; components_names=nothing)
    code_box = :($(p.l_v) = -Inf * ones($q); $(p.u_v) = Inf * ones($q))
    p.box_v = concat(p.box_v, code_box)
    code = :(ExaModels.variable($p_ocp, $q; lvar = $(p.l_v), uvar = $(p.u_v), start = init[1]))
    code = __wrap(code, p.lnum, p.line)
    code = :($v = $code) # affectation must be done outside try ... catch )
    return code
end

function p_time!(p, p_ocp, t, t0, tf; log=false)
    log && println("time: $t, initial time: $t0, final time: $tf")
    t isa Symbol || return __throw("forbidden time name: $t", p.lnum, p.line)
    p.t = t
    p.t0 = t0
    p.tf = tf
    return parsing(:time)(p, p_ocp, t, t0, tf)
end

function p_time_fun!(p, p_ocp, t, t0, tf)
    pref = prefix()
    tt = QuoteNode(t)
    # todo: scalar var aliased to v[1], so some cases below are excluded; move solme part to syntactic pass
    code = @match (has(t0, p.v), has(tf, p.v)) begin
        (false, false) => :($pref.time!($p_ocp; t0=$t0, tf=$tf, time_name=$tt))
        (true, false) => @match t0 begin
            :($v1[$i]) && if (v1 == p.v) end => :($pref.time!($p_ocp; ind0=$i, tf=$tf, time_name=$tt))
            :($v1) && if (v1 == p.v) &&  (p.dim_v == 1) end => :( $pref.time!($p_ocp; ind0=1, tf=$tf, time_name=$tt) ) # todo: never executed (check!)
            :($v1) && if (v1 == p.v) && !(p.dim_v == 1) end => return __throw("variable must be of dimension one for a time", p.lnum, p.line) # todo: v1 immplies dim var > 1 => necessary error
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        (false, true) => @match tf begin
            :($v1[$i]) && if (v1 == p.v) end => :($pref.time!($p_ocp; t0=$t0, indf=$i, time_name=$tt))
            :($v1) && if (v1 == p.v) &&  (p.dim_v == 1) end => :( $pref.time!($p_ocp; t0=$t0, indf=1, time_name=$tt) ) # todo: never executed (check!)
            :($v1) && if (v1 == p.v) && !(p.dim_v ==1) end => return __throw("variable must be of dimension one for a time", p.lnum, p.line) # todo: move above in common syntactic pass
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        _ => @match (t0, tf) begin
            (:($v1[$i]), :($v2[$j])) && if (v1 == v2 == p.v)
            end => :($pref.time!($p_ocp; ind0=$i, indf=$j, time_name=$tt))
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
    end
    return __wrap(code, p.lnum, p.line)
end

function p_time_exa!(p, p_ocp, t, t0, tf)
    @match (has(t0, p.v), has(tf, p.v)) begin
        (false, false) => nothing
        (true, false) => @match t0 begin # todo: should be factored in syntactic pass, see todo's above for :fun
            :($v1[$i]) && if (v1 == p.v) end => nothing 
            :($v1) && if (v1 == p.v) && !(p.dim_v == 1) end => return __throw("variable must be of dimension one for a time", p.lnum, p.line)
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        (false, true) => @match tf begin
            :($v1[$i]) && if (v1 == p.v) end => nothing
            :($v1) && if (v1 == p.v) && !(p.dim_v == 1) end => return __throw("variable must be of dimension one for a time", p.lnum, p.line)
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        _ => @match (t0, tf) begin
            (:($v1[$i]), :($v2[$j])) && if (v1 == v2 == p.v) end => nothing
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
    end
    code = :(($tf - $t0) / grid_size)
    code = __wrap(code, p.lnum, p.line)
    code = :($(p.dt) = $code) # affectation must be done outside try ... catch 
    return code
end

function p_state!(p, p_ocp, x, n; components_names=nothing, log=false)
    log && println("state: $x, dim: $n")
    x isa Symbol || return __throw("forbidden state name: $x", p.lnum, p.line)
    p.aliases[Symbol(Unicode.normalize(string(x, "̇")))] = :(∂($x))
    xx = QuoteNode(x)
    if n == 1
        xg = __symgen(x)
        p.aliases[:($x[1])] = :($xg[1]) # not compulsory as xg[1][1] is ok
        p.aliases[x] = :($xg[1])
        p.aliases[Symbol(x, CTBase.ctindices(1))] = :($xg[1])
        p.aliases[Symbol(x, 1)] = :($xg[1])
        x = xg 
    end
    p.x = x
    p.dim_x = n
    nn = n isa Int ? n : 9
    for i in 1:nn
        p.aliases[Symbol(x, CTBase.ctindices(i))] = :($x[$i])
    end # make x₁, x₂... if the state is named x
    for i in 1:nn
        p.aliases[Symbol(x, i)] = :($x[$i])
    end # make x1, x2... if the state is named x
    if !isnothing(components_names)
        nn == length(components_names.args) ||
            return __throw("the number of state components must be $nn", p.lnum, p.line)
        for i in 1:nn
            p.aliases[components_names.args[i]] = :($x[$i])
            # todo: in future, add aliases for state components (scalar) derivatives, i.e. alias ẋ₁(t), ẋ1(t) to ∂(x[1])(t)
        end
    end
    return parsing(:state)(p, p_ocp, x, n, xx; components_names=components_names)
end
    
function p_state_fun!(p, p_ocp, x, n, xx; components_names=nothing)
    pref = prefix()
    if isnothing(components_names)
        code = :($pref.state!($p_ocp, $n, $xx))
    else
        ss = QuoteNode(string.(components_names.args))
        code = :($pref.state!($p_ocp, $n, $xx, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_state_exa!(p, p_ocp, x, n, xx; components_names=nothing)
    code_box = :($(p.l_x) = -Inf * ones($n); $(p.u_x) = Inf * ones($n))
    p.box_x = concat(p.box_x, code_box)
    code = :(ExaModels.variable($p_ocp, $n, 0:grid_size; lvar = [$(p.l_x)[i] for (i, j) ∈ Base.product(1:$n, 0:grid_size)], uvar = [$(p.u_x)[i] for (i, j) ∈ Base.product(1:$n, 0:grid_size)], start = init[2]))
    code = __wrap(code, p.lnum, p.line)
    code = :($x = $code) # affectation must be done outside try ... catch )
    return code
end

function p_control!(p, p_ocp, u, m; components_names=nothing, log=false)
    log && println("control: $u, dim: $m")
    u isa Symbol || return __throw("forbidden control name: $u", p.lnum, p.line)
    uu = QuoteNode(u)
    if m == 1
        ug = __symgen(u)
        p.aliases[:($u[1])] = :($ug[1]) # not compulsory as ug[1][1] is ok
        p.aliases[u] = :($ug[1])
        p.aliases[Symbol(u, CTBase.ctindices(1))] = :($ug[1])
        p.aliases[Symbol(u, 1)] = :($ug[1])
        u = ug 
    end
    p.u = u
    p.dim_u = m
    mm = m isa Int ? m : 9
    for i in 1:mm
        p.aliases[Symbol(u, CTBase.ctindices(i))] = :($u[$i])
    end # make u₁, u₂... if the control is named u
    for i in 1:mm
        p.aliases[Symbol(u, i)] = :($u[$i])
    end # make u1, u2... if the control is named u
    if !isnothing(components_names)
        mm == length(components_names.args) ||
            return __throw("the number of control components must be $mm", p.lnum, p.line)
        for i in 1:mm
            p.aliases[components_names.args[i]] = :($u[$i])
        end # aliases from names given by the user
    end
    return parsing(:control)(p, p_ocp, u, m, uu; components_names=components_names)
end
    
function p_control_fun!(p, p_ocp, u, m, uu; components_names=nothing)
    pref = prefix()
    if isnothing(components_names)
        code = :($pref.control!($p_ocp, $m, $uu))
    else
        ss = QuoteNode(string.(components_names.args))
        code = :($pref.control!($p_ocp, $m, $uu, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_control_exa!(p, p_ocp, u, m, uu; components_names=nothing)
    code_box = :($(p.l_u) = -Inf * ones($m); $(p.u_u) = Inf * ones($m))
    p.box_u = concat(p.box_u, code_box) 
    code = :(ExaModels.variable($p_ocp, $m, 0:grid_size; lvar = [$(p.l_u)[i] for (i, j) ∈ Base.product(1:$m, 0:grid_size)], uvar = [$(p.u_u)[i] for (i, j) ∈ Base.product(1:$m, 0:grid_size)], start = init[3]))
    code = __wrap(code, p.lnum, p.line)
    code = :($u = $code) # affectation must be done outside try ... catch )
    return code
end

function p_constraint!(p, p_ocp, e1, e2, e3, label = __symgen(:label); log = false)
    c_type = constraint_type(e2, p.t, p.t0, p.tf, p.x, p.u, p.v)
    log && println("constraint ($c_type): $e1 ≤ $e2 ≤ $e3,    ($label)")
    label isa Int && (label = Symbol(:eq, label))
    label isa Symbol || return __throw("forbidden label: $label", p.lnum, p.line)
    return parsing(:constraint)(p, p_ocp, e1, e2, e3, c_type, label)
end

function p_constraint_fun!(p, p_ocp, e1, e2, e3, c_type, label)
    pref = prefix()
    llabel = QuoteNode(label)
    code = @match c_type begin
        :boundary || :variable_fun || (:initial, rg) || (:final, rg) => begin # :initial and :final now treated as boundary
            fun = __symgen(:fun)
            x0 = __symgen(:x0)
            xf = __symgen(:xf)
            r = __symgen(:r)
            ee2 = replace_call(e2, p.x, p.t0, x0)
            ee2 = replace_call(ee2, p.x, p.tf, xf)
            args = [r, x0, xf, p.v]
            quote
                function $fun($(args...))
                    @views $r[:] .= $ee2
                    return nothing
                end
                $pref.constraint!($p_ocp, :boundary; f=$fun, lb=$e1, ub=$e3, label=$llabel)
            end
        end
        (:control_range, rg) =>
            :($pref.constraint!($p_ocp, :control; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        (:state_range, rg) =>
            :($pref.constraint!($p_ocp, :state; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        (:variable_range, rg) =>
            :($pref.constraint!($p_ocp, :variable; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        :state_fun || control_fun || :mixed => begin # now all treated as path
            fun = __symgen(:fun)
            xt = __symgen(:xt)
            ut = __symgen(:ut)
            r = __symgen(:r)
            ee2 = replace_call(e2, [p.x, p.u], p.t, [xt, ut])
            args = [r, p.t, xt, ut, p.v]
            quote
                function $fun($(args...))
                    @views $r[:] .= $ee2
                    return nothing
                end
                $pref.constraint!($p_ocp, :path; f=$fun, lb=$e1, ub=$e3, label=$llabel)
            end
        end
        _ => return __throw("bad constraint declaration", p.lnum, p.line)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_constraint_exa!(p, p_ocp, e1, e2, e3, c_type, label)
    isnothing(e1) && (e1 = :(-Inf * ones(length($e3))))
    isnothing(e3) && (e3 = :( Inf * ones(length($e1))))
    code = @match c_type begin
        :boundary || :variable_fun => begin
            code = :(length($e1) == length($e3) == 1 || throw("this constraint must be scalar")) # (vs. __throw) since raised at runtime
            x0 = __symgen(:x0)
            xf = __symgen(:xf)
            e2 = replace_call(e2, p.x, p.t0, x0)
            e2 = replace_call(e2, p.x, p.tf, xf)
            e2 = subs2(e2, x0, p.x, 0)
            e2 = subs2(e2, xf, p.x, :grid_size)
            concat(code, :(ExaModels.constraint($p_ocp, $e2; lcon = $e1, ucon = $e3)))
        end
        (:initial, rg) => begin
            if isnothing(rg)
                rg = :(1:$(p.dim_x)) # x(t0) implies rg == nothing but means x[1:p.dim_x](t0)
                e2 = subs(e2, p.x, :($(p.x)[$rg]))
            elseif !is_range(rg)
                rg = as_range(rg)
            end
            code = :(length($e1) == length($e3) == length($rg) || throw("wrong bound dimension")) # (vs. __throw) since raised at runtime
            x0 = __symgen(:x0)
            e2 = replace_call(e2, p.x, p.t0, x0)
            e2 = subs3(e2, x0, p.x, :i, 0)
            concat(code, :(ExaModels.constraint($p_ocp, $e2 for i ∈ $rg; lcon = $e1, ucon = $e3)))
        end
        (:final, rg) => begin
            if isnothing(rg)
                rg = :(1:$(p.dim_x))
                e2 = subs(e2, p.x, :($(p.x)[$rg]))
            elseif !is_range(rg)
                rg = as_range(rg)
            end
            code = :(length($e1) == length($e3) == length($rg) || throw("wrong bound dimension")) # (vs. __throw) since raised at runtime
            xf = __symgen(:xf)
            e2 = replace_call(e2, p.x, p.tf, xf)
            e2 = subs3(e2, xf, p.x, :i, :grid_size)
            concat(code, :(ExaModels.constraint($p_ocp, $e2 for i ∈ $rg; lcon = $e1, ucon = $e3)))
        end
        (:variable_range, rg) => begin
            if isnothing(rg)
                rg = :(1:$(p.dim_v))
                e2 = subs(e2, p.v, :($(p.v)[$rg]))
            elseif !is_range(rg)
                rg = as_range(rg)
            end
            code_box = :(length($e1) == length($e3) == length($rg) || throw("wrong bound dimension")) # (vs. __throw) since raised at runtime
            code_box = __wrap(concat(code_box, :($(p.l_v)[$rg] .= $e1; $(p.u_v)[$rg] .= $e3)), p.lnum, p.line)
            p.box_v = concat(p.box_v, code_box) # not __wrapped since contains definition of l_v/u_v
            :()
        end
        (:state_range, rg) => begin
            if isnothing(rg)
                rg = :(1:$(p.dim_x))
                e2 = subs(e2, p.x, :($(p.x)[$rg]))
            elseif !is_range(rg)
                rg = as_range(rg)
            end
            code_box = :(length($e1) == length($e3) == length($rg) || throw("wrong bound dimension")) # (vs. __throw) since raised at runtime
            code_box = __wrap(concat(code_box, :($(p.l_x)[$rg] .= $e1; $(p.u_x)[$rg] .= $e3)), p.lnum, p.line)
            p.box_x = concat(p.box_x, code_box) # not __wrapped since contains definition of l_x/u_x
            :()
        end
        (:control_range, rg) => begin
            if isnothing(rg)
                rg = :(1:$(p.dim_u))
                e2 = subs(e2, p.u, :($(p.u)[$rg]))
            elseif !is_range(rg)
                rg = as_range(rg)
            end
            code_box = :(length($e1) == length($e3) == length($rg) || throw("wrong bound dimension")) # (vs. __throw) since raised at runtime
            code_box = __wrap(concat(code_box, :($(p.l_u)[$rg] .= $e1; $(p.u_u)[$rg] .= $e3)), p.lnum, p.line)
            p.box_u = concat(p.box_u, code_box) # not __wrapped since contains definition of l_u/u_u
            :()
        end
        :state_fun || control_fun || :mixed => begin
            code = :(length($e1) == length($e3) == 1 || throw("this constraint must be scalar")) # (vs. __throw) since raised at runtime
            xt = __symgen(:xt)
            ut = __symgen(:ut)
            e2 = replace_call(e2, p.x, p.t, xt)
            e2 = replace_call(e2, p.u, p.t, ut)
            e2 = subs2(e2, xt, p.x, :j)
            e2 = subs2(e2, ut, p.u, :j)
            e2 = subs(e2, p.t, :($(p.t0) + j * $(p.dt)))
            concat(code, :(ExaModels.constraint($p_ocp, $e2 for j ∈ 0:grid_size; lcon = $e1, ucon = $e3)))
        end
        _ => return __throw("bad constraint declaration", p.lnum, p.line)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_dynamics!(p, p_ocp, x, t, e, label=nothing; log=false)
    log && println("dynamics: ∂($x)($t) == $e")
    isnothing(label) || return __throw("dynamics cannot be labelled", p.lnum, p.line)
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    x ≠ p.x && return __throw("wrong state $x for dynamics", p.lnum, p.line)
    t ≠ p.t && return __throw("wrong time $t for dynamics", p.lnum, p.line)
    return parsing(:dynamics)(p, p_ocp, x, t, e)
end

function p_dynamics_fun!(p, p_ocp, x, t, e)
    pref = prefix()
    xt = __symgen(:xt)
    ut = __symgen(:ut)
    e = replace_call(e, [p.x, p.u], p.t, [xt, ut])
    fun = __symgen(:fun)
    r = __symgen(:r)
    args = [r, p.t, xt, ut, p.v]
    code = quote
        function $fun($(args...))
            @views $r[:] .= $e
            return nothing
        end
        $pref.dynamics!($p_ocp, $fun)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_dynamics_exa!(p, p_ocp, x, t, e)
    return __throw("dynamics must be defined coordinatewise", p.lnum, p.line) # note: scalar case is redirected before to coordinatewise case
end

function p_dynamics_coord!(p, p_ocp, x, i, t, e, label=nothing; log=false)
    log && println("dynamics: ∂($x[$i])($t) == $e")
    isnothing(label) || return __throw("dynamics cannot be labelled", p.lnum, p.line)
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    x ≠ p.x && return __throw("wrong state $x for dynamics", p.lnum, p.line)
    t ≠ p.t && return __throw("wrong time $t for dynamics", p.lnum, p.line)
    return parsing(:dynamics_coord)(p, p_ocp, x, i, t, e)
end
    
function p_dynamics_coord_fun!(p, p_ocp, x, i, t, e)
    p.dim_x == 1 || return __throw("dynamics cannot be defined coordinatewise", p.lnum, p.line)
    i == 1 || return __throw("out of range dynamics index", p.lnum, p.line)
    return p_dynamics!(p, p_ocp, x, t, e) # i.e. implemented only for scalar case (future, to be completed)
end

function p_dynamics_coord_exa!(p, p_ocp, x, i, t, e)
    i ∈ p.dyn_coords && return __throw("dynamics coordinate $i already defined", p.lnum, p.line)
    append!(p.dyn_coords, i)
    xt = __symgen(:xt)
    ut = __symgen(:ut)
    e = replace_call(e, p.x, p.t, xt)
    e = replace_call(e, p.u, p.t, ut)
    j1 = :j
    j2 = :(j + 1)
    ej1 = subs2(e, xt, p.x, j1)
    ej1 = subs2(ej1, ut, p.u, j1)
    ej1 = subs(ej1, p.t, :($(p.t0) + $j1 * $(p.dt)))
    ej2 = subs2(e, xt, p.x, j2)
    ej2 = subs2(ej2, ut, p.u, j2)
    ej2 = subs(ej2, p.t, :($(p.t0) + $j2 * $(p.dt)))
    dxij = :($(p.x)[$i, $j2]- $(p.x)[$i, $j1])
    code = quote 
        if scheme == :trapezoidal
            ExaModels.constraint($p_ocp, $dxij - $(p.dt) * ($ej1 + $ej2) / 2 for j ∈ 0:(grid_size - 1))
        elseif scheme == :euler
            ExaModels.constraint($p_ocp, $dxij - $(p.dt) * $ej1 for j ∈ 0:(grid_size - 1))
        elseif scheme == :euler_b
            ExaModels.constraint($p_ocp, $dxij - $(p.dt) * $ej2 for j ∈ 0:(grid_size - 1))
        else
           throw("unknown numerical scheme") # (vs. __throw) since raised at runtime (and __wrap-ped)
        end
    end
    return __wrap(code, p.lnum, p.line)
end

function p_lagrange!(p, p_ocp, e, type; log=false)
    log && println("objective (Lagrange): ∫($e) → $type")
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    return parsing(:lagrange)(p, p_ocp, e, type)
end
    
function p_lagrange_fun!(p, p_ocp, e, type)
    pref = prefix()
    xt = __symgen(:xt)
    ut = __symgen(:ut)
    e = replace_call(e, [p.x, p.u], p.t, [xt, ut])
    ttype = QuoteNode(type)
    fun = __symgen(:fun)
    args = [p.t, xt, ut, p.v]
    code = quote
        function $fun($(args...))
            return @views $e
        end
        $pref.objective!($p_ocp, $ttype; lagrange=$fun)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_lagrange_exa!(p, p_ocp, e, type)
    xt = __symgen(:xt)
    ut = __symgen(:ut)
    e = replace_call(e, p.x, p.t, xt)
    e = replace_call(e, p.u, p.t, ut)
    j = __symgen(:j)
    ej = subs2(e, xt, p.x, j)
    ej = subs2(ej, ut, p.u, j)
    ej = subs(ej, p.t, :($(p.t0) + $j * $(p.dt)))
    sg = (type == :min) ? 1 : (-1)
    code = quote 
        if scheme == :trapezoidal
            ExaModels.objective($p_ocp, $sg * $(p.dt) * $ej / 2 for $j ∈ (0, grid_size))
            ExaModels.objective($p_ocp, $sg * $(p.dt) * $ej for $j ∈ 1:(grid_size - 1))
        elseif scheme == :euler
            ExaModels.objective($p_ocp, $sg * $(p.dt) * $ej for $j ∈ 0:(grid_size - 1))
        elseif scheme == :euler_b
            ExaModels.objective($p_ocp, $sg * $(p.dt) * $ej for $j ∈ 1:grid_size)
        else
           throw("unknown numerical scheme") # (vs. __throw) since raised at runtime (and __wrap-ped)
        end
    end
    return __wrap(code, p.lnum, p.line)
end

function p_mayer!(p, p_ocp, e, type; log=false)
    log && println("objective (Mayer): $e → $type")
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.t0) && return __throw("time not yet declared", p.lnum, p.line)
    isnothing(p.tf) && return __throw("time not yet declared", p.lnum, p.line)
    has(e, :∫) && return __throw(
        "bad objective declaration resulting in a Mayer term with trailing ∫",
        p.lnum,
        p.line,
    )
    return parsing(:mayer)(p, p_ocp, e, type)
end

function p_mayer_fun!(p, p_ocp, e, type)
    pref = prefix()
    fun = __symgen(:fun)
    x0 = __symgen(:x0)
    xf = __symgen(:xf)
    e = replace_call(e, p.x, p.t0, x0)
    e = replace_call(e, p.x, p.tf, xf)
    ttype = QuoteNode(type)
    args = [x0, xf, p.v]
    code = quote
        function $fun($(args...))
            return @views $e
        end
        $pref.objective!($p_ocp, $ttype; mayer=$fun)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_mayer_exa!(p, p_ocp, e, type)
    x0 = __symgen(:x0) 
    xf = __symgen(:xf) 
    e = replace_call(e, p.x, p.t0, x0)
    e = replace_call(e, p.x, p.tf, xf)
    e = subs2(e, x0, p.x, 0)
    e = subs2(e, xf, p.x, :grid_size)
    # now, x[i](t0) has been replaced by x[i, 0] and x[i](tf) by x[i, grid_size]
    sg = (type == :min) ? 1 : (-1)
    code = :(ExaModels.objective($p_ocp, $sg * $e))
    return __wrap(code, p.lnum, p.line)
end

function p_bolza!(p, p_ocp, e1, e2, type; log=false)
    log && println("objective (Bolza): $e1 + ∫($e2) → $type")
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.t0) && return __throw("time not yet declared", p.lnum, p.line)
    isnothing(p.tf) && return __throw("time not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    return parsing(:bolza)(p, p_ocp, e1, e2, type)
end 

function p_bolza_fun!(p, p_ocp, e1, e2, type)
    pref = prefix()
    fun1 = __symgen(:fun1)
    x0 = __symgen(:x0)
    xf = __symgen(:xf)
    e1 = replace_call(e1, p.x, p.t0, x0)
    e1 = replace_call(e1, p.x, p.tf, xf)
    args1 = [x0, xf, p.v]
    fun2 = __symgen(:fun2)
    xt = __symgen(:xt)
    ut = __symgen(:ut)
    e2 = replace_call(e2, [p.x, p.u], p.t, [xt, ut])
    args2 = [p.t, xt, ut, p.v]
    ttype = QuoteNode(type)
    code = quote
        function $fun1($(args1...))
            return @views $e1
        end
        function $fun2($(args2...))
            return @views $e2
        end
        $pref.objective!($p_ocp, $ttype; mayer=$fun1, lagrange=$fun2)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_bolza_exa!(p, p_ocp, e1, e2, type)
    code1 = p_mayer_exa!(p, p_ocp, e1, type)
    code2 = p_lagrange_exa!(p, p_ocp, e2, type) # in ExaModels, several objectives means sum
    return concat(code1, code2)
end

# Summary of available parsing options

const PARSING_FUN = OrderedDict{Symbol, Function}()
PARSING_FUN[:pragma] = p_pragma_fun!
PARSING_FUN[:alias] = p_alias_fun!
PARSING_FUN[:variable] = p_variable_fun!
PARSING_FUN[:time] = p_time_fun!
PARSING_FUN[:state] = p_state_fun!
PARSING_FUN[:control] = p_control_fun!
PARSING_FUN[:constraint] = p_constraint_fun!
PARSING_FUN[:dynamics] = p_dynamics_fun!
PARSING_FUN[:dynamics_coord] = p_dynamics_coord_fun!
PARSING_FUN[:lagrange] = p_lagrange_fun!
PARSING_FUN[:mayer] = p_mayer_fun!
PARSING_FUN[:bolza] = p_bolza_fun!

const PARSING_EXA = OrderedDict{Symbol, Function}()
PARSING_EXA[:pragma] = p_pragma_exa!
PARSING_EXA[:alias] = p_alias_exa!
PARSING_EXA[:variable] = p_variable_exa!
PARSING_EXA[:time] = p_time_exa!
PARSING_EXA[:state] = p_state_exa!
PARSING_EXA[:control] = p_control_exa!
PARSING_EXA[:constraint] = p_constraint_exa!
PARSING_EXA[:dynamics] = p_dynamics_exa!
PARSING_EXA[:dynamics_coord] = p_dynamics_coord_exa!
PARSING_EXA[:lagrange] = p_lagrange_exa!
PARSING_EXA[:mayer] = p_mayer_exa!
PARSING_EXA[:bolza] = p_bolza_exa!

const PARSING_DIR = OrderedDict{Symbol, OrderedDict{Symbol, Function}}()
PARSING_DIR[:fun] = PARSING_FUN
PARSING_DIR[:exa] = PARSING_EXA

parsing(s) = PARSING_DIR[parsing_backend()][s] # calls the primitive associated with symbol s (:alias, etc.) for the current backend

"""
$(TYPEDSIGNATURES)

Define an optimal control problem. One pass parsing of the definition. Can be used writing either
`ocp = @def begin ... end` or `@def ocp begin ... end`. In the second case, setting `log` to `true`
will display the parsing steps.

# Example
```@example
ocp = @def begin
    tf ∈ R, variable
    t ∈ [ 0, tf ], time
    x ∈ R², state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q = x₁
    v = x₂
    q(0) == 1
    v(0) == 2
    q(tf) == 0
    v(tf) == 0
    0 ≤ q(t) ≤ 5,       (1)
    -2 ≤ v(t) ≤ 3,      (2)
    ẋ(t) == [ v(t), u(t) ]
    tf → min
end

@def ocp begin
    tf ∈ R, variable
    t ∈ [ 0, tf ], time
    x ∈ R², state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q = x₁
    v = x₂
    q(0) == 1
    v(0) == 2
    q(tf) == 0
    v(tf) == 0
    0 ≤ q(t) ≤ 5,       (1)
    -2 ≤ v(t) ≤ 3,      (2)
    ẋ(t) == [ v(t), u(t) ]
    tf → min
end true # final boolean to show parsing log
```
"""
macro def(e)
    try
        e_pref = e_prefix()
        code = @match parsing_backend() begin
            :fun => def_fun(e)
            :exa => def_exa(e)
            _ => throw("unknown parsing backend")
        end
        return esc(code)
    catch ex
        :(throw($ex)) # can be caught by user
    end
end

macro def(ocp, e, log=false)
    try
        e_pref = e_prefix()
        code = @match parsing_backend() begin
            :fun => def_fun(e, log)
            :exa => def_exa(e, log)
            _ => throw("unknown parsing backend") # should be prevented by parsing_backend!
        end
        code = :($ocp = $code)
        return esc(code)
    catch ex
        :(throw($ex)) # can be caught by user
    end
end

function def_fun(e, log=false)
    pref = prefix()
    p_ocp = __symgen(:p_ocp)
    code = :($p_ocp = $pref.PreModel())
    p = ParsingInfo()
    code = concat(code, parse!(p, p_ocp, e; log=log))
    ee = QuoteNode(e)
    code = concat(code, :($pref.definition!($p_ocp, $ee)))
    code = concat(code, :($pref.build_model($p_ocp)))
    return code
end

function def_exa(e, log=false)
    e_pref = e_prefix()
    p_ocp = __symgen(:p_ocp) # ExaModel name (this is the pre OCP, here)
    p = ParsingInfo()
    code = parse!(p, p_ocp, e; log = log)
    dyn_check = quote
        !isempty($(p.dyn_coords)) || throw($e_pref.ParsingError("dynamics not defined"))
        sort($(p.dyn_coords)) == 1:$(p.dim_x) || throw($e_pref.ParsingError("some coordinates of dynamics undefined"))         
    end
    default_scheme = QuoteNode(__default_scheme_exa())
    default_grid_size = __default_grid_size_exa()
    default_backend = __default_backend_exa()
    default_init = __default_init_exa()
    default_base_type = __default_base_type_exa()
    code = quote
        function (; scheme = $default_scheme, grid_size = $default_grid_size, backend = $default_backend, init = $default_init, base_type = $default_base_type)
            $(p.box_v) # lvar and uvar for variable
            $(p.box_x) # lvar and uvar for state
            $(p.box_u) # lvar and uvar for control
            $p_ocp = ExaModels.ExaCore(base_type; backend = backend)
            $code
            $dyn_check
            return ExaModels.ExaModel($p_ocp)
        end
    end
    return code
end
