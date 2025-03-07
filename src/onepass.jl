# onepass
# todo: as_range / as_vector for rg / lb, ub could be done here in p_constraint when calling PREFIX.constraint!
# - cannot call solve if problem not fully defined (dynamics not defined...)
# - doc: explain projections wrt to t0, tf, t; (...x1...x2...)(t) -> ...gensym1...gensym2... (most internal first)
# - robustify repl
# - additional checks: when generating functions (constraints, dynamics, costs), there should not be any x or u left
#   (but the user might indeed do so); meaning that has(ee, x/u/t) must be false (postcondition)
# - tests exceptions (parsing and semantics/runtime)
# - add assert for pre/post conditions and invariants
# - add tests on ParsingError + run time errors (wrapped in try ... catch's - use string to be precise)
# - currently "t ∈ [ 0+0, 1 ], time" is allowed, and compels to declare "x(0+0) == ..."

const PREFIX = :OptimalControl # prefix for generated code, assumed to be evaluated within OptimalControl.jl

"""
$(TYPEDEF)

**Fields**

"""
@with_kw mutable struct ParsingInfo
    v::Symbol = Symbol(:unset_, gensym()) # not nothing as, if unset, this name is still used in function args
    t::Union{Symbol,Nothing} = nothing
    t0::Union{Real,Symbol,Expr,Nothing} = nothing
    tf::Union{Real,Symbol,Expr,Nothing} = nothing
    x::Union{Symbol,Nothing} = nothing
    u::Union{Symbol,Nothing} = nothing
    is_scalar_x::Bool = false # todo: remove when allowing componentwise declaration of dynamics
    aliases::OrderedDict{Symbol,Union{Real,Symbol,Expr}} = __init_aliases()
    lnum::Int = 0
    line::String = ""
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
    al
end

__throw(ex, n, line) = quote
    local info
    info = string("\nLine ", $n, ": ", $line)
    throw($PREFIX.ParsingError(info * "\n" * $ex))
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
        :(∂($x[1])($t) == $e1) => if p.is_scalar_x # todo: remove in future
            p_dynamics!(p, p_ocp, x, t, e1; log)
        else
            (return __throw("Wrong dynamics declaration", p.lnum, p.line))
        end
        :(∂($x[1])($t) == $e1, $label) => if p.is_scalar_x # todo: remove in future
            p_dynamics!(p, p_ocp, x, t, e1, label; log)
        else
            (return __throw("Wrong dynamics declaration", p.lnum, p.line))
        end
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
        :($e1 * ∫($e2) → min) => if has(e1, p.t)
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

function p_alias!(p, p_ocp, a, e; log=false)
    log && println("alias: $a = $e")
    a isa Symbol || return __throw("forbidden alias name: $a", p.lnum, p.line)
    aa = QuoteNode(a)
    ee = QuoteNode(e)
    for i in 1:9
        p.aliases[Symbol(a, CTBase.ctupperscripts(i))] = :($a^$i)
    end
    p.aliases[a] = e
    code = :(LineNumberNode(0, "alias: " * string($aa) * " = " * string($ee)))
    return __wrap(code, p.lnum, p.line)
end

function p_variable!(p, p_ocp, v, q; components_names=nothing, log=false)
    log && println("variable: $v, dim: $q")
    v isa Symbol || return __throw("forbidden variable name: $v", p.lnum, p.line)
    vv = QuoteNode(v)
    if q == 1
        vg = Symbol(v, gensym())
        p.aliases[v] = :($vg[1])
        v = vg 
    end
    p.v = v
    qq = q isa Int ? q : 9
    for i in 1:qq
        p.aliases[Symbol(v, CTBase.ctindices(i))] = :($v[$i])
    end # make: v₁, v₂... if the variable is named v
    for i in 1:qq
        p.aliases[Symbol(v, i)] = :($v[$i])
    end # make: v1, v2... if the variable is named v
    for i in 1:9
        p.aliases[Symbol(v, CTBase.ctupperscripts(i))] = :($v^$i)
    end # make: v¹, v²... if the variable is named v
    if (isnothing(components_names))
        code = :($PREFIX.variable!($p_ocp, $q, $vv))
    else
        qq == length(components_names.args) ||
            return __throw("the number of variable components must be $qq", p.lnum, p.line)
        for i in 1:qq
            p.aliases[components_names.args[i]] = :($v[$i])
        end # aliases from names given by the user
        ss = QuoteNode(string.(components_names.args))
        code = :($PREFIX.variable!($p_ocp, $q, $vv, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_time!(p, p_ocp, t, t0, tf; log=false)
    log && println("time: $t, initial time: $t0, final time: $tf")
    t isa Symbol || return __throw("forbidden time name: $t", p.lnum, p.line)
    p.t = t
    p.t0 = t0
    p.tf = tf
    tt = QuoteNode(t)
    code = @match (has(t0, p.v), has(tf, p.v)) begin
        (false, false) => :($PREFIX.time!($p_ocp; t0=$t0, tf=$tf, time_name=$tt))
        (true, false) => @match t0 begin
            :($v1[$i]) && if (v1 == p.v)
            end => :($PREFIX.time!($p_ocp; ind0=$i, tf=$tf, time_name=$tt))
            :($v1) && if (v1 == p.v)
            end => quote
                ($p_ocp.variable_dimension ≠ 1) && throw( # debug: add info (dim of var) in PreModel
                    $PREFIX.ParsingError("variable must be of dimension one for a time"),
                )
                $PREFIX.time!($p_ocp; ind0=1, tf=$tf, time_name=$tt)
            end
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        (false, true) => @match tf begin
            :($v1[$i]) && if (v1 == p.v)
            end => :($PREFIX.time!($p_ocp; t0=$t0, indf=$i, time_name=$tt))
            :($v1) && if (v1 == p.v)
            end => quote
                ($p_ocp.variable_dimension ≠ 1) && throw( # debug: add info (dim of var) in PreModel
                    $PREFIX.ParsingError("variable must be of dimension one for a time"),
                )
                $PREFIX.time!($p_ocp; t0=$t0, indf=1, time_name=$tt)
            end
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
        _ => @match (t0, tf) begin
            (:($v1[$i]), :($v2[$j])) && if (v1 == v2 == p.v)
            end => :($PREFIX.time!($p_ocp; ind0=$i, indf=$j, time_name=$tt))
            _ => return __throw("bad time declaration", p.lnum, p.line)
        end
    end
    return __wrap(code, p.lnum, p.line)
end

function p_state!(p, p_ocp, x, n; components_names=nothing, log=false)
    log && println("state: $x, dim: $n")
    x isa Symbol || return __throw("forbidden state name: $x", p.lnum, p.line)
    p.aliases[Symbol(Unicode.normalize(string(x, "̇")))] = :(∂($x))
    xx = QuoteNode(x)
    if n == 1
        xg = Symbol(x, gensym())
        p.aliases[x] = :($xg[1])
        x = xg 
        p.is_scalar_x = true # todo: remove in future
    end
    p.x = x
    nn = n isa Int ? n : 9
    for i in 1:nn
        p.aliases[Symbol(x, CTBase.ctindices(i))] = :($x[$i])
    end # Make x₁, x₂... if the state is named x
    for i in 1:nn
        p.aliases[Symbol(x, i)] = :($x[$i])
    end # Make x1, x2... if the state is named x
    for i in 1:9
        p.aliases[Symbol(x, CTBase.ctupperscripts(i))] = :($x^$i)
    end # Make x¹, x²... if the state is named x
    if (isnothing(components_names))
        code = :($PREFIX.state!($p_ocp, $n, $xx))
    else
        nn == length(components_names.args) ||
            return __throw("the number of state components must be $nn", p.lnum, p.line)
        for i in 1:nn
            p.aliases[components_names.args[i]] = :($x[$i])
            # todo: add aliases for state components (scalar) derivatives
        end # Aliases from names given by the user
        ss = QuoteNode(string.(components_names.args))
        code = :($PREFIX.state!($p_ocp, $n, $xx, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_control!(p, p_ocp, u, m; components_names=nothing, log=false)
    log && println("control: $u, dim: $m")
    u isa Symbol || return __throw("forbidden control name: $u", p.lnum, p.line)
    uu = QuoteNode(u)
    if m == 1
        ug = Symbol(u, gensym())
        p.aliases[u] = :($ug[1])
        u = ug 
    end
    p.u = u
    mm = m isa Int ? m : 9
    for i in 1:mm
        p.aliases[Symbol(u, CTBase.ctindices(i))] = :($u[$i])
    end # make: u₁, u₂... if the control is named u
    for i in 1:mm
        p.aliases[Symbol(u, i)] = :($u[$i])
    end # make: u1, u2... if the control is named u
    for i in 1:9
        p.aliases[Symbol(u, CTBase.ctupperscripts(i))] = :($u^$i)
    end # make: u¹, u²... if the control is named u
    if (isnothing(components_names))
        code = :($PREFIX.control!($p_ocp, $m, $uu))
    else
        mm == length(components_names.args) ||
            return __throw("the number of control components must be $mm", p.lnum, p.line)
        for i in 1:mm
            p.aliases[components_names.args[i]] = :($u[$i])
        end # aliases from names given by the user
        ss = QuoteNode(string.(components_names.args))
        code = :($PREFIX.control!($p_ocp, $m, $uu, $ss))
    end
    return __wrap(code, p.lnum, p.line)
end

function p_constraint!(p, p_ocp, e1, e2, e3, label=gensym(); log=false)
    c_type = constraint_type(e2, p.t, p.t0, p.tf, p.x, p.u, p.v)
    log && println("constraint ($c_type): $e1 ≤ $e2 ≤ $e3,    ($label)")
    label isa Int && (label = Symbol(:eq, label))
    label isa Symbol || return __throw("forbidden label: $label", p.lnum, p.line)
    llabel = QuoteNode(label)
    code = @match c_type begin
        :boundary || :variable_fun || (:initial, rg) || (:final, rg) => begin # :initial and :final now treated as boundary
            gs = gensym()
            x0 = gensym()
            xf = gensym()
            r = gensym()
            ee2 = replace_call(e2, p.x, p.t0, x0)
            ee2 = replace_call(ee2, p.x, p.tf, xf)
            args = [r, x0, xf, p.v]
            quote
                function $gs($(args...))
                    @views $r[:] .= $ee2
                    return nothing
                end
                $PREFIX.constraint!($p_ocp, :boundary; f=$gs, lb=$e1, ub=$e3, label=$llabel)
            end
        end
        (:control_range, rg) =>
            :($PREFIX.constraint!($p_ocp, :control; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        (:state_range, rg) =>
            :($PREFIX.constraint!($p_ocp, :state; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        (:variable_range, rg) =>
            :($PREFIX.constraint!($p_ocp, :variable; rg=$rg, lb=$e1, ub=$e3, label=$llabel))
        :state_fun || control_fun || :mixed => begin # now all treated as path
            gs = gensym()
            xt = gensym()
            ut = gensym()
            r = gensym()
            ee2 = replace_call(e2, [p.x, p.u], p.t, [xt, ut])
            args = [r, p.t, xt, ut, p.v]
            quote
                function $gs($(args...))
                    @views $r[:] .= $ee2
                    return nothing
                end
                $PREFIX.constraint!($p_ocp, :path; f=$gs, lb=$e1, ub=$e3, label=$llabel)
            end
        end
        _ => return __throw("bad constraint declaration", p.lnum, p.line)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_dynamics!(p, p_ocp, x, t, e, label=nothing; log=false)
    ∂x = Symbol(:∂, x)
    log && println("dynamics: $∂x($t) == $e")
    isnothing(label) || return __throw("dynamics cannot be labelled", p.lnum, p.line)
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    x ≠ p.x && return __throw("wrong state for dynamics", p.lnum, p.line)
    t ≠ p.t && return __throw("wrong time for dynamics", p.lnum, p.line)
    xt = gensym()
    ut = gensym()
    e = replace_call(e, [p.x, p.u], p.t, [xt, ut])
    gs = gensym()
    r = gensym()
    args = [r, p.t, xt, ut, p.v]
    code = quote
        function $gs($(args...))
            @views $r[:] .= $e
            return nothing
        end
        $PREFIX.dynamics!($p_ocp, $gs)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_lagrange!(p, p_ocp, e, type; log=false)
    log && println("objective (Lagrange): ∫($e) → $type")
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    xt = gensym()
    ut = gensym()
    e = replace_call(e, [p.x, p.u], p.t, [xt, ut])
    ttype = QuoteNode(type)
    gs = gensym()
    r = gensym()
    args = [p.t, xt, ut, p.v]
    code = quote
        function $gs($(args...))
            return @views $e
        end
        $PREFIX.objective!($p_ocp, $ttype; lagrange=$gs)
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
    gs = gensym()
    x0 = gensym()
    xf = gensym()
    r = gensym()
    e = replace_call(e, p.x, p.t0, x0)
    e = replace_call(e, p.x, p.tf, xf)
    ttype = QuoteNode(type)
    args = [x0, xf, p.v]
    code = quote
        function $gs($(args...))
            return @views $e
        end
        $PREFIX.objective!($p_ocp, $ttype; mayer=$gs)
    end
    return __wrap(code, p.lnum, p.line)
end

function p_bolza!(p, p_ocp, e1, e2, type; log=false)
    log && println("objective (Bolza): $e1 + ∫($e2) → $type")
    isnothing(p.x) && return __throw("state not yet declared", p.lnum, p.line)
    isnothing(p.t0) && return __throw("time not yet declared", p.lnum, p.line)
    isnothing(p.tf) && return __throw("time not yet declared", p.lnum, p.line)
    isnothing(p.u) && return __throw("control not yet declared", p.lnum, p.line)
    isnothing(p.t) && return __throw("time not yet declared", p.lnum, p.line)
    gs1 = gensym()
    x0 = gensym()
    xf = gensym()
    r1 = gensym()
    e1 = replace_call(e1, p.x, p.t0, x0)
    e1 = replace_call(e1, p.x, p.tf, xf)
    args1 = [x0, xf, p.v]
    gs2 = gensym()
    xt = gensym()
    ut = gensym()
    r2 = gensym()
    e2 = replace_call(e2, [p.x, p.u], p.t, [xt, ut])
    args2 = [p.t, xt, ut, p.v]
    ttype = QuoteNode(type)
    code = quote
        function $gs1($(args1...))
            return @views $e1
        end
        function $gs2($(args2...))
            return @views $e2
        end
        $PREFIX.objective!($p_ocp, $ttype; mayer=$gs1, lagrange=$gs2)
    end
    return __wrap(code, p.lnum, p.line)
end

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
    ocp = gensym()
    code = quote
        @def $ocp $e
        $ocp
    end
    return esc(code)
end

macro def(ocp, e, log=false)
    try
        p_ocp = gensym()
        code = :($p_ocp = $PREFIX.PreModel())
        p = ParsingInfo()
        code = Expr(:block, code, parse!(p, p_ocp, e; log=log))
        ee = QuoteNode(e)
        code = Expr(:block, code, :($PREFIX.definition!($p_ocp, $ee)))
        code = Expr(:block, code, :($ocp = $PREFIX.build_model($p_ocp)))
        return esc(code)
    catch ex
        :(throw($ex)) # can be caught by user
    end
end
