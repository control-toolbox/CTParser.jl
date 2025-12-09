"""
    @init ocp begin
        ...
    end

Macro to build initialization data (NamedTuple) from a small DSL of the form

    q(t) := sin(t)
    x(T) := X
    u := 0.1
    a = 1.0
    v(t) := a

The macro only transforms this syntax into a `NamedTuple`; all dimensional
validation and detailed handling of OCP aliases is performed by
`build_initial_guess` / `_initial_guess_from_namedtuple` on the backend
module selected by `init_prefix()` (by default `CTModels`).
"""
function __default_init_prefix()
    return :CTModels
end

const INIT_PREFIX = Ref(__default_init_prefix())

function init_prefix()
    return INIT_PREFIX[]
end

function init_prefix!(p)
    INIT_PREFIX[] = p
    return nothing
end

function _collect_init_specs(ex)
    alias_stmts = Expr[]           # statements of the form a = ... or other Julia statements
    keys = Symbol[]                # keys of the NamedTuple (q, v, x, u, tf, ...)
    vals = Any[]                   # expressions for the associated values

    stmts = if ex isa Expr && ex.head == :block
        ex.args
    else
        Any[ex]
    end

    for st in stmts
        st isa LineNumberNode && continue

        @match st begin
            # Alias / ordinary Julia assignments left as-is
            :($lhs = $rhs) => begin
                push!(alias_stmts, st)
            end

            # Forms q(t) := rhs (time-dependent function) or q(T) := rhs (time grid)
            :($lhs($arg) := $rhs) => begin
                lhs isa Symbol || error("Unsupported left-hand side in @init: $lhs")
                if arg == :t
                    # q(t) := rhs → time-dependent function
                    push!(keys, lhs)
                    push!(vals, :($arg -> $rhs))
                else
                    # q(T) := rhs → (T, rhs) for build_initial_guess
                    push!(keys, lhs)
                    push!(vals, :(($arg, $rhs)))
                end
            end

            # Constant / variable form: lhs := rhs
            :($lhs := $rhs) => begin
                lhs isa Symbol || error("Unsupported left-hand side in @init: $lhs")
                push!(keys, lhs)
                push!(vals, rhs)
            end

            # Fallback: any other line is treated as an ordinary Julia statement
            _ => begin
                push!(alias_stmts, st)
            end
        end
    end

    return alias_stmts, keys, vals
end

function init_fun(ocp, e)
    alias_stmts, keys, vals = _collect_init_specs(e)
    pref = init_prefix()

    # If there is no init specification, delegate to build_initial_guess/validate_initial_guess
    if isempty(keys)
        body_stmts = Any[]
        append!(body_stmts, alias_stmts)
        build_call = :($pref.build_initial_guess($ocp, ()))
        validate_call = :($pref.validate_initial_guess($ocp, $build_call))
        push!(body_stmts, validate_call)
        code_expr = Expr(:block, body_stmts...)
        log_str = "()"
        return log_str, code_expr
    end

    # Build the NamedTuple type and its values for execution
    key_nodes = [QuoteNode(k) for k in keys]
    keys_tuple = Expr(:tuple, key_nodes...)
    vals_tuple = Expr(:tuple, vals...)
    nt_expr = :(NamedTuple{$keys_tuple}($vals_tuple))

    body_stmts = Any[]
    append!(body_stmts, alias_stmts)
    build_call = :($pref.build_initial_guess($ocp, $nt_expr))
    validate_call = :($pref.validate_initial_guess($ocp, $build_call))
    push!(body_stmts, validate_call)
    code_expr = Expr(:block, body_stmts...)

    # Build a pretty NamedTuple-like string for logging, of the form (q = ..., v = ..., ...)
    pairs_str = String[]
    for (k, v) in zip(keys, vals)
        vc = v
        if vc isa Expr
            # Remove LineNumberNode noise and print without leading :( ... ) wrapper
            vc_clean = Base.remove_linenums!(deepcopy(vc))
            if vc_clean.head == :-> && length(vc_clean.args) == 2
                arg_expr, body_expr = vc_clean.args
                # Simplify body: strip trivial `begin ... end` with a single non-LineNumberNode expression
                body_clean = body_expr
                if body_clean isa Expr && body_clean.head == :block
                    filtered = [x for x in body_clean.args if !(x isa LineNumberNode)]
                    if length(filtered) == 1
                        body_clean = filtered[1]
                    end
                end
                lhs_str = sprint(Base.show_unquoted, arg_expr)
                rhs_body_str = sprint(Base.show_unquoted, body_clean)
                rhs_str = string(lhs_str, " -> ", rhs_body_str)
            else
                rhs_str = sprint(Base.show_unquoted, vc_clean)
            end
        else
            rhs_str = sprint(show, vc)
        end
        push!(pairs_str, string(k, " = ", rhs_str))
    end
    log_str = if length(pairs_str) == 1
        string("(", pairs_str[1], ",)")
    else
        string("(", join(pairs_str, ", "), ")")
    end

    return log_str, code_expr
end

macro init(ocp, e, rest...)
    src = __source__
    lnum = src.line
    line_str = sprint(show, e)

    # Optional trailing keyword-like argument: @init ocp begin ... end log = true
    log_expr = :(false)
    if length(rest) == 1
        opt = rest[1]
        if opt isa Expr && opt.head == :(=) && opt.args[1] == :log
            log_expr = opt.args[2]
        else
            error(
                "Unsupported trailing argument in @init. Use `log = true` or `log = false`.",
            )
        end
    elseif length(rest) > 1
        error(
            "Too many trailing arguments in @init. Only a single `log = ...` keyword is supported.",
        )
    end

    log_str, code = try
        init_fun(ocp, e)
    catch err
        # Treat unsupported DSL syntax as a static parsing error with proper line info.
        if err isa ErrorException &&
           occursin("Unsupported left-hand side in @init", err.msg)
            throw_expr = CTParser.__throw(err.msg, lnum, line_str)
            return esc(throw_expr)
        else
            rethrow()
        end
    end

    # When log is true, print the NamedTuple-like string corresponding to the DSL
    logged_code = :(
        begin
            if $log_expr
                println($log_str)
            end
            $code
        end
    )

    wrapped = CTParser.__wrap(logged_code, lnum, line_str)
    return esc(wrapped)
end
