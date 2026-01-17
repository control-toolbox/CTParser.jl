"""
$(TYPEDSIGNATURES)

Return the default backend prefix used by `@init`.

The returned symbol identifies the module that implements
`build_initial_guess` and `validate_initial_guess`. In the current
implementation this is `:CTModels`.

# Returns

- `Symbol`: name of the default backend module.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.__default_init_prefix()
:CTModels
```
"""
function __default_init_prefix()
    return :CTModels
end

"""
Current backend prefix used by `@init`.

This reference stores the symbol of the backend module that provides
`build_initial_guess` and `validate_initial_guess`. It is initialised
by [`__default_init_prefix`](@ref) and can be updated at runtime via
[`init_prefix!`](@ref).
"""
const INIT_PREFIX = Ref(__default_init_prefix())

"""
$(TYPEDSIGNATURES)

Return the current backend prefix used by `@init`.

This is the symbol of the module that will be used to interpret the
`NamedTuple` produced by the initialisation DSL, typically `:CTModels`.

# Returns

- `Symbol`: name of the backend module currently used by `@init`.

# Example

```julia-repl
julia> using CTParser

julia> CTParser.init_prefix()
:CTModels
```
"""
function init_prefix()
    return INIT_PREFIX[]
end

"""
$(TYPEDSIGNATURES)

Set the backend prefix used by `@init`.

This function updates the global [`INIT_PREFIX`](@ref), thereby changing
which module is used to build and validate initial guesses.

# Arguments

- `p::Symbol`: name of the backend module to use (e.g. `:CTModels`).

# Returns

- `Nothing`.

# Example

```julia-repl
julia> using CTParser

julia> old = CTParser.init_prefix();

julia> CTParser.init_prefix!(:MyBackend)

julia> CTParser.init_prefix()
:MyBackend

julia> CTParser.init_prefix!(old);  # restore
```
"""
function init_prefix!(p)
    INIT_PREFIX[] = p
    return nothing
end

"""
$(TYPEDSIGNATURES)

Internal helper that parses the body of an `@init` block.

The function walks through the expression `ex` and splits it into

- *alias statements*, which are left as ordinary Julia assignments and
  executed verbatim inside the generated block;
- *initialisation specifications* of the form `lhs := rhs` or
  `lhs(t) := rhs` / `lhs(T) := rhs`, which are converted into keys and
  values used to build a `NamedTuple`.

# Arguments

- `ex::Any`: expression or block coming from the body of `@init`.

# Returns

- `alias_stmts::Vector{Expr}`: ordinary statements to execute before
  building the initial guess.
- `keys::Vector{Symbol}`: names of the components being initialised
  (e.g. `:q`, `:v`, `:u`, `:tf`).
- `vals::Vector{Any}`: expressions representing the corresponding
  values, functions or `(T, data)` pairs.
"""
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

"""
$(TYPEDSIGNATURES)

Lowering function used by `@init` to build the expanded code.

Given an optimal control problem `ocp` and the body `e` of an `@init`
block, this function collects the initialisation specifications, builds
an appropriate `NamedTuple` expression and constructs the call to
`build_initial_guess` / `validate_initial_guess` on the current backend
(prefix returned by [`init_prefix`](@ref)).

It also produces a compact string representation of the specification,
used for optional logging when `log = true` is requested at the
macro level.

# Arguments

- `ocp`: symbolic optimal control problem built with `@def`.
- `e`: expression corresponding to the body of the `@init` block.

# Returns

- `log_str::String`: human-readable `NamedTuple`-like description of
  the specification.
- `code_expr::Expr`: block of Julia code that builds and validates the
  initial guess when executed.
"""
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

"""
    @init ocp begin
        ...
    end

Build an initial guess object for an optimal control problem from a small
initialisation DSL.

The block following `@init` is interpreted as a collection of assignment
rules for the state, control and variable components of an optimal control
problem, using a compact syntax of the form

```julia
q(t) := sin(t)     # time-dependent function
x(T) := X          # time grid and associated samples
u := 0.1           # constant value
a = 1.0           # ordinary Julia alias (not part of the initial guess)
v(t) := a         # time-dependent function using the alias above
```

The macro itself only rewrites this DSL into a `NamedTuple`-based
representation. All dimensional checks, interpretation of aliases and
construction of the concrete initial guess object are delegated to the
backend selected by [`init_prefix`](@ref) (by défaut `:CTModels`), via
`build_initial_guess` and `validate_initial_guess`.

An optional keyword-like trailing argument controls logging:

```julia
ig = @init ocp begin
    u(t) := t
end log = true
```

When `log = true`, the macro additionally prints a human-readable
`NamedTuple`-like representation of the specification.

# Arguments

- `ocp`: symbolic optimal control problem built with `@def`.
- `begin ... end`: block containing the initialisation DSL.
- `log`: optional Boolean keyword (default `false`) enabling textual
  logging of the parsed specification.

# Returns

- `AbstractOptimalControlInitialGuess`: backend-specific initial guess
  object produced by the current backend (par défaut `CTModels`).

# Example

```julia-repl
julia> using CTParser

julia> ocp = @def begin
           t ∈ [0, 1], time
           x ∈ R, state
           u ∈ R, control
           ẋ(t) == u(t)
           x(0) == 0
           x(1) == 0
           ∫(0.5u(t)^2) → min
       end

julia> ig = @init ocp begin
           u(t) := t
       end

julia> ig isa CTModels.AbstractOptimalControlInitialGuess
true
```
"""
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
                "Unsupported trailing argument in @init. Use `log = true` or `log = false`."
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
