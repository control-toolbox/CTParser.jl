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
by `__default_init_prefix` and can be updated at runtime via
`init_prefix!`.
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

This function updates the global `INIT_PREFIX`, thereby changing
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

Generate runtime code for a temporal specification `lhs(arg) := rhs`.

This function produces the Julia expression that will be evaluated at runtime
to determine whether the specification represents a time-dependent function
or a time grid, based on whether `arg` matches `time_name(ocp)`.

# Arguments

- `pref::Symbol`: backend module prefix (e.g. `:CTModels`).
- `ocp`: symbolic OCP variable passed from the macro.
- `arg`: argument used in the specification (e.g. `:t`, `:s`, or a literal array after alias expansion).
- `rhs`: right-hand side expression.
- `arg_in_rhs::Bool`: whether `arg` appears in `rhs` (computed at parse-time via `has`).

# Returns

- `val_sym::Symbol`: generated symbol to store the computed value.
- `code::Expr`: expression block to insert in the generated code.

# Notes

When `arg` is not a Symbol (e.g., a literal array after alias expansion like
`[0.0, 0.5, 1.0]`), it is always treated as a time grid specification.

When `arg` is a Symbol and `arg_in_rhs` is `true`, the specification is definitely
a time-dependent function, so we validate that `arg == Symbol(time_name(ocp))` and
throw an error if not. When `arg_in_rhs` is `false`, we generate a runtime conditional
that checks whether `arg` matches the time name to decide between a constant
function or a time grid.
"""
function __gen_temporal_value(pref, ocp, arg, rhs, arg_in_rhs)
    val_sym = __symgen(:init_val)

    # Early return: if arg is not a Symbol (e.g., literal array after alias expansion),
    # it's always a grid specification
    if !(arg isa Symbol)
        code = :($val_sym = ($arg, $rhs))
        return val_sym, code
    end

    arg_quoted = QuoteNode(arg)

    if arg_in_rhs
        # arg appears in rhs → must be a time-dependent function
        # Validate at runtime that arg matches time_name(ocp)
        code = quote
            let _expected = Symbol($pref.time_name($ocp))
                if $arg_quoted != _expected
                    error(
                        "Incorrect time variable in @init: " *
                        "used :" *
                        string($arg_quoted) *
                        " but time_name(ocp) is " *
                        "\"" *
                        $pref.time_name($ocp) *
                        "\" " *
                        "(expected :" *
                        string(_expected) *
                        "). " *
                        "Please use :" *
                        string(_expected) *
                        " instead of :" *
                        string($arg_quoted) *
                        " " *
                        "in your @init block.",
                    )
                end
            end
            $val_sym = $arg -> $rhs
        end
    else
        # arg does NOT appear in rhs → ambiguous
        # Runtime check: if arg matches time_name → constant function, else → grid
        code = quote
            $val_sym = if Symbol($pref.time_name($ocp)) == $arg_quoted
                $arg -> $rhs          # constant time function
            else
                ($arg, $rhs)          # time grid
            end
        end
    end

    return val_sym, code
end

"""
$(TYPEDSIGNATURES)

Generate runtime code for a single initialisation specification.

This function dispatches based on the specification kind (`:constant` or
`:temporal`) and delegates to the appropriate code generator.

# Arguments

- `pref::Symbol`: backend module prefix.
- `ocp`: symbolic OCP variable.
- `spec::Tuple`: specification tuple, either `(:constant, rhs)` or
  `(:temporal, arg, rhs, arg_in_rhs)`.

# Returns

- `val_sym::Symbol`: generated symbol to store the value.
- `code::Expr`: expression to insert in the generated code.
"""
function __gen_spec_value(pref, ocp, spec)
    kind = spec[1]
    if kind == :constant
        rhs = spec[2]
        val_sym = __symgen(:init_val)
        code = :($val_sym = $rhs)
        return val_sym, code
    elseif kind == :temporal
        arg, rhs, arg_in_rhs = spec[2], spec[3], spec[4]
        return __gen_temporal_value(pref, ocp, arg, rhs, arg_in_rhs)
    else
        error("Unknown spec kind: $kind")
    end
end

"""
$(TYPEDSIGNATURES)

Format a single initialisation specification for logging.

This function produces a human-readable string representation of a
specification, used when `log = true` is passed to `@init`.

# Arguments

- `key::Symbol`: component name (e.g. `:u`, `:x`).
- `spec::Tuple`: specification tuple.

# Returns

- `String`: formatted string like `"u = t -> sin(t)"` or `"x = 1.0"`.
"""
function __log_spec(key, spec)
    kind = spec[1]
    if kind == :constant
        rhs = spec[2]
        rhs_str = if rhs isa Expr
            sprint(Base.show_unquoted, Base.remove_linenums!(deepcopy(rhs)))
        else
            sprint(show, rhs)
        end
        return string(key, " = ", rhs_str)
    elseif kind == :temporal
        arg, rhs = spec[2], spec[3]
        rhs_clean = if rhs isa Expr
            Base.remove_linenums!(deepcopy(rhs))
        else
            rhs
        end
        rhs_str = sprint(Base.show_unquoted, rhs_clean)

        # If arg is not a Symbol (e.g., literal array after alias expansion),
        # format as grid specification
        if arg isa Symbol
            return string(key, " = ", arg, " -> ", rhs_str)
        else
            arg_str = sprint(Base.show_unquoted, arg)
            return string(key, " = (", arg_str, ", ", rhs_str, ")")
        end
    else
        return string(key, " = ???")
    end
end

"""
$(TYPEDSIGNATURES)

Internal helper that parses the body of an `@init` block.

The function walks through the expression `ex` and splits it into

- *alias statements* of the form `lhs = rhs`, which are stored in a dictionary
  and substituted into subsequent statements at parse-time using `subs`;
- *initialisation specifications* of the form `lhs := rhs` or
  `lhs(arg) := rhs`, which are converted into structured specification
  tuples after alias expansion.

For expressions of the form `lhs(arg) := rhs`, this function uses `has(rhs, arg)`
to determine whether `arg` appears in the right-hand side. This information
is stored in the specification tuple and used later to generate appropriate
runtime code that distinguishes time-dependent functions from time grids.

Alias substitution happens before each statement is matched, enabling
time-dependent aliases like `phi = 2π * t` and accumulated aliases like
`a = t; s = a`.

# Arguments

- `ex::Any`: expression or block coming from the body of `@init`.
- `lnum::Int`: line number for error reporting.
- `line_str::String`: line string for error reporting.

# Returns

- `keys::Vector{Symbol}`: names of the components being initialised
  (e.g. `:q`, `:v`, `:u`, `:tf`).
- `specs::Vector{Tuple}`: specification tuples, either `(:constant, rhs)`
  for constant values or `(:temporal, arg, rhs, arg_in_rhs)` for temporal
  specifications where `arg_in_rhs` indicates whether `arg` appears in `rhs`.
"""
function _collect_init_specs(ex, lnum::Int, line_str::String)
    aliases = OrderedCollections.OrderedDict{Union{Symbol,Expr},Any}()
    keys = Symbol[]                # keys of the NamedTuple (q, v, x, u, tf, ...)
    specs = Tuple[]                # specification tuples

    stmts = if ex isa Expr && ex.head == :block
        ex.args
    else
        Any[ex]
    end

    for st in stmts
        st isa LineNumberNode && continue

        # Substitute all known aliases before matching
        for a in Base.keys(aliases)
            st = subs(st, a, aliases[a])
        end

        @match st begin
            # Alias: store for future substitution
            :($lhs = $rhs) => begin
                lhs isa Symbol || error(
                    "Unsupported alias left-hand side in @init: $lhs (only Symbol allowed)",
                )
                aliases[lhs] = rhs
            end

            # Forms q(arg) := rhs
            # After alias expansion, arg may be a Symbol or an Expr (literal grid)
            :($lhs($arg) := $rhs) => begin
                lhs isa Symbol || error("Unsupported left-hand side in @init: $lhs")

                # Check if arg appears in rhs using has() from utils.jl
                # Note: if arg is not a Symbol (e.g., after alias expansion to a literal array),
                # has() will return false, which is correct for grid specifications
                arg_in_rhs = (arg isa Symbol) && has(rhs, arg)

                push!(keys, lhs)
                push!(specs, (:temporal, arg, rhs, arg_in_rhs))
            end

            # Constant / variable form: lhs := rhs
            :($lhs := $rhs) => begin
                lhs isa Symbol || error("Unsupported left-hand side in @init: $lhs")
                push!(keys, lhs)
                push!(specs, (:constant, rhs))
            end

            # Fallback: strict mode - reject unrecognized statements
            _ => begin
                error(
                    "Unrecognized statement in @init block: $st. Only alias assignments (a = expr) and specifications (lhs := rhs or lhs(arg) := rhs) are allowed.",
                )
            end
        end
    end

    return keys, specs
end

"""
$(TYPEDSIGNATURES)

Lowering function used by `@init` to build the expanded code.

Given an optimal control problem `ocp` and the body `e` of an `@init`
block, this function collects the initialisation specifications, builds
an appropriate `NamedTuple` expression and constructs the call to
`build_initial_guess` / `validate_initial_guess` on the current backend
(prefix returned by `init_prefix`).

It also produces a compact string representation of the specification,
used for optional logging when `log = true` is requested at the
macro level.

# Arguments

- `ocp`: symbolic optimal control problem built with `@def`.
- `e`: expression corresponding to the body of the `@init` block.
- `lnum::Int`: line number for error reporting.
- `line_str::String`: line string for error reporting.

# Returns

- `log_str::String`: human-readable `NamedTuple`-like description of
  the specification.
- `code_expr::Expr`: block of Julia code that builds and validates the
  initial guess when executed.
"""
function init_fun(ocp, e, lnum::Int, line_str::String)
    keys, specs = _collect_init_specs(e, lnum, line_str)
    pref = init_prefix()

    # If there is no init specification, delegate to build_initial_guess/validate_initial_guess
    if isempty(keys)
        build_call = :($pref.build_initial_guess($ocp, ()))
        validate_call = :($pref.validate_initial_guess($ocp, $build_call))
        code_expr = validate_call
        log_str = "()"
        return log_str, code_expr
    end

    # Generate runtime code for each specification
    body_stmts = Any[]

    val_syms = Symbol[]
    for spec in specs
        val_sym, code = __gen_spec_value(pref, ocp, spec)
        push!(val_syms, val_sym)
        push!(body_stmts, code)
    end

    # Build the NamedTuple with the generated value symbols
    key_nodes = [QuoteNode(k) for k in keys]
    keys_tuple = Expr(:tuple, key_nodes...)
    vals_tuple = Expr(:tuple, val_syms...)
    nt_expr = :(NamedTuple{$keys_tuple}($vals_tuple))

    build_call = :($pref.build_initial_guess($ocp, $nt_expr))
    validate_call = :($pref.validate_initial_guess($ocp, $build_call))
    push!(body_stmts, validate_call)
    code_expr = Expr(:block, body_stmts...)

    # Build log string using __log_spec helper
    pairs_str = [__log_spec(k, s) for (k, s) in zip(keys, specs)]
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
backend selected by `init_prefix` (by défaut `:CTModels`), via
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

- `AbstractInitialGuess`: backend-specific initial guess
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

julia> ig isa CTModels.AbstractInitialGuess
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
            throw_expr = CTParser.__throw(
                "Unsupported trailing argument in @init. Use `log = true` or `log = false`.",
                lnum,
                line_str,
            )
            return esc(throw_expr)
        end
    elseif length(rest) > 1
        throw_expr = CTParser.__throw(
            "Too many trailing arguments in @init. Only a single `log = ...` keyword is supported.",
            lnum,
            line_str,
        )
        return esc(throw_expr)
    end

    log_str, code = try
        init_fun(ocp, e, lnum, line_str)
    catch err
        # Catch any ErrorException from parsing and convert to __throw
        if err isa ErrorException
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
