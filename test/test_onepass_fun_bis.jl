# test_onepass_fun_bis

function test_onepass_fun_bis()
    @testset "error helpers (__throw and __wrap)" begin
        println("error helpers (bis)")

        # __throw should build an expression that raises a ParsingError when evaluated
        ex = CTParser.__throw("test message", 42, "dummy line")
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # __throw carries the optional suggestion in the ParsingError
        ex = CTParser.__throw("test message", 42, "dummy line"; suggestion="try this")
        err = try
            eval(ex)
        catch e
            e
        end
        @test err isa ParsingError
        @test err.suggestion == "try this"
        @test occursin("Line 42: dummy line", err.msg)
        @test occursin("test message", err.msg)
        err = try
            eval(CTParser.__throw("test message", 42, "dummy line"))
        catch e
            e
        end
        @test isnothing(err.suggestion)

        # __wrap should leave non-throwing code unchanged and propagate its value
        wrapped = CTParser.__wrap(:(1 + 1), 1, "line")
        @test wrapped isa Expr
        @test eval(wrapped) == 2

        # __wrap should catch and rethrow the original exception
        wrapped_err = CTParser.__wrap(:(error("boom")), 1, "line")
        @test wrapped_err isa Expr
        err, captured = capture_stdout(() -> eval(wrapped_err))
        @test err isa ErrorException
        @test occursin("Line 1: line", captured)

        # Structured CTBase exceptions already carry their own diagnostic context
        structured = CTBase.IncorrectArgument(
            "invalid scheme";
            got="gauss_legendre_2",
            expected=":euler or :midpoint",
        )
        wrapped_structured = CTParser.__wrap(:(throw($structured)), 6, "dynamics")
        err, captured = capture_stdout(() -> eval(wrapped_structured))
        @test err === structured
        @test isempty(captured)
        @test err.got == "gauss_legendre_2"
        @test err.expected == ":euler or :midpoint"
    end

    @testset "p_dynamics! precondition errors" begin
        println("p_dynamics! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics test"
        p_ocp = :p_ocp
        x = :x
        t = :t
        e = :(x(t) + u(t))

        # No state declared → "state not yet declared"
        ex = CTParser.p_dynamics!(p, p_ocp, x, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State declared but no control
        p.x = x
        ex = CTParser.p_dynamics!(p, p_ocp, x, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + control declared but no time
        p.u = :u
        ex = CTParser.p_dynamics!(p, p_ocp, x, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Time declared, but wrong state symbol
        p.t = t
        ex = CTParser.p_dynamics!(p, p_ocp, :y, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Correct state but wrong time symbol
        ex = CTParser.p_dynamics!(p, p_ocp, x, :s, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_dynamics! successful setup (no precondition errors)" begin
        println("p_dynamics! success path (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics success test"
        p.x = :x
        p.u = :u
        p.t = :t
        p_ocp = :p_ocp
        e = :(x(t) + u(t))

        # With consistent state/time/control, p_dynamics! should return an Expr
        # that represents wrapped code. We only check that the expression builds
        # successfully (we do not execute it here, as this would require a
        # concrete OCP backend).
        ex = CTParser.p_dynamics!(p, p_ocp, p.x, p.t, e)
        @test ex isa Expr
    end

    @testset "ParsingInfo default values" begin
        println("ParsingInfo defaults (bis)")

        p = CTParser.ParsingInfo()

        # Check that unset fields have expected defaults
        @test p.t === nothing
        @test p.t0 === nothing
        @test p.tf === nothing
        @test p.x === nothing
        @test p.u === nothing
        @test p.dim_v === nothing
        @test p.dim_x === nothing
        @test p.dim_u === nothing
        @test p.is_autonomous == true
        @test p.criterion === nothing
        @test p.lnum == 0
        @test p.line == ""
        @test p.dyn_coords == Int64[]

        # v should be a gensym (Symbol starting with :unset)
        @test p.v isa Symbol
        @test startswith(string(p.v), "unset")

        # aliases should be an OrderedDict with default entries
        @test p.aliases isa OrderedDict
        @test haskey(p.aliases, :R¹) ||
            haskey(p.aliases, Symbol("R", CTBase.ctupperscripts(1)))
        @test haskey(p.aliases, :<=)
        @test haskey(p.aliases, :>=)
        @test haskey(p.aliases, :derivative)
        @test haskey(p.aliases, :integral)
        @test haskey(p.aliases, :(=>))
        @test haskey(p.aliases, :in)
    end

    @testset "p_dynamics_coord! precondition errors" begin
        println("p_dynamics_coord! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics coord test"
        p_ocp = :p_ocp
        x = :x
        t = :t
        i = 1
        e = :(x[1](t) + u(t))

        # No state declared
        ex = CTParser.p_dynamics_coord!(p, p_ocp, x, i, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State declared but no control
        p.x = x
        ex = CTParser.p_dynamics_coord!(p, p_ocp, x, i, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + control but no time
        p.u = :u
        ex = CTParser.p_dynamics_coord!(p, p_ocp, x, i, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Time declared, wrong state
        p.t = t
        ex = CTParser.p_dynamics_coord!(p, p_ocp, :y, i, t, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Correct state, wrong time
        ex = CTParser.p_dynamics_coord!(p, p_ocp, x, i, :s, e)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Labeled dynamics should fail
        ex = CTParser.p_dynamics_coord!(p, p_ocp, x, i, t, e, :my_label)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_time! precondition errors" begin
        println("p_time! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "time test"
        p_ocp = :p_ocp

        # Invalid time name (not a symbol)
        ex = CTParser.p_time!(p, p_ocp, :(t[1]), 0, 1)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_state! precondition errors" begin
        println("p_state! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "state test"
        p_ocp = :p_ocp

        # Invalid state name (not a symbol)
        ex = CTParser.p_state!(p, p_ocp, :(x[1]), 2)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_control! precondition errors" begin
        println("p_control! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "control test"
        p_ocp = :p_ocp

        # Invalid control name (not a symbol)
        ex = CTParser.p_control!(p, p_ocp, :(u[1]), 1)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_variable! precondition errors" begin
        println("p_variable! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "variable test"
        p_ocp = :p_ocp

        # Invalid variable name (not a symbol)
        ex = CTParser.p_variable!(p, p_ocp, :(v[1]), 3)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_alias! precondition errors" begin
        println("p_alias! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "alias test"
        p_ocp = :p_ocp

        # Invalid alias name (not a symbol, e.g. an expression)
        ex = CTParser.p_alias!(p, p_ocp, :(a + b), :c)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Valid alias should work
        p2 = CTParser.ParsingInfo()
        p2.lnum = 1
        p2.line = "alias test valid"
        ex = CTParser.p_alias!(p2, p_ocp, :myalias, :(x + y))
        @test ex isa Expr
        @test haskey(p2.aliases, :myalias)
        @test p2.aliases[:myalias] == :(x + y)
    end

    @testset "p_constraint! label errors" begin
        println("p_constraint! label (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint test"
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.x = :x
        p.u = :u
        p.v = :v
        p_ocp = :p_ocp

        # Invalid label (not a symbol or int)
        ex = CTParser.p_constraint!(p, p_ocp, 0, :(x(t)), 1, :(a + b))
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "p_dynamics! labeled dynamics error" begin
        println("p_dynamics! labeled (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics labeled test"
        p.x = :x
        p.u = :u
        p.t = :t
        p_ocp = :p_ocp
        e = :(x(t) + u(t))

        # Dynamics with a label should fail
        ex = CTParser.p_dynamics!(p, p_ocp, p.x, p.t, e, :my_label)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "autonomy detection" begin
        println("autonomy detection (bis)")

        # Autonomous case: no explicit t in expression after replacing x(t), u(t)
        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "autonomy test"
        p.x = :x
        p.u = :u
        p.t = :t
        p.is_autonomous = true
        p_ocp = :p_ocp
        e = :(x(t) + u(t))

        ex = CTParser.p_dynamics!(p, p_ocp, p.x, p.t, e)
        @test p.is_autonomous == true  # no explicit t remaining

        # Non-autonomous case: explicit t appears
        p2 = CTParser.ParsingInfo()
        p2.lnum = 1
        p2.line = "non-autonomy test"
        p2.x = :x
        p2.u = :u
        p2.t = :t
        p2.is_autonomous = true
        e2 = :(x(t) + t)  # t appears explicitly

        ex2 = CTParser.p_dynamics!(p2, p_ocp, p2.x, p2.t, e2)
        @test p2.is_autonomous == false
    end

    # ============================================================================
    # COST FUNCTIONS - Precondition Errors
    # ============================================================================

    @testset "trace mode shows the names of the user (#354)" begin
        println("trace mode names (bis)")

        # __as_user: g[1] where g is a generated name is the name before the ##
        g = Symbol("u##123")
        @test CTParser.__as_user(Expr(:ref, g, 1)) === :u
        @test CTParser.__as_user(Expr(:call, Expr(:ref, g, 1), :t)) == :(u(t))
        @test CTParser.__as_user(:($(Expr(:ref, g, 1))^2 + 1)) == :(u^2 + 1)
        # ... and nothing else: a component other than the first, a name without ##
        @test CTParser.__as_user(Expr(:ref, g, 2)) == Expr(:ref, g, 2)
        @test CTParser.__as_user(:(x[1])) == :(x[1])
        @test CTParser.__as_user(:(x[2](t))) == :(x[2](t))
        @test CTParser.__as_user(:a) === :a
        @test CTParser.__as_user(3) === 3

        # the example of the issue: no generated name, no index on a scalar
        c(t) = 1.0
        _, trace = capture_stdout() do
            CTParser.def_fun(
                :(
                    begin
                        tf ∈ R, variable
                        t ∈ [0, tf], time
                        x = (q, v) ∈ R², state
                        u ∈ R, control
                        q̇ = v(t)
                        v̇ = u(t) - c(t) * v(t)
                        ẋ(t) == [q̇, v̇]
                        -1 ≤ u(t) ≤ 1
                        u(t) ≤ 2, (bound)
                        ∫(u(t)^2) → min
                    end
                );
                log=true,
            )
        end
        @test !occursin("##", trace)
        @test occursin("time: t, initial time: 0, final time: tf", trace)
        @test occursin("alias: v̇ = u(t) - c(t) * (x[2])(t)", trace)
        @test occursin("dynamics: ∂(x)(t) == [(x[2])(t), u(t) - c(t) * (x[2])(t)]", trace)
        @test occursin("objective (Lagrange): ∫(u(t) ^ 2) → min", trace)
        @test !occursin("u[1]", trace)
        # an unlabelled constraint shows no generated label, a labelled one shows it
        @test occursin("-1 ≤ u(t) ≤ 1\n", trace)
        @test occursin("≤ u(t) ≤ 2,    (bound)", trace)

        # a one-dimensional state, written ∂(y)(t) == …
        _, trace = capture_stdout() do
            CTParser.def_fun(
                :(
                    begin
                        t ∈ [0, 1], time
                        y ∈ R, state
                        w ∈ R, control
                        ∂(y)(t) == w(t)
                        y(0) == 0
                        y(1) → min
                    end
                );
                log=true,
            )
        end
        @test !occursin("##", trace)
        @test occursin("dynamics: ∂(y)(t) == w(t)", trace)
        @test occursin("objective (Mayer): y(1) → min", trace)
    end

    @testset "p_lagrange! precondition errors" begin
        println("p_lagrange! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "lagrange test"
        p_ocp = :p_ocp
        e = :(x(t)^2 + u(t)^2)

        # No state declared
        ex = CTParser.p_lagrange!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State declared but no control
        p.x = :x
        ex = CTParser.p_lagrange!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + control but no time
        p.u = :u
        ex = CTParser.p_lagrange!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # All declared → should build successfully
        p.t = :t
        ex = CTParser.p_lagrange!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test p.criterion == :min

        # Test :max criterion
        p2 = CTParser.ParsingInfo()
        p2.lnum = 1
        p2.line = "lagrange max test"
        p2.x = :x
        p2.u = :u
        p2.t = :t
        ex2 = CTParser.p_lagrange!(p2, :p_ocp, e, :max)
        @test p2.criterion == :max
    end

    @testset "state and control must be evaluated at a time (#193, #139)" begin
        println("stray state and control (bis)")

        # the error raised by the statement `line` in an otherwise valid problem
        function stray_error(line; dynamics=:(ẋ(t) == [x₂(t), u(t)]), cost=:(x₁(1) → min))
            body = quote
                t ∈ [0, 1], time
                x ∈ R², state
                u ∈ R, control
            end
            push!(body.args, dynamics)
            push!(body.args, :(x(0) == [0, 0]))
            isnothing(line) || push!(body.args, line)
            push!(body.args, cost)
            return try
                Core.eval(Main, :(CTParser.@def $body))
                nothing
            catch e
                e
            end
        end

        # a state or a control used without a time (#193)
        for (err, kind, name) in (
            (stray_error(nothing; cost=:(x₁(1) + u → min)), "control", "u"),
            (stray_error(nothing; cost=:(x₁ → min)), "state", "x"),
            (stray_error(nothing; dynamics=:(ẋ(t) == [x₂(t) + x₁, u(t)])), "state", "x"),
            (stray_error(nothing; dynamics=:(ẋ(t) == [x₂(t), u])), "control", "u"),
            (stray_error(nothing; cost=:(∫(x₁(t)^2 + u) → min)), "control", "u"),
            (stray_error(:(x₁(t) + u ≤ 1)), "control", "u"),
            (stray_error(:(x₁(0) + u == 0)), "control", "u"),
            (stray_error(:(x₂(t)^2 + x ≤ 1)), "state", "x"),
        )
            @test err isa ParsingError
            @test occursin("The $kind `$name` is used here without being evaluated at a time", err.msg)
            @test occursin("write `$name(t)`", err.suggestion)
            @test occursin("Line ", err.msg)
        end

        # ... the line shown is the one of the user
        err = stray_error(:(x₁(t) + u ≤ 1))
        @test occursin("x₁(t) + u ≤ 1", err.msg)

        # a state evaluated at a time that is not t, t0 or tf (#139)
        for (err, args) in (
            (stray_error(:(x(0.5) == [0, 0])), "0.5"),
            (stray_error(:(x(s) == [0, 0])), "s"),
            (stray_error(:(x(t0) == [0, 0])), "t0"),
            (stray_error(:(x(0, 1) == [0, 0])), "0, 1"),
            (stray_error(nothing; cost=:(x₁(0.5) → min)), "0.5"),
        )
            @test err isa ParsingError
            @test occursin("The state `x` is evaluated at `$args`", err.msg)
            @test occursin("neither the time `t`, nor the initial time, nor the final time", err.msg)
            @test err.suggestion == "evaluate the state at `t`, `0` or `1`"
        end

        # ... and a control at a time that is not t
        err = stray_error(:(x₁(0) + u(0) ≤ 1))
        @test err isa ParsingError
        @test occursin("The control `u` is evaluated at `0`, which is not the time `t`", err.msg)
        @test err.suggestion == "write `u(t)`"

        # the control is evaluated at the time t only, in every kind of statement: unlike the
        # state, it has no value to give at the initial or the final time (decision of the
        # maintainer, see #139)
        for (err, args) in (
            (stray_error(:(u(1) ≤ 1)), "1"),
            (stray_error(:(u(0) == 0)), "0"),
            (stray_error(:(u₁(0) ≤ 1)), "0"),
            (stray_error(:(x₁(1) + u(1) ≤ 3)), "1"),
            (stray_error(nothing; cost=:(x₁(1) + u(1) → min)), "1"),
            (stray_error(nothing; cost=:(∫(u(0)^2) → min)), "0"),
            (stray_error(nothing; cost=:(u(1)^2 + ∫(u(t)^2) → min)), "1"),
            (stray_error(nothing; dynamics=:(ẋ(t) == [x₂(t), u(0)])), "0"),
        )
            @test err isa ParsingError
            @test occursin("The control `u` is evaluated at `$args`, which is not the time `t`", err.msg)
            @test err.suggestion == "write `u(t)`"
        end

        # a call without argument has no time
        err = stray_error(:(x() == [0, 0]))
        @test err isa ParsingError
        @test occursin("The state `x` is used here without being evaluated at a time", err.msg)

        # the one-dimensional elements are scalars named by the user
        err = try
            @eval @def begin
                t ∈ [0, 1], time
                y ∈ R, state
                w ∈ R, control
                ẏ(t) == w
                y(0) == 0
                y(1) → min
            end
            nothing
        catch e
            e
        end
        @test err isa ParsingError
        @test occursin("The control `w` is used here without being evaluated at a time", err.msg)
        @test !occursin("##", err.msg)

        # what is valid is unchanged
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ẋ(t) == [x₂(t), u(t)]
            x(0) == [0, 0]
            x₁(1) + x₂(0) ≤ 3
            x₁(t) + u(t) ≤ 5
            -1 ≤ u(t) ≤ 1
            ∫(x₁(t)^2 + u(t)^2) + x₁(1) → min
        end
        @test state_dimension(ocp) == 2
        @test control_dimension(ocp) == 1
        t0 = 0
        tf = 1
        ocp = @eval @def begin
            t ∈ [$t0, $tf], time
            x ∈ R², state
            u ∈ R, control
            ẋ(t) == [x₂(t), u(t)]
            x($t0) == [0, 0]
            x($tf) == [1, 0]
            ∫(u(t)^2) → min
        end
        @test state_dimension(ocp) == 2
    end

    @testset "stray check with an undeclared state or control (#193)" begin
        println("stray check, undeclared state or control (bis)")

        # a parser whose times are declared but whose state and control are not: this is what
        # remains after the declaration of the state or of the control has been rejected
        function parser(; x=nothing, u=nothing)
            p = CTParser.ParsingInfo()
            p.t = :t
            p.t0 = 0
            p.tf = 1
            p.x = x
            p.u = u
            return p
        end

        # Unit: nothing to look for when neither is declared, whatever the times
        p = parser()
        @test isnothing(CTParser.__stray_error(p, :(x(0) == 0)))
        @test isnothing(CTParser.__stray_error(p, :(x(1) + u(t) ≤ 1)))
        @test isnothing(CTParser.__stray_error(p, :(x + u)))

        # Unit: only the declared one is looked for
        p = parser(; x=:x)
        @test isnothing(CTParser.__stray_error(p, :(x(0) + u == 0)))
        @test CTParser.__stray_error(p, :(x + 1 ≤ 0)) isa Expr
        p = parser(; u=:u)
        @test isnothing(CTParser.__stray_error(p, :(x(0) + u(t) == 0)))
        @test CTParser.__stray_error(p, :(u + 1 ≤ 0)) isa Expr

        # Unit: the times are optional too
        p = CTParser.ParsingInfo()
        @test isnothing(CTParser.__stray_error(p, :(x(0) + u(t) == 0)))

        # Integration: a rejected declaration gives its own ParsingError, not a MethodError
        # raised by the check of the statements that follow it (initial and final conditions,
        # Mayer and Lagrange costs, dynamics)
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x = (q,) ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) → min
        end
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x = (q,) ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) == 1
            ∫(u(t)^2) → min
        end
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u = (z,) ∈ R, control
            ẋ(t) == z(t)
            x(0) == 0
            x(1) → min
        end
    end

    @testset "p_mayer! precondition errors" begin
        println("p_mayer! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "mayer test"
        p_ocp = :p_ocp
        e = :(x(tf))

        # No state declared
        ex = CTParser.p_mayer!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State declared but no t0
        p.x = :x
        ex = CTParser.p_mayer!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + t0 but no tf
        p.t0 = 0
        ex = CTParser.p_mayer!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # All declared → should build successfully
        p.tf = :tf
        ex = CTParser.p_mayer!(p, p_ocp, e, :min)
        @test ex isa Expr
        @test p.criterion == :min

        # Test trailing integral error
        p2 = CTParser.ParsingInfo()
        p2.lnum = 1
        p2.line = "mayer integral error test"
        p2.x = :x
        p2.t0 = 0
        p2.tf = 1
        e_bad = :(x(0) + ∫(u(t)))  # trailing ∫
        ex2 = CTParser.p_mayer!(p2, p_ocp, e_bad, :min)
        @test ex2 isa Expr
        @test_throws ParsingError eval(ex2)
    end

    @testset "p_bolza! precondition errors" begin
        println("p_bolza! preconditions (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "bolza test"
        p_ocp = :p_ocp
        e1 = :(x(tf))
        e2 = :(u(t)^2)

        # No state declared
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State declared but no t0
        p.x = :x
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + t0 but no tf
        p.t0 = 0
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + t0 + tf but no control
        p.tf = :tf
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # State + t0 + tf + control but no time
        p.u = :u
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # All declared → should build successfully
        p.t = :t
        ex = CTParser.p_bolza!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
        @test p.criterion == :min
    end

    # ============================================================================
    # PARSE! - Direct tests for unknown syntax
    # ============================================================================

    @testset "parse! unknown syntax" begin
        println("parse! unknown syntax (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 0
        p.line = ""
        p_ocp = :p_ocp

        # Unknown syntax should return a __throw expression
        ex = CTParser.parse!(p, p_ocp, :(unknown_syntax_that_does_not_match))
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Another unknown pattern (using a valid but unrecognized expression)
        p2 = CTParser.ParsingInfo()
        ex2 = CTParser.parse!(p2, :p_ocp, :(foo ~ bar))
        @test ex2 isa Expr
        @test_throws ParsingError eval(ex2)
    end

    @testset "constraint bound errors suggest a fix (#352)" begin
        println("constraint bound errors (bis)")

        # the error of the first constraint that fails, as raised through @def
        function bound_error(constraint)
            body = quote
                v ∈ R, variable
                t ∈ [0, 2], time
                x ∈ R², state
                u ∈ R, control
            end
            push!(body.args, constraint)
            push!(body.args, :(ẋ(t) == [x₂(t), u(t)]))
            return try
                Core.eval(Main, :(CTParser.@def $body))
                nothing
            catch e
                e
            end
        end

        # the example of the issue: the sides are in the wrong order
        err = bound_error(:(1 ≤ x₂(t)))
        @test err isa ParsingError
        @test occursin("The upper bound `x₂(t)` of this constraint depends on the state", err.msg)
        @test occursin("Line 5: 1 ≤ x₂(t)", err.msg)
        @test err.suggestion == "put the expression that depends on the state first: write `x₂(t) ≥ 1`"
        # ... and the other way round for a lower bound
        err = bound_error(:(1 ≥ x₂(t)))
        @test err isa ParsingError
        @test occursin("The lower bound `x₂(t)`", err.msg)
        @test err.suggestion == "put the expression that depends on the state first: write `x₂(t) ≤ 1`"

        # an equality: the term moves to the constrained side
        err = bound_error(:(x₂(0) == v))
        @test err isa ParsingError
        @test occursin("depends on the variable", err.msg)
        @test err.suggestion == "move the term to the constrained side: write `x₂(0) - v == 0`"

        # both sides depend on something: the term moves to the constrained side
        err = bound_error(:(x₁(t) ≤ u(t)))
        @test err isa ParsingError
        @test occursin("The upper bound `u(t)` of this constraint depends on the control", err.msg)
        @test err.suggestion == "move the term to the constrained side: write `x₁(t) - u(t) ≤ 0`"
        err = bound_error(:(u(t) ≥ x₁(t)))
        @test occursin("The lower bound `x₁(t)`", err.msg)
        @test err.suggestion == "move the term to the constrained side: write `x₁(t) - u(t) ≤ 0`"

        # a bound that depends on the time
        err = bound_error(:(x₁(t) ≤ t))
        @test err isa ParsingError
        @test occursin("depends on the time", err.msg)

        # a two-sided constraint: the dependent bound is named
        err = bound_error(:(0 ≤ x₂(0) ≤ v))
        @test err isa ParsingError
        @test occursin("The upper bound `v` of this constraint depends on the variable", err.msg)
        @test err.suggestion == "move the term to the constrained side: write `x₂(0) - v ≤ 0`"

        # a labelled constraint is read the same way
        err = bound_error(:((1 ≤ x₂(t), c1)))
        @test err isa ParsingError
        @test err.suggestion == "put the expression that depends on the state first: write `x₂(t) ≥ 1`"

        # the operands of a constraint statement, in the roles of p_constraint!
        parts = CTParser.__constraint_parts
        @test parts(:(a == b)) == (:b, :a, :b)
        @test parts(:((a == b, c1))) == (:b, :a, :b)
        @test parts(:(l ≤ e ≤ r)) == (:l, :e, :r)
        @test parts(:((l ≤ e ≤ r, c1))) == (:l, :e, :r)
        @test parts(:(e ≤ r)) == (nothing, :e, :r)
        @test parts(:((e ≤ r, c1))) == (nothing, :e, :r)
        @test parts(:(r ≥ e ≥ l)) == (:l, :e, :r)
        @test parts(:((r ≥ e ≥ l, c1))) == (:l, :e, :r)
        @test parts(:(e ≥ l)) == (:l, :e, nothing)
        @test parts(:((e ≥ l, c1))) == (:l, :e, nothing)
        @test isnothing(parts(:(∫(u(t)) → min)))
        @test isnothing(parts(:x))

        # the message no longer carries the hard-coded example of the old suggestion
        @test !occursin("x₂(0) - v == 0` rather than", err.msg)

        # constraints whose bounds are constants are still fine
        ocp = @eval @def begin
            t ∈ [0, 2], time
            x ∈ R², state
            u ∈ R, control
            x₂(t) ≥ 1
            -1 ≤ u(t) ≤ 1
            ẋ(t) == [x₂(t), u(t)]
            x(0) == [0, 1]
            x₁(2) → min
        end
        @test state_dimension(ocp) == 2
    end

    @testset "parse! malformed declarations (#4)" begin
        println("parse! malformed declarations (bis)")

        # the shape test: only `(names) ∈ dom, kind` is a malformed declaration
        @test CTParser.__is_malformed_declaration(:((z,) ∈ R, control))
        @test CTParser.__is_malformed_declaration(:((z) ∈ R, state))
        @test CTParser.__is_malformed_declaration(:([z] ∈ R, variable))
        @test !CTParser.__is_malformed_declaration(:(x₁(t)^2))
        @test !CTParser.__is_malformed_declaration(:(a, b))
        @test !CTParser.__is_malformed_declaration(:((z,) ∈ R, other))

        # the error is a ParsingError, with the offending line and a suggestion
        function declaration_error(ex)
            return try
                eval(ex)
                nothing
            catch e
                e
            end
        end

        # (Meta.parse: inside a `:( … )` quote, `u = (z,) ∈ R, control` is a tuple, not an
        # assignment as it is on a line of a @def block)
        for (line, kind) in (
            ("u = (z,) ∈ R, control", "control"),
            ("u = (z) ∈ R, control", "control"),
            ("x = (q,) ∈ R, state", "state"),
            ("w = (a,) ∈ R, variable", "variable"),
            ("u = (z, w) ∈ R, control", "control"),
        )
            p = CTParser.ParsingInfo()
            ex = CTParser.parse!(p, :p_ocp, Meta.parse(line))
            err = declaration_error(ex)
            @test err isa ParsingError
            @test occursin("invalid $kind declaration", err.msg)
            @test occursin("named components need a domain `R^n`", err.msg)
            @test occursin("Line 1:", err.msg)
            @test occursin("∈ R, $kind", err.suggestion)
        end

        # through @def: no silent alias, whatever the kind
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u = (z,) ∈ R, control
            ẋ(t) == [x₂(t), z(t)]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x = (q,) ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) → min
        end
        @test_throws ParsingError @eval @def begin
            w = (a,) ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) → min
        end

        # what is valid is unchanged
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u = (z, w) ∈ R², control
            ẋ(t) == [z(t), w(t)]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test control_dimension(ocp) == 2
        @test control_name(ocp) == "u"
        @test control_components(ocp) == ["z", "w"]
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) → min
        end
        @test control_dimension(ocp) == 1
        @test control_components(ocp) == ["u"]
        # ... and so is a name for the only component with an explicit R^1 (#4): R and R^1
        # are treated alike, so there is a single way to declare a scalar
        for (line, kind) in (
            ("u = (z,) ∈ R^1, control", "control"),
            ("u = (z) ∈ R^1, control", "control"),
            ("u = [z] ∈ R^1, control", "control"),
            ("u = (z,) ∈ R¹, control", "control"),
            ("x = (q,) ∈ R^1, state", "state"),
            ("w = (a,) ∈ R^1, variable", "variable"),
        )
            p = CTParser.ParsingInfo()
            ex = CTParser.parse!(p, :p_ocp, Meta.parse(line))
            err = declaration_error(ex)
            @test err isa ParsingError
            @test occursin("invalid $kind declaration", err.msg)
            @test occursin("a one-dimensional $kind has no component names", err.msg)
            @test occursin("∈ R, $kind", err.suggestion)
        end
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u = (z,) ∈ R^1, control
            ẋ(t) == z(t)
            x(0) == 0
            x(1) → min
        end
        # a symbolic dimension is not looked at: n is not known when the block is parsed
        n = 2
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u = (a, b) ∈ R^$n, control
            ẋ(t) == a(t) + b(t)
            x(0) == 0
            x(1) → min
        end
        @test control_components(ocp) == ["a", "b"]
        # R^1 without names is still a scalar
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R^1, control
            ẋ(t) == u(t)
            x(0) == 0
            x(1) → min
        end
        @test control_dimension(ocp) == 1
        # a plain alias is still an alias
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            a = x₁(t)^2
            ẋ(t) == [x₂(t), u(t) + a]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test control_dimension(ocp) == 1
    end

    @testset "parse! `=` written instead of `==` (#353)" begin
        println("parse! equation with = (bis)")

        # __as_written: no line-number node, no wrapper around a single statement
        ex = :(f(t) = u(t))
        @test ex.args[2] isa Expr && ex.args[2].head == :block  # the lowered form
        @test CTParser.__as_written(ex) == Expr(:(=), :(f(t)), :(u(t)))
        @test string(CTParser.__as_written(ex)) == "f(t) = u(t)"
        @test CTParser.__as_written(:x) === :x
        @test CTParser.__as_written(:(a + b)) == :(a + b)
        blk = CTParser.__as_written(Expr(:block, :a, LineNumberNode(1, :f), :b))
        @test blk.head == :block && length(blk.args) == 2

        # the shape test: only `f(args) = body` is an equation written with `=`
        @test CTParser.__is_equation_with_assign(:(ẋ(t) = u(t)))
        @test CTParser.__is_equation_with_assign(:(x₁(0) = 1))
        @test !CTParser.__is_equation_with_assign(:(a = x₁(t)^2))
        @test !CTParser.__is_equation_with_assign(:(ẋ(t) == u(t)))
        @test !CTParser.__is_equation_with_assign(:(a + b))

        function assign_error(src)
            p = CTParser.ParsingInfo()
            ex = CTParser.parse!(p, :p_ocp, Meta.parse(src))
            return try
                eval(ex)
                nothing
            catch e
                e
            end
        end

        for (src, hint) in (
            ("ẋ(t) = u(t)", "ẋ(t) == u(t)"),
            ("ẋ₁(t) = x₂(t)", "ẋ₁(t) == x₂(t)"),
            ("x(0) = [0, 1]", "x(0) == [0, 1]"),
            ("x₂(t) = 1", "x₂(t) == 1"),
        )
            err = assign_error(src)
            @test err isa ParsingError
            @test occursin("written with `==`", err.msg)
            # the line is shown as the user wrote it
            @test occursin("Line 1: $src", err.msg)
            @test !occursin("#=", err.msg)
            @test !occursin("begin", err.msg)
            @test err.suggestion == "write `$hint`"
        end

        # through @def, with the derivative of the state
        err = try
            @eval @def begin
                t ∈ [0, 1], time
                x ∈ R, state
                u ∈ R, control
                ẋ(t) = u(t)
                x(0) == 0
                x(1) → min
            end
            nothing
        catch e
            e
        end
        @test err isa ParsingError
        @test occursin("Line 4: ẋ(t) = u(t)", err.msg)
        @test err.suggestion == "write `ẋ(t) == u(t)`"
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x = (q, v) ∈ R², state
            u ∈ R, control
            q̇ = v
            v̇ = u
            ẋ(t) = [q̇, v̇]
            x(0) == [0, 0]
            x(1) → min
        end

        # an alias is still an alias, and the equation with == still works
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x = (q, v) ∈ R², state
            u ∈ R, control
            q̇ = v(t)
            v̇ = u(t)
            ẋ(t) == [q̇, v̇]
            x(0) == [0, 0]
            x(1) → min
        end
        @test state_dimension(ocp) == 2
    end

    # ============================================================================
    # P_TIME! - Additional precondition errors
    # ============================================================================

    @testset "p_time! bad time declarations" begin
        println("p_time! bad declarations (bis)")

        # Variable declared, but t0 depends on wrong variable expression
        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "time bad declaration test"
        p.v = :v
        p_ocp = :p_ocp

        # t0 = v (not v[i]) when v is the variable → bad
        ex = CTParser.p_time!(p, p_ocp, :t, :(v + 1), 1)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # tf = v expression when v is the variable → bad
        p2 = CTParser.ParsingInfo()
        p2.lnum = 1
        p2.line = "time bad declaration test 2"
        p2.v = :v
        ex2 = CTParser.p_time!(p2, :p_ocp, :t, 0, :(v * 2))
        @test ex2 isa Expr
        @test_throws ParsingError eval(ex2)
    end

    # ============================================================================
    # P_CONSTRAINT! - :other constraint type error (invalid constraints)
    # ============================================================================

    @testset "p_constraint_fun! :other constraint type error" begin
        println("p_constraint_fun! :other type (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint :other test"
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.x = :x
        p.u = :u
        p.v = :v
        p_ocp = :p_ocp

        # Constraint with :other type should raise an error
        # This simulates what happens when constraint_type returns :other
        ex = CTParser.p_constraint_fun!(p, p_ocp, 0, :(x[1](0) + u[1](t)), 1, :other, :c1)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Another :other case
        ex2 = CTParser.p_constraint_fun!(
            p, p_ocp, nothing, :(x[1](0) * u[1](t) + u[2](t)^2), 1, :other, :c2
        )
        @test ex2 isa Expr
        @test_throws ParsingError eval(ex2)
    end

    @testset "p_constraint! detects :other constraint type" begin
        println("p_constraint! detects :other (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint detection test"
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.x = :x
        p.u = :u
        p.v = :v
        p.dim_x = 2
        p.dim_u = 2
        p_ocp = :p_ocp

        # Test that p_constraint! correctly identifies invalid constraints
        # Mixed initial state and control: x1(0) * u1(t) + u2(t)^2 <= 1
        # This should result in constraint_type returning :other
        ex = CTParser.p_constraint!(p, p_ocp, nothing, :(x[1](0) * u[1](t) + u[2](t)^2), 1)
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # Mixed final state and control at initial time: x(tf) + u(t0)
        # This should also result in :other
        ex2 = CTParser.p_constraint!(p, p_ocp, 0, :(x[1](tf) + u[1](0)), 1)
        @test ex2 isa Expr
        @test_throws ParsingError eval(ex2)

        # Control at initial and final time: u(t0) + u(tf)
        ex3 = CTParser.p_constraint!(p, p_ocp, 0, :(u[1](0) + u[1](tf)), 1)
        @test ex3 isa Expr
        @test_throws ParsingError eval(ex3)
    end

    # ============================================================================
    # @def MACRO - Invalid constraints that should raise ParsingError
    # ============================================================================

    @testset "@def macro :other constraint type error" begin
        println("@def macro :other constraint (bis)")

        # Test 1: Mixed initial state and control - x1(0) * u1(t) + u2(t)^2 <= 1
        # This should trigger constraint_type to return :other and raise ParsingError
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R², control
            x[1](0) * u[1](t) + u[2](t)^2 ≤ 1
            x(0) == [0, 0]
            x(1) == [1, 1]
            ẋ(t) == [u[1](t), u[2](t)]
            ∫(u[1](t)^2 + u[2](t)^2) → min
        end

        # Test 2: Mixed final state and control at initial time - x(tf) + u(t0)
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            x(1) + u(0) ≤ 1
            x(0) == 0
            ẋ(t) == u(t)
            ∫(u(t)^2) → min
        end

        # Test 3: Control at both initial and final time - u(t0) + u(tf)
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            u(0) + u(1) ≤ 1
            x(0) == 0
            x(1) == 1
            ẋ(t) == u(t)
            ∫(u(t)^2) → min
        end

        # Test 4: Another invalid mixing - state at t and control at t0
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            x(t) + u(0) ≤ 1
            x(0) == 0
            x(1) == 1
            ẋ(t) == u(t)
            ∫(u(t)^2) → min
        end
    end

    @testset "aliases defined from other aliases (#389)" begin
        println("aliases defined from other aliases")
        x = [1.0, 2.0]
        u = 3.0

        function rhs(o)
            r = zeros(2)
            CTModels.dynamics(o)(r, 0.0, x, u, Float64[])
            return r
        end

        # an alias defined from an alias defined before it
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            a = x₂(t)
            b = a + u(t)
            ẋ(t) == [a, b]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test rhs(o) == [2.0, 5.0]

        # an alias that names an alias defined after it: the later alias is substituted next
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            a = c
            c = 5
            ẋ(t) == [a, u(t)]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test rhs(o) == [5.0, 3.0]

        # a chain written backwards
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            a = b
            b = c
            c = 3
            ẋ(t) == [a, u(t)]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test rhs(o) == [3.0, 3.0]

        # an alias defined from a component alias of the state
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            a = x₁(t)
            ẋ(t) == [a, u(t)]
            x(0) == [0, 0]
            x₁(1) → min
        end
        @test rhs(o) == [1.0, 3.0]
    end

    @testset "symbolic dimensions: R^n works, Rⁿ is rejected (#60)" begin
        println("symbolic dimensions (bis)")

        # `Rⁿ` is one identifier with a superscript letter, not `R^n`: only the digit
        # exponents R², R³, … are recognised, so a symbolic dimension cannot be read from it
        for (src, kind) in (
            ("x ∈ Rⁿ, state", "state"),
            ("u ∈ Rᵐ, control", "control"),
            ("v ∈ Rⁿ, variable", "variable"),
        )
            p = CTParser.ParsingInfo()
            ex = CTParser.parse!(p, :p_ocp, Meta.parse(src))
            err = try
                eval(ex)
                nothing
            catch e
                e
            end
            @test err isa ParsingError
            @test occursin("Line 1: ($src)", err.msg) || occursin("Line 1: $src", err.msg)
            @test occursin("unknown syntax", err.msg)
        end
        @test_throws ParsingError @eval @def begin
            t ∈ [0, 1], time
            x ∈ Rⁿ, state
            u ∈ R, control
            ẋ(t) == u(t)
            x(0) == 0
            x₁(1) → min
        end

        # `R^n` is the spelling of a symbolic dimension, for any expression
        n = 3
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R^$n, state
            u ∈ R, control
            ẋ(t) == -x(t)
            x(0) == zeros($n)
            x₁(1) → min
        end
        @test state_dimension(ocp) == 3
        ocp = @eval @def begin
            t ∈ [0, 1], time
            x ∈ R^($n + 1), state
            u ∈ R, control
            ẋ(t) == -x(t)
            x(0) == zeros($n + 1)
            x₁(1) → min
        end
        @test state_dimension(ocp) == 4

        # the automatic aliases of the components: x₁, x₂, … and x1, x2, … for a literal
        # dimension, up to 9 only for a symbolic one
        p = CTParser.ParsingInfo()
        CTParser.parse!(p, :p_ocp, :(x ∈ R^12, state))
        @test haskey(p.aliases, :x₁₂) && haskey(p.aliases, :x12)
        p = CTParser.ParsingInfo()
        CTParser.parse!(p, :p_ocp, :(x ∈ R^n, state))
        @test haskey(p.aliases, :x₉) && haskey(p.aliases, :x9)
        @test !haskey(p.aliases, :x₁₀) && !haskey(p.aliases, :x10)

        # R², R³, … are aliases of R^2, R^3, … up to R²⁰
        p = CTParser.ParsingInfo()
        @test p.aliases[:R²] == :(R^2)
        @test p.aliases[:R²⁰] == :(R^20)
        @test !haskey(p.aliases, :R²¹)
    end

    @testset "generated :fun functions do not allocate on a vector literal (#21)" begin
        println("allocations of the generated functions")

        # the call is wrapped in a function: @allocated of a global call would count boxing
        calls = (f, args...) -> (f(args...); @allocated f(args...))

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x₁(0) + x₂(1) == 1
            [x₁(0) + x₂(1), x₃(1)] == [1, 2]
            [x₁(t) + u(t), x₂(t) * x₃(t)] ≤ [1, 2]
            ẋ(t) == [x₂(t), x₃(t), u(t)]
            ∫(u(t)^2) → min
        end
        x = [0.3, 0.7, 0.5]
        u = 0.5
        v = Float64[]

        # values
        r = zeros(3)
        CTModels.dynamics(o)(r, 0.0, x, u, v)
        @test r == [0.7, 0.5, 0.5]
        _, path, _, _ = CTModels.path_constraints_nl(o)
        rp = zeros(2)
        path(rp, 0.0, x, u, v)
        @test rp == [0.3 + 0.5, 0.7 * 0.5]
        _, bd, _, _ = CTModels.boundary_constraints_nl(o)
        rb = zeros(3)
        bd(rb, x, x, v)
        @test rb == [0.3 + 0.7, 0.3 + 0.7, 0.5]

        # allocations
        @test calls(CTModels.dynamics(o), r, 0.0, x, u, v) == 0
        @test calls(path, rp, 0.0, x, u, v) == 0
        @test calls(bd, rb, x, x, v) == 0

        # the entries are evaluated before any is written: the output may share the input
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ẋ(t) == [x₂(t), x₁(t)]
            ∫(u(t)^2) → min
        end
        y = [1.0, 2.0]
        CTModels.dynamics(o)(y, 0.0, y, u, v)
        @test y == [2.0, 1.0]

        # a splat is not a plain list of entries: broadcast as before
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            ẋ(t) == [x₂(t), (x₃(t), u(t))...]
            ∫(u(t)^2) → min
        end
        r = zeros(3)
        CTModels.dynamics(o)(r, 0.0, x, u, v)
        @test r == [0.7, 0.5, 0.5]

        # the number of entries is still checked
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            ẋ(t) == [x₂(t), x₃(t)]
            ∫(u(t)^2) → min
        end
        @test_throws DimensionMismatch CTModels.dynamics(o)(zeros(3), 0.0, x, u, v)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ẋ(t) == [x₂(t), u(t), x₁(t)]
            ∫(u(t)^2) → min
        end
        @test_throws DimensionMismatch CTModels.dynamics(o)(zeros(2), 0.0, x[1:2], u, v)
    end
end
