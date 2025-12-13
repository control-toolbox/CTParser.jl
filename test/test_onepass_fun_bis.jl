# test_onepass_fun_bis

function test_onepass_fun_bis()
    @testset "error helpers (__throw and __wrap)" begin
        println("error helpers (bis)")

        # __throw should build an expression that raises a ParsingError when evaluated
        ex = CTParser.__throw("test message", 42, "dummy line")
        @test ex isa Expr
        @test_throws ParsingError eval(ex)

        # __wrap should leave non-throwing code unchanged and propagate its value
        wrapped = CTParser.__wrap(:(1 + 1), 1, "line")
        @test wrapped isa Expr
        @test eval(wrapped) == 2

        # __wrap should catch and rethrow the original exception
        wrapped_err = CTParser.__wrap(:(error("boom")), 1, "line")
        @test wrapped_err isa Expr
        @test_throws ErrorException eval(wrapped_err)
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
end
