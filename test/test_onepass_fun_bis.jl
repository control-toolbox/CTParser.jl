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
end
