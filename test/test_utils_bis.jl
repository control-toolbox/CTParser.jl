# test_utils_bis

function test_utils_bis()
    @testset "subs (additional cases)" begin
        println("subs (bis)")

        # literal replacement (Real)
        e = :(x + 2)
        @test subs(e, 2, :two) == :(x + two)

        # non-trivial expression replacement (Expr → Expr)
        e = :(f(g(x)) + g(x))
        @test subs(e, :(g(x)), :h) == :(f(h) + h)

        # ensure unrelated symbols are untouched
        e = :(g(y) + g(x))
        @test subs(e, :(g(x)), :h) == :(g(y) + h)
    end

    @testset "replace_call (edge cases)" begin
        println("replace_call (bis)")

        t = :t
        x = :x
        u = :u

        # no matching call → expression unchanged
        e = :(A * x(s) + B * u(s))
        @test replace_call(e, x, t, :xx) == e

        # mixed: only calls with the right time symbol are replaced
        e = :(A * x(t) + B * x(s))
        @test replace_call(e, x, t, :xx) == :(A * xx + B * x(s))

        # vector form: one symbol appears, the other does not
        e = :(F(x(t)) + G(y(t)))
        @test replace_call(e, [x, u], t, [:xx, :uu]) == :(F(xx) + G(y(t)))
    end

    @testset "has (edge cases)" begin
        println("has (bis)")

        e = :(∫(x[1](t)^2 + 2 * u(t0)) → min)
        t = :t
        t0 = :t0
        x = :x
        u = :u

        # symbol appears, but not with the queried time → should be false
        @test !has(e, u, t)
        @test has(e, u, t0)

        # nested call where the inner expression contains the symbol
        e = :(H(x(t)) + K())
        @test has(e, x, t)
    end

    @testset "constraint_type (additional patterns)" begin
        println("constraint_type (bis)")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        v = :v

        # mixed state/control inside a nonlinear expression
        e = :((x(t) + u(t))^2)
        @test constraint_type(e, t, t0, tf, x, u, v) == :mixed

        # control fun combined with a variable-only term → still mixed
        e = :(2u[1](t)^2 * x(t) + v^3)
        @test constraint_type(e, t, t0, tf, x, u, v) == :mixed

        # variable-only nonlinear expression without ranges → variable_fun
        e = :(sin(v) + v^3)
        @test constraint_type(e, t, t0, tf, x, u, v) == :variable_fun
    end

    @testset "subs2/3/4/5 (pathological cases)" begin
        println("subs2/3/4/5 (bis)")

        e = :(x0[1] * 2xf[3])

        # symbol does not appear at all → expression unchanged
        @test subs2(e, :z, :x, 0) == e
        @test subs3(e, :z, :x, :i, 0) == e
        @test subs4(e, :z, :z, :i) == e
        @test subs5(e, :z, :x, 0) == e
    end

    @testset "replace_call (errors)" begin
        println("replace_call (errors)")

        t = :t
        x = :x
        u = :u

        e = :(x(t) + u(t))

        # length mismatch between x and y should trigger an assertion failure
        @test_throws AssertionError replace_call(e, [x, u], t, [:xx])
    end

    @testset "constraint_type (pathological cases)" begin
        println("constraint_type (pathological)")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        v = :v

        # expression with no relevant symbols
        e = :(42)
        @test constraint_type(e, t, t0, tf, x, u, v) == :other

        # unrelated function of time
        e = :(w(t))
        @test constraint_type(e, t, t0, tf, x, u, v) == :other

        # variable with time argument → classified as variable_fun
        e = :(v(t))
        @test constraint_type(e, t, t0, tf, x, u, v) == :variable_fun

        # control evaluated only at initial / final time → not a control_range
        e = :(u(0))
        @test constraint_type(e, t, t0, tf, x, u, v) == :other

        e = :(u(tf))
        @test constraint_type(e, t, t0, tf, x, u, v) == :other
    end
end
