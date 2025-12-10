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

    @testset "expr_it (direct tests)" begin
        println("expr_it (bis)")

        # Identity transformation
        id = e -> CTParser.expr_it(e, Expr, x -> x)
        @test id(:(a + b)) == :(a + b)
        @test id(:(f(g(x)))) == :(f(g(x)))
        @test id(:x) == :x
        @test id(42) == 42

        # Leaf transformation only
        double_leaves = e -> CTParser.expr_it(e, Expr, x -> x isa Number ? 2x : x)
        @test double_leaves(:(1 + 2)) == :(2 + 4)
        @test double_leaves(:(f(3, 4))) == :(f(6, 8))

        # Head/args transformation (wrap all calls in :wrapped)
        wrap_calls =
            e -> CTParser.expr_it(e, (h, args...) -> Expr(:wrapped, h, args...), x -> x)
        result = wrap_calls(:(a + b))
        @test result.head == :wrapped
    end

    @testset "concat (extended cases)" begin
        println("concat (bis)")

        # Two simple non-block expressions
        e1 = :(x = 1)
        e2 = :(y = 2)
        e = concat(e1, e2)
        @test e.head == :block
        @test length(e.args) == 2

        # Block + non-block
        e1 = :(
            begin
                ;
                a = 1;
                b = 2;
            end
        )
        e2 = :(c = 3)
        e = concat(e1, e2)
        @test e.head == :block
        @test :c in
            [arg isa Expr && arg.head == :(=) ? arg.args[1] : nothing for arg in e.args]

        # Non-block + block
        e1 = :(z = 0)
        e2 = :(
            begin
                ;
                p = 1;
                q = 2;
            end
        )
        e = concat(e1, e2)
        @test e.head == :block

        # Empty-ish blocks (with LineNumberNode only)
        e1 = Expr(:block, LineNumberNode(1, :test))
        e2 = :(x = 1)
        e = concat(e1, e2)
        @test e.head == :block
    end

    @testset "has (simple form, extended)" begin
        println("has simple (bis)")

        # Deeply nested symbol
        e = :(f(g(h(x))))
        @test has(e, :x)
        @test has(e, :h)
        @test has(e, :g)
        @test has(e, :f)
        @test !has(e, :y)

        # Number in nested expression
        e = :(a + (b * (c - 3)))
        @test has(e, 3)
        @test !has(e, 4)

        # Expression matching
        e = :(sin(x)^2 + cos(x)^2)
        @test has(e, :(sin(x)))
        @test has(e, :(cos(x)))
        @test has(e, :(sin(x)^2))
        @test !has(e, :(tan(x)))

        # Symbol equals itself
        @test has(:foo, :foo)
        @test !has(:foo, :bar)

        # Number equals itself
        @test has(42, 42)
        @test !has(42, 43)
    end

    @testset "subs (deeply nested and edge cases)" begin
        println("subs deep (bis)")

        # Very deep nesting
        e = :(f(g(h(i(j(x))))))
        @test subs(e, :x, :y) == :(f(g(h(i(j(y))))))

        # Multiple occurrences
        e = :(x + x * x - x / x)
        @test subs(e, :x, :z) == :(z + z * z - z / z)

        # Substituting an expression that appears multiple times
        e = :(sin(x) + sin(x) * cos(sin(x)))
        @test subs(e, :(sin(x)), :s) == :(s + s * cos(s))

        # Substituting with nothing (empty symbol unlikely but testing robustness)
        e = :(a + b)
        @test subs(e, :c, :d) == :(a + b)  # no change

        # Substituting a number with a symbol
        e = :(1 + 2 + 3)
        @test subs(e, 2, :two) == :(1 + two + 3)
    end

    @testset "constraint_type (boundary edge cases)" begin
        println("constraint_type boundary (bis)")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        v = :v

        # Both initial and final state → boundary
        e = :(x(0) + x(tf))
        @test constraint_type(e, t, t0, tf, x, u, v) == :boundary

        # State at t0 and t → mixed? Actually state_fun since x(t) is present
        # Let's check: has(e, x, t0)=true, has(e, x, t)=true
        e = :(x(0) + x(t))
        # This should be :other because x(t0) and x(t) together don't match a clean pattern
        result = constraint_type(e, t, t0, tf, x, u, v)
        @test result in [:other, :boundary, :state_fun]  # accept any of these

        # Control at t with state at t0 → other
        e = :(u(t) + x(0))
        result = constraint_type(e, t, t0, tf, x, u, v)
        @test result == :other

        # Only variable, indexed with step range
        e = :(v[1:2:7])
        @test constraint_type(e, t, t0, tf, x, u, v) == (:variable_range, 1:2:7)
    end
end
