# test_utils

function test_utils()
    @testset "subs" begin
        println("subs")

        e = :(∫(r(t)^2 + 2u₁(t)) → min)
        @test subs(e, :r, :(x[1])) == :(∫((x[1])(t)^2 + 2 * u₁(t)) → min)

        e = :(∫(u₁(t)^2 + 2u₂(t)) → min)
        for i in 1:2
            e = subs(e, Symbol(:u, Char(8320 + i)), :(u[$i]))
        end
        @test e == :(∫((u[1])(t)^2 + 2 * (u[2])(t)) → min)

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        e = :(x[1](0) * 2x(tf) - x[2](tf) * 2x(0))
        x0 = Symbol(x, 0)
        @test subs(e, :($x[1]($(t0))), :($x0[1])) ==
            :(x0[1] * (2 * x(tf)) - (x[2])(tf) * (2 * x(0)))
    end

    @testset "subs2" begin
        println("subs2")

        e = :(x0[1] * 2xf[3] - cos(xf[2]) * 2x0[2])
        @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
            :(x[1, 0] * (2 * x[3, N]) - cos(x[2, N]) * (2 * x[2, 0]))

        e = :(x0 * 2xf[3] - cos(xf) * 2x0[2])
        @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
            :(x0 * (2 * x[3, N]) - cos(xf) * (2 * x[2, 0]))
    end

    @testset "subs3" begin
        println("subs3")

        e = :(x0[1:2:d] * 2xf[1:3])
        @test subs3(e, :x0, :x, :i, 0) == :(x[i, 0] * (2 * xf[1:3]))
        @test subs3(e, :xf, :x, 1, :N) == :(x0[1:2:d] * (2 * x[1, N]))
    end

    @testset "subs4" begin
        println("subs4")

        e = :(v[1:2:d] * 2xf[1:3])
        @test subs4(e, :v, :v, :i) == :(v[i] * (2 * xf[1:3]))
        @test subs4(e, :xf, :xf, 1) == :(v[1:2:d] * (2 * xf[1]))
    end

    @testset "subs5" begin
        println("subs5")

        e = :(x0[1] * 2xf[3] - cos(xf[2]) * 2x0[2])
        @test subs5(subs5(e, :x0, :x, 0), :xf, :x, :N) == :(
            ((x[1, 0] + x[1, 0 + 1]) / 2) * (2 * ((x[3, N] + x[3, N + 1]) / 2)) -
            cos((x[2, N] + x[2, N + 1]) / 2) * (2 * ((x[2, 0] + x[2, 0 + 1]) / 2))
        )

        e = :(x0 * 2xf[3] - cos(xf) * 2x0[2])
        @test subs5(subs5(e, :x0, :x, 0), :xf, :x, :N) == :(
            x0 * (2 * ((x[3, N] + x[3, N + 1]) / 2)) -
            cos(xf) * (2 * ((x[2, 0] + x[2, 0 + 1]) / 2))
        )
    end

    @testset "replace_call" begin
        println("replace_call")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        e = :(x[1](0) * 2x(tf) - x[2](tf) * 2x(0))
        x0 = Symbol(x, "#0")
        xf = Symbol(x, "#f")
        e = replace_call(e, x, t0, x0)
        @test replace_call(e, x, tf, xf) ==
            :(var"x#0"[1] * (2var"x#f") - var"x#f"[2] * (2var"x#0"))

        e = :(A * x(t) + B * u(t))
        @test replace_call(replace_call(e, x, t, x), u, t, u) == :(A * x + B * u)

        e = :(F0(x(t)) + u(t) * F1(x(t)))
        @test replace_call(replace_call(e, x, t, x), u, t, u) == :(F0(x) + u * F1(x))

        e = :(0.5u(t)^2)
        @test replace_call(e, u, t, u) == :(0.5 * u^2)

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        e = :((x^2 + u[1])(t))
        @test replace_call(e, [x, u], t, [:xx, :uu]) == :(xx^2 + uu[1])

        e = :(((x^2)(t) + u[1])(t))
        @test replace_call(e, [x, u], t, [:xx, :uu]) == :(xx^2 + uu[1])

        e = :(((x^2)(t0) + u[1])(t))
        @test replace_call(e, [x, u], t, [:xx, :uu]) == :((xx^2)(t0) + uu[1])
    end

    @testset "has" begin
        println("has")

        e = :(∫(x[1](t)^2 + 2 * u(t)) → min)
        @test has(e, :x, :t)
        @test has(e, :u, :t)
        @test !has(e, :v, :t)
        @test has(e, 2)
        @test has(e, :x)
        @test has(e, :min)
        @test has(e, :(x[1](t)^2))
        @test !has(e, :(x[1](t)^3))
        @test !has(e, 3)
        @test !has(e, :max)
        @test has(:x, :x)
        @test !has(:x, 2)
        @test !has(:x, :y)
    end

    @testset "concat" begin
        println("concat")

        e1 = :(x = 1; y = 2)
        e2 = :(z = 3)
        e = concat(e1, e2)
        @test length(e.args) == length(e1.args) + 1

        e1 = :(z = 3)
        e2 = :(x = 1; y = 2)
        e = concat(e1, e2)
        @test length(e.args) == 1 + length(e2.args)

        e1 = :(z = 3)
        e = concat(e1, e1)
        @test length(e.args) == 2

        e1 = :(x = 1; y = 2)
        e = concat(e1, e1)
        @test length(e.args) == 2 * length(e1.args)
    end

    @testset "constraint_type" begin
        println("constraint_type")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        v = :v
        @test constraint_type(:(ẏ(t)), t, t0, tf, x, u, v) == :other
        @test constraint_type(:(ẋ(s)), t, t0, tf, x, u, v) == :other
        @test constraint_type(:(x(0)'), t, t0, tf, x, u, v) == :boundary
        @test constraint_type(:(x'(0)), t, t0, tf, x, u, v) == :boundary
        @test constraint_type(:(x(t)'), t, t0, tf, x, u, v) == :state_fun
        @test constraint_type(:(x(0)), t, t0, tf, x, u, v) == (:initial, nothing)
        @test constraint_type(:(x[1:2:5](0)), t, t0, tf, x, u, v) == (:initial, 1:2:5)
        @test constraint_type(:(x[1:2](0)), t, t0, tf, x, u, v) == (:initial, 1:2)
        @test constraint_type(:(x[1](0)), t, t0, tf, x, u, v) == (:initial, 1)
        @test constraint_type(:(x[1:2](0)), t, t0, tf, x, u, v) == (:initial, 1:2)
        @test constraint_type(:(2x[1](0)^2), t, t0, tf, x, u, v) == :boundary
        @test constraint_type(:(x(tf)), t, t0, tf, x, u, v) == (:final, nothing)
        @test constraint_type(:(x[1:2:5](tf)), t, t0, tf, x, u, v) == (:final, 1:2:5)
        @test constraint_type(:(x[1:2](tf)), t, t0, tf, x, u, v) == (:final, 1:2)
        @test constraint_type(:(x[1](tf)), t, t0, tf, x, u, v) == (:final, 1)
        @test constraint_type(:(x[1:2](tf)), t, t0, tf, x, u, v) == (:final, 1:2)
        @test constraint_type(:(x[1](tf) - x[2](0)), t, t0, tf, x, u, v) == :boundary
        @test constraint_type(:(2x[1](tf)^2), t, t0, tf, x, u, v) == :boundary
        @test constraint_type(:(u[1:2:5](t)), t, t0, tf, x, u, v) == (:control_range, 1:2:5)
        @test constraint_type(:(u[1:2](t)), t, t0, tf, x, u, v) == (:control_range, 1:2)
        @test constraint_type(:(u[1](t)), t, t0, tf, x, u, v) == (:control_range, 1)
        @test constraint_type(:(u(t)), t, t0, tf, x, u, v) == (:control_range, nothing)
        @test constraint_type(:(2u[1](t)^2), t, t0, tf, x, u, v) == :control_fun
        @test constraint_type(:(x[1:2:5](t)), t, t0, tf, x, u, v) == (:state_range, 1:2:5)
        @test constraint_type(:(x[1:2](t)), t, t0, tf, x, u, v) == (:state_range, 1:2)
        @test constraint_type(:(x[1](t)), t, t0, tf, x, u, v) == (:state_range, 1)
        @test constraint_type(:(x(t)), t, t0, tf, x, u, v) == (:state_range, nothing)
        @test constraint_type(:(2x[1](t)^2), t, t0, tf, x, u, v) == :state_fun
        @test constraint_type(:(2u[1](t)^2 * x(t)), t, t0, tf, x, u, v) == :mixed
        @test constraint_type(:(2u[1](0)^2 * x(t)), t, t0, tf, x, u, v) == :other
        @test constraint_type(:(2u[1](t)^2 * x(t) + v), t, t0, tf, x, u, v) == :mixed
        @test constraint_type(:(v[1:2:10]), t, t0, tf, x, u, v) == (:variable_range, 1:2:9)
        @test constraint_type(:(v[1:10]), t, t0, tf, x, u, v) == (:variable_range, 1:10)
        @test constraint_type(:(v[2]), t, t0, tf, x, u, v) == (:variable_range, 2)
        @test constraint_type(:(v), t, t0, tf, x, u, v) == (:variable_range, nothing)
        @test constraint_type(:(v^2 + 1), t, t0, tf, x, u, v) == :variable_fun
        @test constraint_type(:(v[2]^2 + 1), t, t0, tf, x, u, v) == :variable_fun
    end
end
