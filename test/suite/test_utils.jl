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

        # ===== EXISTING FUNCTIONALITY (scalar indexing) =====

        @testset "scalar indexing (existing)" begin
            # Test 1: Basic scalar substitution
            e = :(x0[1] * 2xf[3] - cos(xf[2]) * 2x0[2])
            @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
                :(x[1, 0] * (2 * x[3, N]) - cos(x[2, N]) * (2 * x[2, 0]))

            # Test 2: Bare symbols are NOT substituted
            e = :(x0 * 2xf[3] - cos(xf) * 2x0[2])
            @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
                :(x0 * (2 * x[3, N]) - cos(xf) * (2 * x[2, 0]))

            # Test 3: Numeric index
            e = :(x0[5] + x0[10])
            @test subs2(e, :x0, :x, 0) == :(x[5, 0] + x[10, 0])

            # Test 4: Symbolic index
            e = :(x0[i] + x0[j])
            @test subs2(e, :x0, :x, 0) == :(x[i, 0] + x[j, 0])
        end

        # ===== NEW FUNCTIONALITY (range indexing) =====

        @testset "range indexing (new)" begin
            # Test 5: Simple range 1:3
            e = :(x0[1:3])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :([x[k, 0] for k in 1:3])

            # Test 6: Range with step 1:2:5
            e = :(x0[1:2:5])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :([x[k, 0] for k in 1:2:5])

            # Test 7: Range with symbolic bounds
            e = :(x0[1:n])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :([x[k, 0] for k in 1:n])

            # Test 8: Multiple ranges in same expression
            e = :(x0[1:3] + xf[2:4])
            result = subs2(subs2(e, :x0, :x, 0; k=:k1), :xf, :x, :N; k=:k2)
            @test result == :([x[k1, 0] for k1 in 1:3] + [x[k2, N] for k2 in 2:4])

            # Test 9: Range inside function call
            e = :(sum(x0[1:n]))
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :(sum([x[k, 0] for k in 1:n]))
        end

        @testset "mixed scalar and range" begin
            # Test 10: Expression with both scalars and ranges
            e = :(x0[1] + x0[2:4] + x0[5])
            result = subs2(e, :x0, :x, 0; k=:k)
            # x0[1] → x[1, 0]
            # x0[2:4] → [x[k, 0] for k ∈ 2:4]
            # x0[5] → x[5, 0]
            @test result == :(x[1, 0] + [x[k, 0] for k in 2:4] + x[5, 0])
        end

        @testset "nested and complex expressions" begin
            # Test 11: Nested function calls with ranges
            e = :(norm(x0[1:3]) + cos(x0[4]))
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :(norm([x[k, 0] for k in 1:3]) + cos(x[4, 0]))

            # Test 12: Range in matrix operations
            e = :(A * x0[1:n])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :(A * [x[k, 0] for k in 1:n])

            # Test 13: Multiple substitutions with symbolic j
            e = :(x0[1:3] + xf[2:4])
            result = subs2(subs2(e, :x0, :x, :j; k=:k1), :xf, :x, :(j+1); k=:k2)
            @test result == :([x[k1, j] for k1 in 1:3] + [x[k2, j + 1] for k2 in 2:4])
        end

        @testset "edge cases" begin
            # Test 14: Single-element range (should still create comprehension)
            e = :(x0[1:1])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :([x[k, 0] for k in 1:1])

            # Test 15: Wrong variable name (should not substitute)
            e = :(y0[1:3])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == e  # Unchanged

            # Test 16: Complex symbolic j expression
            e = :(x0[1:3])
            result = subs2(e, :x0, :x, :grid_size; k=:k)
            @test result == :([x[k, grid_size] for k in 1:3])

            # Test 17: Scalar index that is a range expression (should not match)
            # This tests that we properly distinguish i (scalar) from 1:3 (range)
            e = :(x0[i])
            result = subs2(e, :x0, :x, 0; k=:k)
            @test result == :(x[i, 0])  # Scalar behavior
        end

        @testset "backward compatibility" begin
            # Test 18: Scalar indexing still works
            e = :(x0[1] * 2xf[3] - cos(xf[2]) * 2x0[2])
            @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
                :(x[1, 0] * (2 * x[3, N]) - cos(x[2, N]) * (2 * x[2, 0]))

            # Test 19: Bare symbols are NOT substituted
            e = :(x0 * 2xf[3] - cos(xf) * 2x0[2])
            @test subs2(subs2(e, :x0, :x, 0), :xf, :x, :N) ==
                :(x0 * (2 * x[3, N]) - cos(xf) * (2 * x[2, 0]))
        end
    end

    @testset "subs3" begin
        println("subs3")

        e = :(x0[1:2:d] * 2xf[1:3])
        @test subs3(e, :x0, :x, :i, 0) == :(x[i, 0] * (2 * xf[1:3]))
        @test subs3(e, :xf, :x, 1, :N) == :(x0[1:2:d] * (2 * x[1, N]))
    end

    @testset "subs2m" begin
        println("subs2m")

        @testset "range indexing" begin
            # Test 1: Basic range substitution
            e = :(x0[1:3])
            result = subs2m(e, :x0, :x, 0; k=:k)
            @test result == :([((x[k, 0] + x[k, 0 + 1]) / 2) for k in 1:3])

            # Test 2: Range with step
            e = :(x0[1:2:5])
            result = subs2m(e, :x0, :x, 0; k=:k)
            @test result == :([((x[k, 0] + x[k, 0 + 1]) / 2) for k in 1:2:5])

            # Test 3: Range in arithmetic expression
            e = :(2 * x0[1:3])
            result = subs2m(e, :x0, :x, 0; k=:k)
            @test result == :(2 * [((x[k, 0] + x[k, 0 + 1]) / 2) for k in 1:3])

            # Test 4: Multiple ranges in same expression
            e = :(x0[1:2] + xf[2:4])
            result = subs2m(subs2m(e, :x0, :x, 0; k=:k), :xf, :x, :N; k=:k)
            @test result == :(
                [((x[k, 0] + x[k, 0 + 1]) / 2) for k in 1:2] + [((x[k, N] + x[k, N + 1]) / 2) for k in 2:4]
            )

            # Test 5: Range with symbolic j
            e = :(x0[1:3])
            result = subs2m(e, :x0, :x, :j; k=:k)
            @test result == :([((x[k, j] + x[k, j + 1]) / 2) for k in 1:3])

            # Test 6: Single-element range
            e = :(x0[2:2])
            result = subs2m(e, :x0, :x, 0; k=:k)
            @test result == :([((x[k, 0] + x[k, 0 + 1]) / 2) for k in 2:2])
        end

        @testset "backward compatibility" begin
            # Test 7: Scalar indexing still works
            e = :(x0[1] * 2xf[3] - cos(xf[2]) * 2x0[2])
            @test subs2m(subs2m(e, :x0, :x, 0), :xf, :x, :N) == :(
                ((x[1, 0] + x[1, 0 + 1]) / 2) * (2 * ((x[3, N] + x[3, N + 1]) / 2)) -
                cos((x[2, N] + x[2, N + 1]) / 2) * (2 * ((x[2, 0] + x[2, 0 + 1]) / 2))
            )

            # Test 8: Bare symbols are NOT substituted
            e = :(x0 * 2xf[3] - cos(xf) * 2x0[2])
            @test subs2m(subs2m(e, :x0, :x, 0), :xf, :x, :N) == :(
                x0 * (2 * ((x[3, N] + x[3, N + 1]) / 2)) -
                cos(xf) * (2 * ((x[2, 0] + x[2, 0 + 1]) / 2))
            )
        end
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
