# test onepass, functional parsing (aka CTModels primitives)
# todo:
 
function to_out_of_place(f!, n; T=Float64)
    function f(args...; kwargs...)
        r = zeros(T, n)
        f!(r, args...; kwargs...)
        #return n == 1 ? r[1] : r
        return r # everything is now a vector
    end
    return isnothing(f!) ? nothing : f
end

function __constraint(ocp, label)
    type_, fun_, lb_, ub_ = constraint(ocp,label)
    function fun(args...)
        r = fun_(args...)
        if length(r) == 1 # f returns a scalar if length(r) == 1
            return [r[1]] # make it a vector
        else
            return r
        end
    end
    lb = length(lb_) == 1 ? [lb_[1]] : lb_
    ub = length(ub_) == 1 ? [ub_[1]] : ub_
    return (type_, fun, lb, ub)
end

__dynamics(ocp) = to_out_of_place(dynamics(ocp), state_dimension(ocp))

function test_onepass_fun()

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "@def o syntax" begin
        println("@def o syntax testset...")

        oo = @def begin
            λ ∈ R^2, variable
            tf = λ₂
            t ∈ [0, tf], time
            x ∈ R, state
            u ∈ R, control
            ẋ(t) == u(t)
            tf → min 
        end
        @test initial_time(oo) == 0
        @test final_time(oo, [0, 1]) == 1

        @def o begin
            t ∈ [0, 1], time
            x = [r, v] ∈ R², state
            u ∈ R, control
            w = r + 2v
            r(0) == 0, (1)
            v(0) == 1, (♡)
            x(t) + [u(t), 1] <= [1, 2], (2)
            x(t) <= [0, 0], (3)
            r(t) <= 0, (4)
            u(t) <= 0, (5)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __constraint(o, :eq2)[2](0, x, u, nothing) == x + [u[1], 1]

        @test __constraint(o, :eq3)[2](0, x, u, nothing) == x
        @test __constraint(o, :eq4)[2](0, x, u, nothing) == x[1:1]
        @test __constraint(o, :eq5)[2](0, x, u, nothing) == u 

        @test __dynamics(o)(0, x, u, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]
        @test criterion(o) == :min
    
        @def oo begin
            λ ∈ R^2, variable
            tf = λ₂
            t ∈ [0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(oo) == 0
        λ = [1, 2]
        @test final_time(oo, λ) == λ[2]

        a = 1
        f(b) = begin # closure of a, local c, and @def in function
            c = 3
            ocp = @def begin
                t ∈ [a, b], time
                x ∈ R, state
                u ∈ R, control
                ẋ(t) == x(t) + u(t) + b + c + d
                0 => min # generic (untested)
            end
            return ocp
        end
        b = 2
        o = f(b)
        d = 4
        x = [10]
        u = [20]
        @test __dynamics(o)(0, x, u, nothing) == [x[1] + u[1] + b + 3 + d]
    end

    @testset "log" begin
        println("log testset...")

        @def o begin
            λ ∈ R^2, variable
            tf = λ₂
            t ∈ [0, tf], time
            x ∈ R, state # generic (untested)
            u ∈ R, control # generic (untested)
            ẋ(t) == u(t) # generic (untested)
            0 => min # generic (untested)
        end true
        @test initial_time(o) == 0
        @test final_time(o, [0, 2]) == 2
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "aliases" begin
        println("aliases testset...")

        @def o begin
            t in [0, 1], time
            x = (y, z) in R², state
            u = (uu1, uu2, uu3) in R³, control
            v = (vv1, vv2) in R², variable
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test state_components(o) == ["y", "z"]
        @test control_components(o) == ["uu1", "uu2", "uu3"]
        @test variable_components(o) == ["vv1", "vv2"]

        @def o begin
            t in [0, 1], time
            x = (y, z) ∈ R², state
            u = (uu1, uu2, uu3) ∈ R³, control
            v = (vv1, vv2) ∈ R², variable
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test state_components(o) == ["y", "z"]
        @test control_components(o) == ["uu1", "uu2", "uu3"]
        @test variable_components(o) == ["vv1", "vv2"]

        @def o begin
            t in [0, 1], time
            x = [y, z] ∈ R², state
            u = [uu1, uu2, uu3] ∈ R³, control
            v = [vv1, vv2] ∈ R², variable
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test state_components(o) == ["y", "z"]
        @test control_components(o) == ["uu1", "uu2", "uu3"]
        @test variable_components(o) == ["vv1", "vv2"]

        @test_throws ParsingError @def o begin # a name must be provided
            (y, z) ∈ R², state
        end

        @test_throws ParsingError @def o begin # a name must be provided
            (uu1, uu2, uu3) ∈ R³, control
        end

        @test_throws ParsingError @def o begin # a name must be provided
            (vv1, vv2) ∈ R², variable
        end

        @test_throws ParsingError @def o begin # a name must be provided
            [y, z] ∈ R², state
        end

        @test_throws ParsingError @def o begin # a name must be provided
            [uu1, uu2, uu3] ∈ R³, control
        end

        @test_throws ParsingError @def o begin # a name must be provided
            [vv1, vv2] ∈ R², variable
        end

        @def o begin
            t ∈ [0, 1], time
            x = (r, v) ∈ R², state
            u ∈ R, control
            w = r + 2v
            r(0) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, u, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]

        @def o begin
            t ∈ [0, 1], time
            x = [r, v] ∈ R², state
            u ∈ R, control
            w = r + 2v
            r(0) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, u, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]

        @def o begin
            t ∈ [0, 1], time
            x = [r, v] ∈ R², state
            c = [u, b] ∈ R², control
            w = r + 2v
            b(t) == 0
            r(0) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + b(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        c = [u[1], 0]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, c, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, c, nothing) == u[1]^2 + x[1]

        @def o begin
            t ∈ [0, 1], time
            x ∈ R^3, state
            u = (u₁, v) ∈ R^2, control
            ẋ(t) == [x[1](t) + 2v(t), 2x[3](t), x[1](t) + v(t)]
            0 => min # generic (untested)
        end
        @test state_dimension(o) == 3
        @test control_dimension(o) == 2
        x = [1, 2, 3]
        u = [-1, 2]
        @test __dynamics(o)(0, x, u, nothing) == [x[1] + 2u[2], 2x[3], x[1] + u[2]]

        t0 = 0.0
        tf = 0.1
        @def ocp begin
            t ∈ [t0, tf], time
            x ∈ R^3, state
            u ∈ R^3, control
            r = x[1]
            v = x₂
            a = x₃
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_name(ocp) == "u"
        @test control_dimension(ocp) == 3
        @test state_name(ocp) == "x"
        @test state_dimension(ocp) == 3
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "variable" begin
        println("variable testset...")

        @def o begin
            λ ∈ R^2, variable
            tf = λ₂
            t ∈ [0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)u in R, control # generic (untested)
        end
        @test initial_time(o) == 0
        @test final_time(o, [1, 2]) == 2

        @def o begin
            λ = (λ₁, tf) ∈ R^2, variable
            t ∈ [0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o) == 0
        @test final_time(o, [1, 2]) == 2

        @def o begin
            t0 ∈ R, variable
            t ∈ [t0, 1], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o, [1]) == 1
        @test final_time(o) == 1

        @def o begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o) == 0
        @test final_time(o, [1]) == 1

        @def o begin
            v ∈ R², variable
            s ∈ [v[1], v[2]], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(s) == x(s) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o, [1, 2]) == 1
        @test final_time(o, [1, 2]) == 2

        @def o begin
            v ∈ R², variable
            s0 = v₁
            sf = v₂
            s ∈ [s0, sf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(s) == x(s) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o, [1, 2]) == 1
        @test final_time(o, [1, 2]) == 2

        @test_throws ParsingError @def o begin
            t0 ∈ R², variable
            t ∈ [t0, 1], time
        end
 
        @test_throws ParsingError @def o begin
            tf ∈ R², variable
            t ∈ [0, tf], time
        end
 
        @test_throws ParsingError @def o begin
            v, variable
            t ∈ [0, tf[v]], time
        end

        @test_throws ParsingError @def o begin
            v, variable
            t ∈ [t0[v], 1], time
        end

        @test_throws ParsingError @def o begin
            v, variable
            t ∈ [t0[v], tf[v + 1]], time
        end

        t0 = 0.0
        tf = 0.1
        @def ocp begin
            t ∈ [t0, tf], time
            a ∈ R, variable
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test variable_dimension(ocp) == 1
        @test variable_name(ocp) == "a"

        t0 = 0.0
        tf = 0.1
        @def ocp begin
            t ∈ [t0, tf], time
            a ∈ R³, variable
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test variable_dimension(ocp) == 3
        @test variable_name(ocp) == "a"
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "time" begin
        println("time testset...")

        t0 = 0
        @def o begin
            t ∈ [t0, t0 + 4], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test initial_time(o) == t0
        @test final_time(o) == t0 + 4

        @test_throws ParsingError @def o t ∈ 1

        @def ocp begin
            t ∈ [0.0, 1.0], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == 0.0
        @test final_time(ocp) == 1.0

        t0 = 3.0
        @def ocp begin
            tf ∈ R, variable
            t ∈ [t0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp, [1]) == 1

        tf = 3.14
        @def ocp begin
            t0 ∈ R, variable
            t ∈ [t0, tf], time
            x in R, state # generic (untested)
            u in R, control # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp, [1]) == 1
        @test final_time(ocp) == tf
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "state / control" begin
        println("state / control testset...")

        @def o begin
            t in [0, 1], time
            x ∈ R, state
            u ∈ R, control
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test state_dimension(o) == 1
        @test control_dimension(o) == 1

        # state
        t0 = 1.0
        tf = 1.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R, state
            v in R, control # generic (untested)
            derivative(u)(t) == u(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 1
        @test state_name(ocp) == "u"

        t0 = 2.0
        tf = 2.1
        @def ocp begin
            t ∈ [t0, tf], time
            v ∈ R^4, state
            u in R, control # generic (untested)
            derivative(v)(t) == v(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 4
        @test state_name(ocp) == "v"

        t0 = 3.0
        tf = 3.1
        @def ocp begin
            t ∈ [t0, tf], time
            w ∈ R^3, state
            u in R, control # generic (untested)
            derivative(w)(t) == w(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 3
        @test state_name(ocp) == "w"

        t0 = 4.0
        tf = 4.1
        @def ocp begin
            t ∈ [t0, tf], time
            a ∈ R, state
            u in R, control # generic (untested)
            derivative(a)(t) == a(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 1
        @test state_name(ocp) == "a"

        t0 = 5.0
        tf = 5.1
        @def ocp begin
            t ∈ [t0, tf], time
            b ∈ R¹, state
            u in R, control # generic (untested)
            derivative(b)(t) == b(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 1
        @test state_name(ocp) == "b"

        t0 = 6.0
        tf = 6.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R⁹, state
            v in R, control # generic (untested)
            derivative(u)(t) == x(u) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == 9
        @test state_name(ocp) == "u"

        n = 3
        t0 = 7.0
        tf = 7.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R^n, state
            v in R, control # generic (untested)
            derivative(u)(t) == u(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test state_dimension(ocp) == n
        @test state_name(ocp) == "u"

        # control
        t0 = 1.0
        tf = 1.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 1
        @test control_name(ocp) == "u"

        t0 = 2.0
        tf = 2.1
        @def ocp begin
            t ∈ [t0, tf], time
            v ∈ R^4, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 4
        @test control_name(ocp) == "v"

        t0 = 3.0
        tf = 3.1
        @def ocp begin
            t ∈ [t0, tf], time
            w ∈ R^3, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 3
        @test control_name(ocp) == "w"

        t0 = 4.0
        tf = 4.1
        @def ocp begin
            t ∈ [t0, tf], time
            a ∈ R, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 1
        @test control_name(ocp) == "a"

        t0 = 5.0
        tf = 5.1
        @def ocp begin
            t ∈ [t0, tf], time
            b ∈ R¹, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 1
        @test control_name(ocp) == "b"

        t0 = 6.0
        tf = 6.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R⁹, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == 9
        @test control_name(ocp) == "u"

        n = 3
        t0 = 7.0
        tf = 7.1
        @def ocp begin
            t ∈ [t0, tf], time
            u ∈ R^n, control
            x in R, state # generic (untested)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_dimension(ocp) == n
        @test control_name(ocp) == "u"
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "dynamics" begin
        println("dynamics testset...")

        @def o begin
            t ∈ [0, 1], time
            x ∈ R^3, state
            u ∈ R^2, control
            ẋ(t) == [x[1](t) + 2u[2](t), 2x[3](t), x[1](t) + u[2](t)]
            0 => min # generic (untested)
        end
        @test state_dimension(o) == 3
        @test control_dimension(o) == 2
        x = [1, 2, 3]
        u = [-1, 2]
        @test __dynamics(o)(0, x, u, nothing) == [x[1] + 2u[2], 2x[3], x[1] + u[2]]
        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁
            ẏ(s) == [aa(s), r(s)^2 + w(s) + z₁, 0, 0]
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        w = [9]
        @test __dynamics(o)(0, y, w, z) == [y[1], y[3]^2 + w[1] + z[1], 0, 0]

        @def o begin
            z ∈ R², variable
            __s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(__s)
            ẏ(__s) == [aa(__s), r²(__s) + w(__s) + z₁, 0, 0]
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        w = [9]
        @test_throws MethodError __dynamics(o)(0, y, w, z)

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(s) + v^3 + z₂
            ẏ(s) == [aa(s) + w(s)^2, r(s)^2, 0, 0]
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        ww = [19]
        @test __dynamics(o)(0, y, ww, z) == [y[1] + ww[1]^2 + y[4]^3 + z[2], y[3]^2, 0, 0]

        @def o begin
            z ∈ R², variable
            __t ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(0) + v^3 + z₂
            ẏ(__t) == [aa(__t) + (w^2)(__t), r(__t)^2, 0, 0]
            aa(0) + y₂(z₁) → min
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        w = [11]
        @test_throws MethodError __dynamics(o)(0, y, w, z)
        @test mayer(o)(y0, yf, z) == y0[1] + y0[4]^3 + z[2] + yf[2]
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "dynamics_coords" begin
        println("dynamics_coords testset...")

        @def o begin
            t ∈ [0, 1], time
            x ∈ R^3, state
            u ∈ R^2, control
            ∂(x₁)(t) == x[1](t) + 2u[2](t) 
            ∂(x₂)(t) == 2x[3](t)
            ∂(x₃)(t) == x[1](t) + u[2](t) 
            0 => min # generic (untested)
        end
        @test state_dimension(o) == 3
        @test control_dimension(o) == 2
        x = [1, 2, 3]
        u = [-1, 2]
        @test __dynamics(o)(0, x, u, nothing) == [x[1] + 2u[2], 2x[3], x[1] + u[2]]

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁
            ∂(y[1])(s) == aa(s)
            ∂(y[2])(s) == r(s)^2 + w(s) + z₁
            ∂(y[3])(s) == 0
            ∂(y[4])(s) == 0
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        w = [9]
        @test __dynamics(o)(0, y, w, z) == [y[1], y[3]^2 + w[1] + z[1], 0, 0]

        @def o begin
            z ∈ R², variable
            __s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(__s)
            ∂(y[1])(__s) == aa(__s)
            ∂(y[2])(__s) == r²(__s) + w(__s) + z₁
            ∂(y[3])(__s) == 0
            ∂(y[4])(__s) == 0
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        w = [9]
        @test_throws MethodError __dynamics(o)(0, y, w, z)

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(s) + v^3 + z₂
            ∂(y[1])(s) == aa(s) + w(s)^2
            ∂(y[2])(s) == r(s)^2
            ∂(y[3])(s) == 0
            ∂(y[4])(s) == 0
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        ww = [19]
        @test __dynamics(o)(0, y, ww, z) == [y[1] + ww[1]^2 + y[4]^3 + z[2], y[3]^2, 0, 0]

        @def o begin
            z ∈ R², variable
            __t ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(0) + v^3 + z₂
            ∂(y[1])(__t) == aa(__t) + (w^2)(__t)
            ∂(y[2])(__t) == r(__t)^2
            ∂(y[3])(__t) == 0
            ∂(y[4])(__t) == 0
            aa(0) + y₂(z₁) → min
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        w = [11]
        @test_throws MethodError __dynamics(o)(0, y, w, z)
        @test mayer(o)(y0, yf, z) == y0[1] + y0[4]^3 + z[2] + yf[2]

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁹, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁
            ∂(y[1])(s) == aa(s)
            ∂(y[2:3])(s) == [r(s)^2 + w(s) + z₁, 0]
            ∂(y[4:6])(s) == [y₁(s), y₂(s), 3]
            ∂(y[7:7])(s) == [4]
            ∂(y[8:8])(s) == 5
            ∂(y[9])(s) == [6]
            0 => min # generic (untested)
        end
        z = [5, 6]
        y = [1, 2, 3, 4, 5, 6, 7, 8]
        w = [9]
        @test __dynamics(o)(0, y, w, z) == [y[1], y[3]^2 + w[1] + z[1], 0, y[1], y[2], 3, 4, 5, 6]

    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "constraints" begin
        println("constraints testset...")

        @def o begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            w = r + 2v^3
            r(0) + w(tf) - tf^2 == 0, (1)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        tf = [2]
        x0 = [1, 2]
        xf = [3, 4]
        @test __constraint(o, :eq1)[2](x0, xf, tf) == [x0[1] + (xf[1] + 2xf[2]^3) - tf[1]^2]

        n = 11
        m = 6
        @def o begin
            t ∈ [0, 1], time
            x ∈ R^n, state
            u ∈ R^m, control
            r = x₁
            v = x₂
            0 ≤ r(t) ≤ 1, (1)
            zeros(n) ≤ x(t) ≤ ones(n), (2)
            [0, 0] ≤ x[1:2](t) ≤ [1, 1], (3)
            [0, 0] ≤ x[1:2:4](t) ≤ [1, 1], (4)
            0 ≤ v(t)^2 ≤ 1, (5)
            zeros(m) ≤ u(t) ≤ ones(m), (6)
            [0, 0] ≤ u[1:2](t) ≤ [1, 1], (7)
            [0, 0] ≤ u[1:2:4](t) ≤ [1, 1], (8)
            0 ≤ u₂(t)^2 ≤ 1, (9)
            u₁(t) * x[1:2](t) == [1, 1], (10)
            [0, 0] ≤ u₁(t) * x[1:2](t) .^ 3 ≤ [1, 1], (11)
            derivative(x)(t) == x(t)
            0 => min # generic (untested)
        end
        x = Vector{Float64}(1:n)
        u = 2 * Vector{Float64}(1:m)
        @test __constraint(o, :eq1)[2](0, x, u, nothing) == x[1:1]
        @test __constraint(o, :eq2)[2](0, x, u, nothing) == x
        @test __constraint(o, :eq3)[2](0, x, u, nothing) == x[1:2]
        @test __constraint(o, :eq4)[2](0, x, u, nothing) == x[1:2:4]
        @test __constraint(o, :eq5)[2](0, x, u, nothing) == [x[2]^2]
        @test __constraint(o, :eq6)[2](0, x, u, nothing) == u
        @test __constraint(o, :eq7)[2](0, x, u, nothing) == u[1:2]
        @test __constraint(o, :eq8)[2](0, x, u, nothing) == u[1:2:4]
        @test __constraint(o, :eq9)[2](0, x, u, nothing) == [u[2]^2]
        @test __constraint(o, :eq10)[2](0, x, u, nothing) == u[1] * x[1:2]
        @test __constraint(o, :eq11)[2](0, x, u, nothing) == u[1] * x[1:2] .^ 3

        n = 11
        m = 6
        @def o begin
            z ∈ R^2, variable
            t ∈ [0, 1], time
            x ∈ R^n, state
            u ∈ R^m, control
            r = x₁
            v = x₂
            0 ≤ r(t) ≤ 1, (1)
            zeros(n) ≤ x(t) ≤ ones(n), (2)
            [0, 0] ≤ x[1:2](t) - [z₁, 1] ≤ [1, 1], (3)
            [0, 0] ≤ x[1:2:4](t) ≤ [1, 1], (4)
            0 ≤ v(t)^2 ≤ 1, (5)
            zeros(m) ≤ u(t) ≤ ones(m), (6)
            [0, 0] ≤ u[1:2](t) ≤ [1, 1], (7)
            [0, 0] ≤ u[1:2:4](t) ≤ [1, 1], (8)
            0 ≤ u₂(t)^2 ≤ 1, (9)
            u₁(t) * x[1:2](t) + z + f() == [1, 1], (10)
            [0, 0] ≤ u₁(t) * x[1:2](t) .^ 3 + z ≤ [1, 1], (11)
            derivative(x)(t) == x(t)
            0 => min # generic (untested)
        end
        f() = [1, 1]
        z = 3 * Vector{Float64}(1:2)
        x = Vector{Float64}(1:n)
        u = 2 * Vector{Float64}(1:m)
        @test __constraint(o, :eq1)[2](0, x, u, z) == x[1:1]
        @test __constraint(o, :eq2)[2](0, x, u, z) == x
        @test __constraint(o, :eq3)[2](0, x, u, z) == x[1:2] - [z[1], 1]
        @test __constraint(o, :eq4)[2](0, x, u, z) == x[1:2:4]
        @test __constraint(o, :eq5)[2](0, x, u, z) == [x[2]^2]
        @test __constraint(o, :eq6)[2](0, x, u, z) == u
        @test __constraint(o, :eq7)[2](0, x, u, z) == u[1:2]
        @test __constraint(o, :eq8)[2](0, x, u, z) == u[1:2:4]
        @test __constraint(o, :eq9)[2](0, x, u, z) == [u[2]^2]
        @test __constraint(o, :eq10)[2](0, x, u, z) == u[1] * x[1:2] + z + f()
        @test __constraint(o, :eq11)[2](0, x, u, z) == u[1] * x[1:2] .^ 3 + z

        @def o begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            begin
                r = x₁
                v = x₂
                w = r + 2v
                r(0) == 0, (1)
            end
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, u, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]

        @def o begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            w = r + 2v
            r(0) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, u, nothing) == [x[2], (x[1] + 2x[2])^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]

        @def o begin
            z ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            w = r + 2v
            r(0) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), w(t)^2 + z₁]
            ∫(u(t)^2 + z₂ * x₁(t)) → min
        end
        x = [1, 2]
        x0 = 2 * [1, 2]
        xf = 3 * [1, 2]
        u = [3]
        z = [4, 5]
        @test __constraint(o, :eq1)[2](x0, xf, z) == x0[1:1]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0[2:2]
        @test __dynamics(o)(0, x, u, z) == [x[2], (x[1] + 2x[2])^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[2] * x[1]

        @def o begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            r(0)^2 + v(1) == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), r(t)^2]
            ∫(u(t)^2 + x₁(t)) → min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == [x0[1]^2 + xf[2]]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, nothing) == x0[2:2]
        @test __dynamics(o)(0, x, u, nothing) == [x[2], x[1]^2]
        @test lagrange(o)(0, x, u, nothing) == u[1]^2 + x[1]

        @def o begin
            z ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            r(0) - z == 0, (1)
            v(0) == 1, (♡)
            ẋ(t) == [v(t), r(t)^2 + z]
            ∫(u(t)^2 + z * x₁(t)) → min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        z = [4]
        @test __constraint(o, :eq1)[2](x0, xf, z) == [x0[1] - z[1]]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0[2:2]
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]

        @def o begin
            z ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            0 ≤ r(0) - z ≤ 1, (1)
            0 ≤ v(1)^2 ≤ 1, (2)
            [0, 0] ≤ x(0) ≤ [1, 1], (♡)
            ẋ(t) == [v(t), r(t)^2 + z]
            ∫(u(t)^2 + z * x₁(t)) → min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        z = [4]
        @test __constraint(o, :eq1)[2](x0, xf, z) == [x0[1] - z[1]]
        @test __constraint(o, :eq2)[2](x0, xf, z) == [xf[2]^2]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]

        @def o begin
            z ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            r = x₁
            v = x₂
            1 ≥ r(0) - z ≥ 0, (1)
            1 ≥ v(1)^2 ≥ 0, (2)
            [1, 1] ≥ x(0) ≥ [0, 0], (3)
            ẋ(t) == [v(t), r(t)^2 + z]
            ∫(u(t)^2 + z * x₁(t)) → min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        z = [4]
        @test __constraint(o, :eq1)[2](x0, xf, z) == [x0[1] - z[1]]
        @test __constraint(o, :eq2)[2](x0, xf, z) == [xf[2]^2]
        @test __constraint(o, :eq3)[2](x0, xf, z) == x0
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]
        @test __constraint(o, :eq1)[3] == [0]
        @test __constraint(o, :eq1)[4] == [1]
        @test __constraint(o, :eq2)[3] == [0]
        @test __constraint(o, :eq2)[4] == [1]
        @test __constraint(o, :eq3)[3] == [0, 0]
        @test __constraint(o, :eq3)[4] == [1, 1]

        @def o begin
            v ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            x(0) - v₁ == 0, (1)
            x(1) - v₁ == 0, (2)
            0 ≤ x(0) - v₁ ≤ 1, (3)
            0 ≤ x(1) - v₁ ≤ 1, (4)
            x(0) + x(1) - v₂ == 0, (5)
            0 ≤ x(0) + x(1) - v₂ ≤ 1, (6)
            x(t) - v₁ == 0, (7)
            u(t) - v₁ == 0, (8)
            z = v₁ + 2v₂
            0 ≤ x(t) - z ≤ 1, (9)
            0 ≤ u(t) - z ≤ 1, (10)
            0 ≤ x(t) + u(t) - z ≤ 1, (11)
            ẋ(t) == z * x(t) + 2u(t)
            v₁ == 1, (12)
            0 ≤ v₁ ≤ 1, (13)
            z == 1, (14)
            0 ≤ z ≤ 1, (15)
            z * x(1) → min
        end
        x = [1]
        x0 = [2]
        xf = [3]
        u = [4]
        v = [5, 6]
        z = v[1] + 2v[2]
        @test __constraint(o, :eq1)[2](x0, xf, v) == x0 - v[1:1]
        @test __constraint(o, :eq2)[2](x0, xf, v) == xf - v[1:1]
        @test __constraint(o, :eq3)[2](x0, xf, v) == x0 - v[1:1]
        @test __constraint(o, :eq4)[2](x0, xf, v) == xf - v[1:1]
        @test __constraint(o, :eq5)[2](x0, xf, v) == x0 + xf - v[2:2]
        @test __constraint(o, :eq6)[2](x0, xf, v) == x0 + xf - v[2:2]
        @test __constraint(o, :eq7)[2](0, x, u, v) == x - v[1:1]
        @test __constraint(o, :eq9)[2](0, x, u, v) == [x[1] - z]
        @test __constraint(o, :eq10)[2](0, x, u, v) == [u[1] - z]
        @test __constraint(o, :eq11)[2](0, x, u, v) == [x[1] + u[1] - z]
        @test __constraint(o, :eq12)[2](x0, xf, v) == v[1:1]
        @test __constraint(o, :eq13)[2](x0, xf, v) == v[1:1]
        @test __constraint(o, :eq14)[2](x0, xf, v) == v[1:1] + 2v[2:2]
        @test __constraint(o, :eq15)[2](x0, xf, v) == v[1:1] + 2v[2:2]

        @def o begin
            v ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            x(0) ≤ 0
            x(0) ≤ 0, (1)
            x(1) ≤ 0
            x(1) ≤ 0, (2)
            x³(0) ≤ 0
            x³(0) ≤ 0, (3)
            x³(1) ≤ 0
            x³(1) ≤ 0, (4)
            x(t) ≤ 0
            x(t) ≤ 0, (5)
            x(t) ≤ 0
            x(t) ≤ 0, (6)
            u₁(t) ≤ 0
            u₁(t) ≤ 0, (7)
            u₁(t) ≤ 0
            u₁(t) ≤ 0, (8)
            x³(t) ≤ 0
            x³(t) ≤ 0, (9)
            x³(t) ≤ 0
            x³(t) ≤ 0, (10)
            (u₁^3)(t) ≤ 0
            (u₁^3)(t) ≤ 0, (11)
            (u₁^3)(t) ≤ 0
            (u₁^3)(t) ≤ 0, (12)
            x(t) + (u₁^3)(t) ≤ 0
            x(t) + (u₁^3)(t) ≤ 0, (13)
            x(t) + (u₁^3)(t) ≤ 0
            x(t) + (u₁^3)(t) ≤ 0, (14)
            v ≤ 0
            v ≤ 0, (15)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end

        @test __constraint(o, :eq1)[3] == [-Inf]
        @test __constraint(o, :eq2)[3] == [-Inf]
        @test __constraint(o, :eq3)[3] == [-Inf]
        @test __constraint(o, :eq4)[3] == [-Inf]
        @test __constraint(o, :eq5)[3] == [-Inf]
        @test __constraint(o, :eq6)[3] == [-Inf]
        @test __constraint(o, :eq7)[3] == [-Inf]
        @test __constraint(o, :eq8)[3] == [-Inf]
        @test __constraint(o, :eq9)[3] == [-Inf]
        @test __constraint(o, :eq10)[3] == [-Inf]
        @test __constraint(o, :eq11)[3] == [-Inf]
        @test __constraint(o, :eq12)[3] == [-Inf]
        @test __constraint(o, :eq13)[3] == [-Inf]
        @test __constraint(o, :eq14)[3] == [-Inf]
        @test __constraint(o, :eq15)[3] == [-Inf]
        @test __constraint(o, :eq1)[4] == [0]
        @test __constraint(o, :eq2)[4] == [0]
        @test __constraint(o, :eq3)[4] == [0]
        @test __constraint(o, :eq4)[4] == [0]
        @test __constraint(o, :eq5)[4] == [0]
        @test __constraint(o, :eq6)[4] == [0]
        @test __constraint(o, :eq7)[4] == [0]
        @test __constraint(o, :eq8)[4] == [0]
        @test __constraint(o, :eq9)[4] == [0]
        @test __constraint(o, :eq10)[4] == [0]
        @test __constraint(o, :eq11)[4] == [0]
        @test __constraint(o, :eq12)[4] == [0]
        @test __constraint(o, :eq13)[4] == [0]
        @test __constraint(o, :eq14)[4] == [0]
        @test __constraint(o, :eq15)[4] == [0]

        @def o begin
            v ∈ R, variable
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            x(0) ≥ 0
            x(0) ≥ 0, (1)
            x(1) ≥ 0
            x(1) ≥ 0, (2)
            x³(0) ≥ 0
            x³(0) ≥ 0, (3)
            x³(1) ≥ 0
            x³(1) ≥ 0, (4)
            x(t) ≥ 0
            x(t) ≥ 0, (5)
            x(t) ≥ 0
            x(t) ≥ 0, (6)
            u₁(t) ≥ 0
            u₁(t) ≥ 0, (7)
            u₁(t) ≥ 0
            u₁(t) ≥ 0, (8)
            x³(t) ≥ 0
            x³(t) ≥ 0, (9)
            x³(t) ≥ 0
            x³(t) ≥ 0, (10)
            (u₁^3)(t) ≥ 0
            (u₁^3)(t) ≥ 0, (11)
            (u₁^3)(t) ≥ 0
            (u₁^3)(t) ≥ 0, (12)
            x(t) + (u₁^3)(t) ≥ 0
            x(t) + (u₁^3)(t) ≥ 0, (13)
            x(t) + (u₁^3)(t) ≥ 0
            x(t) + (u₁^3)(t) ≥ 0, (14)
            v ≥ 0
            v ≥ 0, (15)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end

        @test __constraint(o, :eq1)[3] == [0]
        @test __constraint(o, :eq2)[3] == [0]
        @test __constraint(o, :eq3)[3] == [0]
        @test __constraint(o, :eq4)[3] == [0]
        @test __constraint(o, :eq5)[3] == [0]
        @test __constraint(o, :eq6)[3] == [0]
        @test __constraint(o, :eq7)[3] == [0]
        @test __constraint(o, :eq8)[3] == [0]
        @test __constraint(o, :eq9)[3] == [0]
        @test __constraint(o, :eq10)[3] == [0]
        @test __constraint(o, :eq11)[3] == [0]
        @test __constraint(o, :eq12)[3] == [0]
        @test __constraint(o, :eq13)[3] == [0]
        @test __constraint(o, :eq14)[3] == [0]
        @test __constraint(o, :eq15)[3] == [0]
        @test __constraint(o, :eq1)[4] == [Inf]
        @test __constraint(o, :eq2)[4] == [Inf]
        @test __constraint(o, :eq3)[4] == [Inf]
        @test __constraint(o, :eq4)[4] == [Inf]
        @test __constraint(o, :eq5)[4] == [Inf]
        @test __constraint(o, :eq6)[4] == [Inf]
        @test __constraint(o, :eq7)[4] == [Inf]
        @test __constraint(o, :eq8)[4] == [Inf]
        @test __constraint(o, :eq9)[4] == [Inf]
        @test __constraint(o, :eq10)[4] == [Inf]
        @test __constraint(o, :eq11)[4] == [Inf]
        @test __constraint(o, :eq12)[4] == [Inf]
        @test __constraint(o, :eq13)[4] == [Inf]
        @test __constraint(o, :eq14)[4] == [Inf]
        @test __constraint(o, :eq15)[4] == [Inf]

        @def o begin
            v ∈ R^2, variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R², control
            x(0) ≤ [0, 0]
            x(0) ≤ [0, 0], (1)
            x(1) ≤ [0, 0]
            x(1) ≤ [0, 0], (2)
            [x₁(0)^3, 0] ≤ [0, 0]
            [x₁(0)^3, 0] ≤ [0, 0], (3)
            x(t) ≤ [0, 0]
            x(t) ≤ [0, 0], (4)
            u(t) ≤ [0, 0]
            u(t) ≤ [0, 0], (5)
            [x₁(t)^3, 0] ≤ [0, 0]
            [x₁(t)^3, 0] ≤ [0, 0], (6)
            [u₁(t)^3, 0] ≤ [0, 0]
            [u₁(t)^3, 0] ≤ [0, 0], (7)
            [u₁(t)^3, x₁(t)] ≤ [0, 0]
            [u₁(t)^3, x₁(t)] ≤ [0, 0], (8)
            v ≤ [0, 0]
            v ≤ [0, 0], (9)
            [v₁^2, 0] ≤ [0, 0]
            [v₁^2, 0] ≤ [0, 0], (10)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end

        @test __constraint(o, :eq1)[3] == -[Inf, Inf]
        @test __constraint(o, :eq2)[3] == -[Inf, Inf]
        @test __constraint(o, :eq3)[3] == -[Inf, Inf]
        @test __constraint(o, :eq4)[3] == -[Inf, Inf]
        @test __constraint(o, :eq5)[3] == -[Inf, Inf]
        @test __constraint(o, :eq6)[3] == -[Inf, Inf]
        @test __constraint(o, :eq7)[3] == -[Inf, Inf]
        @test __constraint(o, :eq8)[3] == -[Inf, Inf]
        @test __constraint(o, :eq9)[3] == -[Inf, Inf]
        @test __constraint(o, :eq10)[3] == -[Inf, Inf]
        @test __constraint(o, :eq1)[4] == [0, 0]
        @test __constraint(o, :eq2)[4] == [0, 0]
        @test __constraint(o, :eq3)[4] == [0, 0]
        @test __constraint(o, :eq4)[4] == [0, 0]
        @test __constraint(o, :eq5)[4] == [0, 0]
        @test __constraint(o, :eq6)[4] == [0, 0]
        @test __constraint(o, :eq7)[4] == [0, 0]
        @test __constraint(o, :eq8)[4] == [0, 0]
        @test __constraint(o, :eq9)[4] == [0, 0]
        @test __constraint(o, :eq10)[4] == [0, 0]

        @def o begin
            v ∈ R^2, variable
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R², control
            x(0) ≥ [0, 0]
            x(0) ≥ [0, 0], (1)
            x(1) ≥ [0, 0]
            x(1) ≥ [0, 0], (2)
            [x₁(0)^3, 0] ≥ [0, 0]
            [x₁(0)^3, 0] ≥ [0, 0], (3)
            x(t) ≥ [0, 0]
            x(t) ≥ [0, 0], (4)
            u(t) ≥ [0, 0]
            u(t) ≥ [0, 0], (5)
            [x₁(t)^3, 0] ≥ [0, 0]
            [x₁(t)^3, 0] ≥ [0, 0], (6)
            [u₁(t)^3, 0] ≥ [0, 0]
            [u₁(t)^3, 0] ≥ [0, 0], (7)
            [u₁(t)^3, x₁(t)] ≥ [0, 0]
            [u₁(t)^3, x₁(t)] ≥ [0, 0], (8)
            v ≥ [0, 0]
            v ≥ [0, 0], (9)
            [v₁^2, 0] ≥ [0, 0]
            [v₁^2, 0] ≥ [0, 0], (10)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end

        @test __constraint(o, :eq1)[4] == [Inf, Inf]
        @test __constraint(o, :eq2)[4] == [Inf, Inf]
        @test __constraint(o, :eq3)[4] == [Inf, Inf]
        @test __constraint(o, :eq4)[4] == [Inf, Inf]
        @test __constraint(o, :eq5)[4] == [Inf, Inf]
        @test __constraint(o, :eq6)[4] == [Inf, Inf]
        @test __constraint(o, :eq7)[4] == [Inf, Inf]
        @test __constraint(o, :eq8)[4] == [Inf, Inf]
        @test __constraint(o, :eq9)[4] == [Inf, Inf]
        @test __constraint(o, :eq10)[4] == [Inf, Inf]
        @test __constraint(o, :eq1)[3] == [0, 0]
        @test __constraint(o, :eq2)[3] == [0, 0]
        @test __constraint(o, :eq3)[3] == [0, 0]
        @test __constraint(o, :eq4)[3] == [0, 0]
        @test __constraint(o, :eq5)[3] == [0, 0]
        @test __constraint(o, :eq6)[3] == [0, 0]
        @test __constraint(o, :eq7)[3] == [0, 0]
        @test __constraint(o, :eq8)[3] == [0, 0]
        @test __constraint(o, :eq9)[3] == [0, 0]
        @test __constraint(o, :eq10)[3] == [0, 0]

        t0 = 9.0
        tf = 9.1
        r0 = 1.0
        r1 = 2.0
        v0 = 2.0
        vmax = sqrt(2)
        m0 = 3.0
        mf = 1.1
        @def ocp begin
            t ∈ [t0, tf], time
            x ∈ R^3, state
            u ∈ R^2, control

            m = x₂

            x(t0) == [r0, v0, m0], (1)
            0 ≤ u[1](t) ≤ 1, (deux)
            r0 ≤ x(t)[1] ≤ r1, (trois)
            0 ≤ x₂(t) ≤ vmax, (quatre)
            mf ≤ m(t) ≤ m0, (5)
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_name(ocp) == "u"
        @test control_dimension(ocp) == 2
        @test state_name(ocp) == "x"
        @test state_dimension(ocp) == 3

        @def ocp begin
            t ∈ [t0, tf], time
            x ∈ R^3, state
            u ∈ R^2, control

            m = x₂

            x(t0) == [r0, v0, m0]
            0 ≤ u(t)[2] ≤ 1
            r0 ≤ x(t)[1] ≤ r1
            0 ≤ x₂(t) ≤ vmax
            mf ≤ m(t) ≤ m0
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "t"
        @test initial_time(ocp) == t0
        @test final_time(ocp) == tf
        @test control_name(ocp) == "u"
        @test control_dimension(ocp) == 2
        @test state_name(ocp) == "x"
        @test state_dimension(ocp) == 3

        # dyslexic definition:  t -> u -> x -> t
        u0 = 9.0
        uf = 9.1
        z0 = 1.0
        z1 = 2.0
        k0 = 2.0
        kmax = sqrt(2)
        b0 = 3.0
        bf = 1.1

        @def ocp begin
            u ∈ [u0, uf], time
            t ∈ R^3, state
            x ∈ R^2, control
            b = t₂
            t(u0) == [z0, k0, b0]
            0 ≤ x[2](u) ≤ 1
            z0 ≤ t(u)[1] ≤ z1
            0 ≤ t₂(u) ≤ kmax
            bf ≤ b(u) ≤ b0
            derivative(t)(u) == t(u) # generic (untested)
            0 => min # generic (untested)u in R, control # generic (untested)
        end
        @test ocp isa Model
        @test time_name(ocp) == "u"
        @test initial_time(ocp) == u0
        @test final_time(ocp) == uf
        @test control_name(ocp) == "x"
        @test control_dimension(ocp) == 2
        @test state_name(ocp) == "t"
        @test state_dimension(ocp) == 3

        #
        # test all constraints on @def macro
        #

        #function test_ctparser_constraints()

        # all used variables must be definedbefore each test
        x0 = [1, 2, 11.11]
        x02 = 11.111
        x0_b = 11.1111
        x0_u = 11.11111
        y0 = [1, 2.22]
        y0_b = [1, 2.222]
        y0_u = [2, 2.2222]

        # === initial
        t0 = 0.0
        tf = 1.0
        n = 3
        @def ocp1 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x(t0) == x0
            x[2](t0) == x02
            x[2:3](t0) == y0
            x0_b ≤ x₂(t0) ≤ x0_u
            y0_b ≤ x[2:3](t0) ≤ y0_u
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)u in R, control # generic (untested)
        end
        @test ocp1 isa Model
        @test state_dimension(ocp1) == n
        @test control_dimension(ocp1) == n
        @test initial_time(ocp1) == t0
        @test final_time(ocp1) == tf

        t0 = 0.1
        tf = 1.1
        x0 = ones(4)
        n = 4
        @def ocp2 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control

            x(t0) == x0, initial_1
            x[2](t0) == 1, initial_2
            x[2:3](t0) == [1, 2], initial_3
            x0 ≤ x(t0) ≤ x0 .+ 1, initial_4
            [1, 2] ≤ x[2:3](t0) ≤ [3, 4], initial_5
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp2 isa Model
        @test state_dimension(ocp2) == n
        @test control_dimension(ocp2) == n
        @test initial_time(ocp2) == t0
        @test final_time(ocp2) == tf

        # all used variables must be defined before each test
        xf = 11.11 * ones(4)
        xf2 = 11.111
        xf_b = 11.1111 * ones(4)
        xf_u = 11.11111 * ones(4)
        yf = 2.22 * ones(2)
        yf_b = 2.222 * ones(2)
        yf_u = 2.2222 * ones(2)

        # === final
        t0 = 0.2
        tf = 1.2
        n = 4
        @def ocp3 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x(tf) == xf
            xf_b ≤ x(tf) ≤ xf_u
            x[2](tf) == xf2
            x[2:3](tf) == yf
            yf_b ≤ x[2:3](tf) ≤ yf_u
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp3 isa Model
        @test state_dimension(ocp3) == n
        @test control_dimension(ocp3) == n
        @test initial_time(ocp3) == t0
        @test final_time(ocp3) == tf

        t0 = 0.3
        tf = 1.3
        n = 6
        xf = 11.11 * ones(n)
        xf2 = 11.111
        xf_b = 11.1111 * ones(n)
        xf_u = 11.11111 * ones(n)
        yf = 2.22 * ones(2)
        yf_b = 2.222 * ones(2)
        yf_u = 2.2222 * ones(2)

        @def ocp4 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x(tf) == xf, final_1
            xf_b ≤ x(tf) ≤ xf_u, final_2
            x[2](tf) == xf2, final_3
            x[2:3](tf) == yf, final_4
            yf_b ≤ x[2:3](tf) ≤ yf_u, final_5
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp4 isa Model
        @test state_dimension(ocp4) == n
        @test control_dimension(ocp4) == n
        @test initial_time(ocp4) == t0
        @test final_time(ocp4) == tf

        # === boundary
        t0 = 0.4
        tf = 1.4
        n = 2
        @def ocp5 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x(tf) - tf * x(t0) == [0, 1]
            [0, 1] ≤ x(tf) - tf * x(t0) ≤ [1, 3]
            x[2](t0)^2 == 1
            1 ≤ x[2](t0)^2 ≤ 2
            x[2](tf)^2 == 1
            1 ≤ x[2](tf)^2 ≤ 2
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp5 isa Model
        @test state_dimension(ocp5) == n
        @test control_dimension(ocp5) == n
        @test initial_time(ocp5) == t0
        @test final_time(ocp5) == tf

        t0 = 0.5
        tf = 1.5
        n = 2
        @def ocp6 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x(tf) - tf * x(t0) == [0, 1], boundary_1
            [0, 1] ≤ x(tf) - tf * x(t0) ≤ [1, 3], boundary_2
            x[2](t0)^2 == 1, boundary_3
            1 ≤ x[2](t0)^2 ≤ 2, boundary_4
            x[2](tf)^2 == 1, boundary_5
            1 ≤ x[2](tf)^2 ≤ 2, boundary_6
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp6 isa Model
        @test state_dimension(ocp6) == n
        @test control_dimension(ocp6) == n
        @test initial_time(ocp6) == t0
        @test final_time(ocp6) == tf

        # define more variables
        u_b = 1.0
        u_u = 2.0
        u2_b = 3.0
        u2_u = 4.0
        v_b = 5.0
        v_u = 6.0

        t0 = 0.6
        tf = 1.6
        n = 2
        @def ocp7 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            u_b ≤ u[1](t) ≤ u_u
            u2_b ≤ u[1](t) ≤ u2_u
            v_b ≤ u[2](t) ≤ v_u
            #u[2:3](t) == v_u
            u[1](t)^2 + u[2](t)^2 == 1
            1 ≤ u[1](t)^2 + u[2](t)^2 ≤ 2
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp7 isa Model
        @test state_dimension(ocp7) == n
        @test control_dimension(ocp7) == n
        @test initial_time(ocp7) == t0
        @test final_time(ocp7) == tf

        t0 = 0.7
        tf = 1.7
        n = 2
        u_b = 1.0
        u_u = 2.0
        u2_b = 3.0
        u2_u = 4.0
        v_b = 5.0
        v_u = 6.0
        @def ocp8 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            u_b ≤ u[2](t) ≤ u_u, control_1
            u2_b ≤ u[1](t) ≤ u2_u, control_3
            [1, v_b] ≤ u[1:2](t) ≤ [2, v_u], control_5
            u[1](t)^2 + u[2](t)^2 == 1, control_7
            1 ≤ u[1](t)^2 + u[2](t)^2 ≤ 2, control_8
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp8 isa Model
        @test state_dimension(ocp8) == n
        @test control_dimension(ocp8) == n
        @test initial_time(ocp8) == t0
        @test final_time(ocp8) == tf

        # more vars
        x_b = 10.0
        x_u = 11.0
        x2_b = 13.0
        x2_u = 14.0
        x_u = 15.0
        y_u = 16.0

        # === state
        t0 = 0.8
        tf = 1.8
        n = 10
        @def ocp9 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x_b ≤ x[3](t) ≤ x_u
            #x(t) == x_u
            x2_b ≤ x[2](t) ≤ x2_u
            #x[2](t) == x2_u
            #x[2:3](t) == y_u
            x_u ≤ x[10](t) ≤ y_u
            x[1:2](t) + x[3:4](t) == [-1, 1]
            [-1, 1] ≤ x[1:2](t) + x[3:4](t) ≤ [0, 2]
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp9 isa Model
        @test state_dimension(ocp9) == n
        @test control_dimension(ocp9) == n
        @test initial_time(ocp9) == t0
        @test final_time(ocp9) == tf

        t0 = 0.9
        tf = 1.9
        n = 11
        @def ocp10 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            x_b ≤ x[3](t) ≤ x_u, state_1
            #x(t) == x_u                                  , state_2
            x2_b ≤ x[2](t) ≤ x2_u, state_3
            #x[2](t) == x2_u                              , state_4
            #x[2:3](t) == y_u                             , state_5
            x_u ≤ x[3](t) ≤ y_u, state_6
            x[1:2](t) + x[3:4](t) == [-1, 1], state_7
            [-1, 1] ≤ x[1:2](t) + x[3:4](t) ≤ [0, 2], state_8
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp10 isa Model
        @test state_dimension(ocp10) == n
        @test control_dimension(ocp10) == n
        @test initial_time(ocp10) == t0
        @test final_time(ocp10) == tf

        # === mixed
        t0 = 0.111
        tf = 1.111
        n = 12
        @def ocp11 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            u[2](t) * x[1:2](t) == [-1, 1]
            [-1, 1] ≤ u[2](t) * x[1:2](t) ≤ [0, 2]
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp11 isa Model
        @test state_dimension(ocp11) == n
        @test control_dimension(ocp11) == n
        @test initial_time(ocp11) == t0
        @test final_time(ocp11) == tf

        @def ocp12 begin
            t ∈ [t0, tf], time
            x ∈ R^n, state
            u ∈ R^n, control
            u[2](t) * x[1:2](t) == [-1, 1], mixed_1
            [-1, 1] ≤ u[2](t) * x[1:2](t) ≤ [0, 2], mixed_2
            derivative(x)(t) == x(t) # generic (untested)
            0 => min # generic (untested)
        end
        @test ocp12 isa Model
        @test state_dimension(ocp12) == n
        @test control_dimension(ocp12) == n
        @test initial_time(ocp12) == t0
        @test final_time(ocp12) == tf

        # === dynamics

        t0 = 0.112
        tf = 1.112
        @def ocp13 begin
            t ∈ [t0, tf], time
            x ∈ R, state
            u ∈ R, control
            ẋ(t) == 2x(t) + u(t)^2
            0 => min # generic (untested)
        end
        @test ocp13 isa Model
        @test state_dimension(ocp13) == 1
        @test control_dimension(ocp13) == 1
        @test initial_time(ocp13) == t0
        @test final_time(ocp13) == tf

        # some syntax (even parseable) are not allowed
        # this is the actual exhaustive list
        # note: equality constraints on ranges for state and control
        # are now allowed to ensure a uniform treatment of equalities
        # as particular inequalities
        @test_throws ParsingError @def o begin
            t ∈ [t0, tf], time
            x ∈ R, state
            u ∈ R, control
            ẋ(t) == f(x(t), u(t)), named_dynamics_not_allowed  # but allowed if unnamed !
            0 => min # generic (untested)
        end
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "Lagrange cost" begin
        println("lagrange testset...")

        # --------------------------------
        # min
        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            ∫(0.5u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            -∫(0.5u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == -0.5u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            0.5 * ∫(u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            0.5∫(u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            -0.5 * ∫(u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == -0.5u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            (-0.5 + tf) * ∫(u(t)^2) → min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == (-0.5 + tf) * u[1]^2
        @test criterion(o) == :min

        t0 = 0
        tf = 1
        @test_throws ParsingError @def o begin # a call to the time (t, here) must not appear before the integral
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            (-0.5 + t) * ∫(u(t)^2) → min
        end

        t0 = 0
        tf = 1
        @test_throws ParsingError @def o begin # a call to the time (t, here) must not appear before the integral
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            (-0.5 + x(t)) * ∫(u(t)^2) → min
        end

        # -----------------------------------
        # max 
        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            ∫(0.5u(t)^2) → max
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :max

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            -∫(0.5u(t)^2) → max
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == -0.5u[1]^2
        @test criterion(o) == :max

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            0.5 * ∫(u(t)^2) → max
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :max

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            0.5∫(u(t)^2) → max
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :max

        t0 = 0
        tf = 1
        @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            -0.5 * ∫(u(t)^2) → max
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == -0.5u[1]^2
        @test criterion(o) == :max

        # -----------------------------------
        t0 = 0.0
        tf = 0.1
        @def ocp begin
            t ∈ [t0, tf], time
            x ∈ R^3, state
            u ∈ R^3, control
            ∫(0.5u(t)^2) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        @test ocp isa Model

        t0 = 0.0
        tf = 0.1
        @def ocp begin
            t ∈ [t0, tf], time
            x ∈ R^3, state
            u ∈ R^3, control
            ∫(0.5u(t)^2) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        @test ocp isa Model
    end

    t0 = 0
    tf = 1
    @test_throws ParsingError @def o begin # a call to the time (t, here) must not appear before the integral
        t ∈ [t0, tf], time
        x ∈ R^2, state
        u ∈ R, control
        x(t0) == [-1, 0], (1)
        x(tf) == [0, 0]
        ẋ(t) == A * x(t) + B * u(t)
        (-0.5 + t) * ∫(u(t)^2) → max
    end

    t0 = 0
    tf = 1
    @test_throws ParsingError @def o begin # a call to the time (t, here) must not appear before the integral
        t ∈ [t0, tf], time
        x ∈ R^2, state
        u ∈ R, control
        x(t0) == [-1, 0], (1)
        x(tf) == [0, 0]
        ẋ(t) == A * x(t) + B * u(t)
        (-0.5 + x(t)) * ∫(u(t)^2) → max
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "Bolza cost" begin
        println("Bolza testset...")

        # -------------------------------
        # min 
        # Mayer ± Lagrange
        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 3x(1)) + ∫(x(t) + u(t)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 3xf[1]
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            derivative(x)(t) == x(t) # generic (untested)
            (x(0) + 2x(1)) + ∫(x(t) + u(t)) → min
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) + 2 * ∫(x(t) + u(t)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - ∫(x(t) + u(t)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = 1
        u = 2
        x0 = 3
        xf = 4
        @test mayer(o)(x0, xf, nothing) == x0 + 2xf
        @test lagrange(o)(0, x, u, nothing) == -(x[1] + u[1])
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - 2 * ∫(x(t) + u(t)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == -2(x[1] + u[1])
        @test criterion(o) == :min

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) + t * ∫(x(t) + u(t)) → min
        end

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - t * ∫(x(t) + u(t)) → min
        end

        # -------------------------------
        # max 
        # Mayer ± Lagrange
        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 5x(1)) + ∫(x(t) + u(t)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 5xf[1]
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) + 2 * ∫(x(t) + u(t)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - ∫(x(t) + u(t)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == -(x[1] + u[1])
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - 2 * ∫(x(t) + u(t)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == -2(x[1] + u[1])
        @test criterion(o) == :max

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) + t * ∫(x(t) + u(t)) → max
        end

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            (x(0) + 2x(1)) - t * ∫(x(t) + u(t)) → max
        end

        # -------------------------------
        # min 
        # Lagrange ± Mayer
        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ∫(x(t) + u(t)) + (x(0) + 2x(1)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            2 * ∫(x(t) + u(t)) + (x(0) + 2x(1)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ∫(x(t) + u(t)) - (x(0) + 2x(1)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == -(x0[1] + 2xf[1])
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :min

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            2 * ∫(x(t) + u(t)) - (x(0) + 2x(1)) → min
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [1]
        x0 = [1]
        xf = [1]
        @test mayer(o)(x0, xf, nothing) == -(x0[1] + 2xf[1])
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :min

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            t * ∫(x(t) + u(t)) + 1 → min
        end

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            t * ∫(x(t) + u(t)) - 1 → min
        end

        # -------------------------------
        # max
        # Lagrange ± Mayer
        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ∫(x(t) + u(t)) + (x(0) + 2x(1)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            2 * ∫(x(t) + u(t)) + (x(0) + 2x(1)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == x0[1] + 2xf[1]
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            ∫(x(t) + u(t)) - (x(0) + 2x(1)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == -(x0[1] + 2xf[1])
        @test lagrange(o)(0, x, u, nothing) == x[1] + u[1]
        @test criterion(o) == :max

        @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            2 * ∫(x(t) + u(t)) - (x(0) + 2x(1)) → max
            derivative(x)(t) == x(t) # generic (untested)
        end
        x = [1]
        u = [2]
        x0 = [3]
        xf = [4]
        @test mayer(o)(x0, xf, nothing) == -(x0[1] + 2xf[1])
        @test lagrange(o)(0, x, u, nothing) == 2(x[1] + u[1])
        @test criterion(o) == :max

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            t * ∫(x(t) + u(t)) + 1 → max
        end

        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R, control
            t * ∫(x(t) + u(t)) - 1 → max
        end
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "Mayer cost" begin
        println("Mayer testset...")

        @def o begin
            s ∈ [0, 1], time
            y ∈ R^4, state
            w ∈ R, control
            r = y₃
            v = y₄
            r(0) + v(1) → min
            derivative(y)(s) == y(s) # generic (untested)
        end
        y0 = [1, 2, 3, 4]
        yf = 2 * [1, 2, 3, 4]
        @test criterion(o) == :min
        @test mayer(o)(y0, yf, nothing) == y0[3] + yf[4]

        @def o begin
            s ∈ [0, 1], time
            y ∈ R^4, state
            w ∈ R, control
            r = y₃
            v = y₄
            r(0) + v(1) → max
            derivative(y)(s) == y(s) # generic (untested)
        end
        y0 = [1, 2, 3, 4]
        yf = 2 * [1, 2, 3, 4]
        @test criterion(o) == :max
        @test mayer(o)(y0, yf, nothing) == y0[3] + yf[4]

        @def o begin
            z ∈ R^2, variable
            s ∈ [0, z₁], time
            y ∈ R^4, state
            w ∈ R, control
            r = y₃
            v = y₄
            r(0) + v(z₁) + z₂ → min
            derivative(y)(s) == y(s) # generic (untested)
        end
        z = [5, 6]
        y0 = [1, 2, 3, 4]
        yf = 2 * [1, 2, 3, 4]
        @test criterion(o) == :min
        @test mayer(o)(y0, yf, z) == y0[3] + yf[4] + z[2]

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁ + w^2 + v^3 + z₂
            ẏ(s) == [aa(s), (r^2)(s), 0, 0]
            r(0) + v(z₁) + z₂ → min
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        w = [7]
        @test __dynamics(o)(0, y, w, z) == [y[1] + w[1]^2 + y[4]^3 + z[2], y[3]^2, 0, 0]
        @test mayer(o)(y0, yf, z) == y0[3] + yf[4] + z[2]

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(s) + v^3 + z₂
            ẏ(s) == [aa(s) + (w^2)(s), r(s)^2, 0, 0]
            r(0) + v(z₁) + z₂ → min
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        w = [7]
        @test __dynamics(o)(0, y, w, z) == [y[1] + w[1]^2 + y[4]^3 + z[2], y[3]^2, 0, 0]
        @test mayer(o)(y0, yf, z) == y0[3] + yf[4] + z[2]

        @def o begin
            z ∈ R², variable
            s ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁ + v^3 + z₂
            aa(0) + y₂(z₁) → min
            derivative(y)(s) == y(s) # generic (untested)
        end
        z = [5, 6]
        y0 = y
        yf = 3y0
        @test mayer(o)(y0, yf, z) == y0[1] + y0[4]^3 + z[2] + yf[2]

        @def o begin
            z ∈ R², variable
            __t ∈ [0, z₁], time
            y ∈ R⁴, state
            w ∈ R, control
            r = y₃
            v = y₄
            aa = y₁(__t) + v^3 + z₂
            ẏ(__t) == [aa(__t) + (w^2)(__t), r(__t)^2, 0, 0]
            aa(0) + y₂(z₁) → min
        end
        z = [5, 6]
        y = [1, 2, 3, 4]
        y0 = y
        yf = 3y0
        w = [11]
        @test __dynamics(o)(0, y, w, z) == [y[1] + w[1]^2 + y[4]^3 + z[2], y[3]^2, 0, 0]
        @test_throws UndefVarError mayer(o)(y0, yf, z)
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "closure" begin
        println("closure testset...")

        a = 1
        f(b) = begin # closure of a, local c, and @def in function
            c = 3
            @def ocp begin
                t ∈ [a, b], time
                x ∈ R, state
                u ∈ R, control
                ẋ(t) == x(t) + u(t) + b + c + d
                0 => min # generic (untested)
            end
            return ocp
        end
        b = 2
        o = f(b)
        d = 4
        x = [10]
        u = [20]
        @test __dynamics(o)(0, x, u, nothing) == x + u + [b + 3 + d]
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "error detection" begin

        # error detections (this can be tricky -> need more work)

        # this one is detected by the generated code (and not the parser)
        t0 = 9.0
        tf = 9.1
        @test_throws UnauthorizedCall @def o begin
            t ∈ [t0, tf], time
            t ∈ [t0, tf], time
        end

        # illegal constraint name (1bis), detected by the parser
        t0 = 9.0
        tf = 9.1
        r0 = 1.0
        v0 = 2.0
        m0 = 3.0
        @test_throws ParsingError @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R^2, control
            0 ≤ u(t) ≤ 1, (1bis)
        end

        # t0 is unknown in the x(t0) constraint, detected by the parser
        r0 = 1.0
        v0 = 2.0
        m0 = 3.0
        @test_throws ParsingError @def o begin
            t ∈ [0, 1], time
            x ∈ R^2, state
            u ∈ R^2, control
            x(t0) == [r0, v0, m0], (1)
            0 ≤ u(t) ≤ 1, (1bis)
        end

        # bad syntax for Bolza cost interpreted as a Mayer term with trailing ∫ 
        @test_throws ParsingError @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            1 + 2 + ∫(u(t)^2) → min # should be ( 1 + 2 ) + ∫(...)
        end

        @test_throws ParsingError @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            ∫(u(t)^2) + 1 + 2 → min # should be ∫(...) + ( 1 + 2 )
        end

        @test_throws ParsingError @def o begin
            t ∈ [t0, tf], time
            x ∈ R^2, state
            u ∈ R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            ẋ(t) == A * x(t) + B * u(t)
            ∫(u(t)^2) / 2 → min # forbidden
        end
    end

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "non unicode keywords" begin
        println("non unicode keywords testset...")

        # --------------------------------
        # min
        t0 = 0
        tf = 1
        @def o begin
            t in [t0, tf], time
            x in R^2, state
            u in R, control
            x(t0) == [-1, 0], (1)
            x(tf) == [0, 0]
            derivative(x)(t) == A * x(t) + B * u(t)
            integral(0.5u(t)^2) => min
        end
        x = [1, 2]
        x0 = 2 * x
        xf = 3 * x
        u = [-1]
        A = [
            0 1
            0 0
        ]
        B = [ 0 1 ]'
        @test __constraint(o, :eq1)[2](x0, xf, nothing) == x0
        @test __dynamics(o)(0, x, u, nothing) == A * x + B * u
        @test lagrange(o)(0, x, u, nothing) == 0.5u[1]^2
        @test criterion(o) == :min

        @def o begin
            z in R, variable
            t in [0, 1], time
            x in R^2, state
            u in R, control
            r = x[1]
            v = x[2]
            0 <= r(0) - z <= 1, (1)
            0 <= v(1)^2 <= 1, (2)
            [0, 0] <= x(0) <= [1, 1], (♡)
            z >= 0, (3)
            derivative(x)(t) == [v(t), r(t)^2 + z]
            integral(u(t)^2 + z * x[1](t)) => min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        z = [4]
        @test __constraint(o, :eq1)[2](x0, xf, z) == x0[1:1] - z
        @test __constraint(o, :eq2)[2](x0, xf, z) == [xf[2]^2]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0
        @test __constraint(o, :eq3)[2](x0, xf, z) == z
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]

        @def o begin
            z in R, variable
            t in [0, 1], time
            x in R^2, state
            u in R, control
            r = x[1]
            v = x[2]
            0 <= r(0) - z <= 1, (1)
            0 <= v(1)^2 <= 1, (2)
            [0, 0] <= x(0) <= [1, 1], (♡)
            z >= 0, (3)
            derivative(x)(t) == [v(t), r(t)^2 + z]
            integral(u(t)^2 + z * x[1](t)) => min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3]
        z = [4]
        @test __constraint(o, :eq1)[2](x0, xf, z) == x0[1:1] - z
        @test __constraint(o, :eq2)[2](x0, xf, z) == [xf[2]^2]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0
        @test __constraint(o, :eq3)[2](x0, xf, z) == z
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]

        @def o begin
            z in R^2, variable
            t in [0, 1], time
            x in R^2, state
            u in R^2, control
            r = x1
            v = x2
            0 <= r(0) - z1 <= 1, (1)
            0 <= v(1)^2 <= 1, (2)
            [0, 0] <= x(0) <= [1, 1], (♡)
            z1 >= 0, (3)
            z2 == 1, (4)
            u2(t) == 0
            derivative(x)(t) == [v(t), r(t)^2 + z1]
            integral(u1(t)^2 + z1 * x1(t)) => min
        end
        x0 = [2, 3]
        xf = [4, 5]
        x = [1, 2]
        u = [3, 0]
        z = [4, 1]
        @test __constraint(o, :eq1)[2](x0, xf, z) == x0[1:1] - z[1:1]
        @test __constraint(o, :eq2)[2](x0, xf, z) == [xf[2]^2]
        @test __constraint(o, Symbol("♡"))[2](x0, xf, z) == x0
        @test __constraint(o, :eq3)[2](x0, xf, z) == z[1:1]
        @test __constraint(o, :eq4)[2](x0, xf, z) == z[2:2]
        @test __dynamics(o)(0, x, u, z) == [x[2], x[1]^2 + z[1]]
        @test lagrange(o)(0, x, u, z) == u[1]^2 + z[1] * x[1]
    end
    
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    @testset "scalar variable, state, constraint" begin
        println("scalar variable, state, constraint...")

        # variable
        o = @def begin
            v ∈ R, variable
            t ∈ [0, v], time
            x ∈ R², state
            u ∈ R², control
            derivative(x)(t) == t * v * x(t) + u(t)
            1 → min 
        end

        t = 7.
        v = [2.]
        x = [3., 4.]
        u = [10., 20.]
        r = similar(x)
        @test final_time(o, v) == v[1]
        dynamics(o)(r, t, x, u, v)
        @test r == t * v[1] * x + u

        o = @def begin
            v ∈ R, variable
            t ∈ [0, v₁], time
            x ∈ R², state
            u ∈ R², control
            derivative(x)(t) == t * v₁ * x(t) + u(t)
            1 → min 
        end

        @test final_time(o, v) == v[1]
        dynamics(o)(r, t, x, u, v)
        @test r == t * v[1] * x + u

        o = @def begin
            v ∈ R, variable
            t ∈ [0, v1], time
            x ∈ R², state
            u ∈ R², control
            derivative(x)(t) == t * v1 * x(t) + u(t)
            1 → min 
        end
        
        @test final_time(o, v) == v[1]
        dynamics(o)(r, t, x, u, v)
        @test r == t * v[1] * x + u

        o = @def begin
            v ∈ R, variable
            t ∈ [0, v[1]], time
            x ∈ R², state
            u ∈ R², control
            derivative(x)(t) == t * v[1] * x(t) + u(t)
            1 → min 
        end

        @test final_time(o, v) == v[1]
        dynamics(o)(r, t, x, u, v)
        @test r == t * v[1] * x + u

        # state
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x)(t) == t * x(t) + u₁(t)
            1 → min 
        end

        v = nothing
        t = 7.
        x = [3.]
        u = [10., 20.]
        r = similar(x)
        dynamics(o)(r, t, x, u, v)
        @test r == [t * x[1] + u[1]]

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x[1])(t) == t * x(t) + u₁(t) # should also parse and be OK
            1 → min 
        end

        v = nothing
        t = 7.
        x = [3.]
        u = [10., 20.]
        r = similar(x)
        dynamics(o)(r, t, x, u, v)
        @test r == [t * x[1] + u[1]]

        @test_throws ParsingError @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x[2])(t) == t * x(t) + u₁(t) # out of range
            1 → min 
        end

        @test_throws UnauthorizedCall @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R², control
            derivative(x[1])(t) == t * x[1](t) + u₁(t) # incomplete
            1 → min 
        end

        @test_throws UnauthorizedCall @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            derivative(x[1:2])(t) == t * x[1](t) + u₁(t) # incomplete 
            1 → min 
        end

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x)(t) == t * x₁(t) + u₁(t)
            1 → min
        end

        @test r == [t * x[1] + u[1]]

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x)(t) == t * x1(t) + u₁(t)
            1 → min 
        end

        @test r == [t * x[1] + u[1]]

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R², control
            derivative(x)(t) == t * x[1](t) + u₁(t)
            1 → min 
        end

        @test r == [t * x[1] + u[1]]

        # control
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            derivative(x)(t) == t * u(t) * x(t)
            1 → min 
        end

        v = nothing
        t = 7.
        x = [3., 4.]
        u = [10.]
        r = similar(x)
        dynamics(o)(r, t, x, u, v)
        @test r == t * u[1] * x

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            derivative(x)(t) == t * u₁(t) * x(t)
            1 → min 
        end

        dynamics(o)(r, t, x, u, v)
        @test r == t * u[1] * x

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            derivative(x)(t) == t * u1(t) * x(t)
            1 → min 
        end

        dynamics(o)(r, t, x, u, v)
        @test r == t * u[1] * x

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            derivative(x)(t) == t * u[1](t) * x(t)
            1 → min 
        end

        dynamics(o)(r, t, x, u, v)
        @test r == t * u[1] * x

    end

    test_name = "pragma"
    @testset "$test_name" begin println(test_name)

        o = @def begin
            PRAGMA(println("foo"))
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            derivative(x)(t) == t * u[1](t) * x(t)
            1 → min 
        end
        @test o isa CTModels.Model 

    end

    test_name = "parsing_backends"
    @testset "$test_name" begin println(test_name)

        @test is_active_backend(:fun)
        activate_backend(:exa)
        @test is_active_backend(:exa)
        deactivate_backend(:exa)
        @test !is_active_backend(:exa)
        @test_throws String activate_backend(:fun) 
        @test_throws String deactivate_backend(:fun)
        @test_throws String activate_backend(:foo) 
        @test_throws String deactivate_backend(:foo) 

    end

    test_name = "dimensions at runtime"
    @testset "$test_name" begin println(test_name)

        n = 4
        o = @def begin
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^n, state
            u ∈ R, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        m = 1
        o = @def begin
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^4, state
            u ∈ R^m, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        k = 2
        o = @def begin
            v ∈ R^k, variable
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^4, state
            u ∈ R, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        o = @def begin
            v ∈ R^k, variable
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^n, state
            u ∈ R^1, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        o = @def begin
            v ∈ R^k, variable
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^4, state
            u ∈ R^m, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        o = @def begin
            v ∈ R^2, variable
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^n, state
            u ∈ R^m, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

        o = @def begin
            v ∈ R^k, variable
            t ∈ [0, 1], time
            x = (a, b, c, d) ∈ R^n, state
            u ∈ R^m, control
            ẋ(t) == x(t)
            ∫( 0.5u(t)^2 ) → min
        end

    end

end
