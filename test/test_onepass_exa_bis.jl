# test_onepass_exa_bis

function test_onepass_exa_bis()
    @testset "p_dynamics_exa! errors" begin
        println("p_dynamics_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics exa test"

        ex = CTParser.p_dynamics_exa!(p, :p_ocp, :x, :t, :(x(t)))
        @test ex isa Expr
        @test_throws ParsingError eval(ex)
    end

    @testset "parsing(:alias, backend)" begin
        println("parsing (bis)")

        # Valid backends should return callable primitives
        f_fun = CTParser.parsing(:alias, :fun)
        @test f_fun isa Function

        f_exa = CTParser.parsing(:alias, :exa)
        @test f_exa isa Function

        # Unknown backend should throw a String error
        @test_throws String CTParser.parsing(:alias, :unknown_backend)

        # Unknown primitive for a valid backend should raise a KeyError
        @test_throws KeyError CTParser.parsing(:unknown_primitive, :fun)
    end

    @testset "is_range / as_range (bis)" begin
        println("is_range/as_range (bis)")

        # Non-range symbols should not be considered ranges
        @test !CTParser.is_range(:x)

        # Plain AbstractRange
        @test CTParser.is_range(1:2:5)
        @test CTParser.as_range(1:2:5) == 1:2:5

        # Expr range
        @test CTParser.is_range(:(a:b))
        @test CTParser.as_range(:(a:b:c)) == :(a:b:c)

        # Fallback to single-element vector when not a range
        @test CTParser.as_range(:foo) == [:foo]

        # Edge cases
        @test !CTParser.is_range(42)
        @test !CTParser.is_range("string")
        @test !CTParser.is_range(:(f(x)))
        @test CTParser.is_range(1:10)
        @test CTParser.as_range(42) == [42]
        @test CTParser.as_range(:(x + y)) == [:(x + y)]
    end

    @testset "p_dynamics_coord_exa! preconditions" begin
        println("p_dynamics_coord_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "dynamics coord exa test"
        p.x = :x
        p.u = :u
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.dim_x = 2
        p_ocp = :p_ocp
        i = 1
        e = :(x[1](t) + u(t))

        # p_dynamics_coord_exa! should return an Expr (code generation)
        ex = CTParser.p_dynamics_coord_exa!(p, p_ocp, p.x, i, p.t, e)
        @test ex isa Expr

        # Check that dyn_coords is updated
        @test i in p.dyn_coords
    end

    @testset "PARSING_BACKENDS and PARSING_DIR" begin
        println("PARSING_BACKENDS/DIR (bis)")

        # Check that PARSING_BACKENDS contains expected backends
        @test :fun in CTParser.PARSING_BACKENDS
        @test :exa in CTParser.PARSING_BACKENDS

        # Check that PARSING_DIR has entries for both backends
        @test haskey(CTParser.PARSING_DIR, :fun)
        @test haskey(CTParser.PARSING_DIR, :exa)

        # Each entry should be a Dict-like structure with parsing primitives
        for backend in [:fun, :exa]
            dir = CTParser.PARSING_DIR[backend]
            @test haskey(dir, :alias)
            @test haskey(dir, :variable)
            @test haskey(dir, :time)
            @test haskey(dir, :state)
            @test haskey(dir, :control)
            @test haskey(dir, :constraint)
            @test haskey(dir, :dynamics)
            @test haskey(dir, :dynamics_coord)
        end
    end

    @testset "p_pragma! (exa vs fun)" begin
        println("p_pragma! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "pragma test"
        p_ocp = :p_ocp

        # p_pragma_fun! returns a LineNumberNode wrapper (no-op for :fun)
        ex_fun = CTParser.p_pragma_fun!(p, p_ocp, :(some_pragma))
        @test ex_fun isa Expr

        # p_pragma_exa! returns the pragma expression wrapped
        ex_exa = CTParser.p_pragma_exa!(p, p_ocp, :(some_pragma))
        @test ex_exa isa Expr
    end

    @testset "p_alias! for :exa" begin
        println("p_alias! exa (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "alias exa test"
        p_ocp = :p_ocp

        # p_alias_exa! is the same as p_alias_fun!
        @test CTParser.p_alias_exa! === CTParser.p_alias_fun!
    end

    @testset "constraint_type edge cases for :exa parsing" begin
        println("constraint_type for exa (bis)")

        t = :t
        t0 = 0
        tf = :tf
        x = :x
        u = :u
        v = :v

        # state_range with nothing (full state)
        e = :(x(t))
        c_type = constraint_type(e, t, t0, tf, x, u, v)
        @test c_type == (:state_range, nothing)

        # control_range with nothing (full control)
        e = :(u(t))
        c_type = constraint_type(e, t, t0, tf, x, u, v)
        @test c_type == (:control_range, nothing)

        # variable_range with nothing (full variable)
        e = :v
        c_type = constraint_type(e, t, t0, tf, x, u, v)
        @test c_type == (:variable_range, nothing)

        # initial with range expression
        e = :(x[1:3](0))
        c_type = constraint_type(e, t, t0, tf, x, u, v)
        @test c_type == (:initial, 1:3)

        # final with step range expression
        e = :(x[1:2:5](tf))
        c_type = constraint_type(e, t, t0, tf, x, u, v)
        @test c_type == (:final, 1:2:5)
    end

    @testset "__symgen uniqueness" begin
        println("__symgen (bis)")

        # Each call to __symgen should produce a unique symbol
        s1 = CTParser.__symgen(:test)
        s2 = CTParser.__symgen(:test)
        s3 = CTParser.__symgen(:other)

        @test s1 isa Symbol
        @test s2 isa Symbol
        @test s3 isa Symbol
        @test s1 != s2
        @test s1 != s3
        @test s2 != s3

        # Should start with the prefix
        @test startswith(string(s1), "test")
        @test startswith(string(s3), "other")
    end

    @testset "__init_aliases content" begin
        println("__init_aliases (bis)")

        al = CTParser.__init_aliases()

        # Check R^1 to R^20 aliases
        for i in 1:20
            key = Symbol(:R, CTBase.ctupperscripts(i))
            @test haskey(al, key)
            @test al[key] == :(R^$i)
        end

        # Check operator aliases
        @test al[:<=] == :≤
        @test al[:>=] == :≥
        @test al[:derivative] == :∂
        @test al[:integral] == :∫
        @test al[:(=>)] == :→
        @test al[:in] == :∈
    end

    # ============================================================================
    # EXA CODE GENERATION - Unit tests for :exa backend functions
    # ============================================================================

    @testset "p_variable_exa! code generation" begin
        println("p_variable_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "variable exa test"
        p_ocp = :p_ocp

        # p_variable_exa! should return an Expr for code generation
        ex = CTParser.p_variable_exa!(p, p_ocp, :v, 2, QuoteNode(:v))
        @test ex isa Expr

        # Check that box_v is updated with bounds code
        @test p.box_v isa Expr
    end

    @testset "p_time_exa! code generation" begin
        println("p_time_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "time exa test"
        p.t0 = 0
        p.tf = 1
        p_ocp = :p_ocp

        # p_time_exa! should return an Expr for dt calculation
        ex = CTParser.p_time_exa!(p, p_ocp, :t, 0, 1)
        @test ex isa Expr

        # The expression should contain dt assignment
        @test p.dt isa Symbol
    end

    @testset "p_state_exa! code generation" begin
        println("p_state_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "state exa test"
        p_ocp = :p_ocp

        # p_state_exa! should return an Expr for state variable creation
        ex = CTParser.p_state_exa!(p, p_ocp, :x, 3, QuoteNode(:x))
        @test ex isa Expr

        # Check that box_x is updated
        @test p.box_x isa Expr
    end

    @testset "p_control_exa! code generation" begin
        println("p_control_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "control exa test"
        p_ocp = :p_ocp

        # p_control_exa! should return an Expr for control variable creation
        ex = CTParser.p_control_exa!(p, p_ocp, :u, 2, QuoteNode(:u))
        @test ex isa Expr

        # Check that box_u is updated
        @test p.box_u isa Expr
    end

    @testset "p_mayer_exa! code generation" begin
        println("p_mayer_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "mayer exa test"
        p.x = :x
        p.t0 = 0
        p.tf = :tf
        p_ocp = :p_ocp
        e = :(x[1](tf))

        # p_mayer_exa! should return wrapped code
        ex = CTParser.p_mayer_exa!(p, p_ocp, e, :min)
        @test ex isa Expr
    end

    @testset "p_lagrange_exa! code generation" begin
        println("p_lagrange_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "lagrange exa test"
        p.x = :x
        p.u = :u
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.dt = :dt
        p_ocp = :p_ocp
        e = :(x[1](t)^2 + u[1](t)^2)

        # p_lagrange_exa! should return wrapped code with scheme conditionals
        ex = CTParser.p_lagrange_exa!(p, p_ocp, e, :min)
        @test ex isa Expr
    end

    @testset "p_bolza_exa! code generation" begin
        println("p_bolza_exa! (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "bolza exa test"
        p.x = :x
        p.u = :u
        p.t = :t
        p.t0 = 0
        p.tf = :tf
        p.dt = :dt
        p_ocp = :p_ocp
        e1 = :(x[1](tf))
        e2 = :(u[1](t)^2)

        # p_bolza_exa! should return wrapped code
        ex = CTParser.p_bolza_exa!(p, p_ocp, e1, e2, :min)
        @test ex isa Expr
    end

    # ============================================================================
    # CONSTRAINT EXA - Code generation tests
    # ============================================================================

    @testset "p_constraint_exa! boundary" begin
        println("p_constraint_exa! boundary (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa boundary test"
        p.x = :x
        p.u = :u
        p.v = :v
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.dim_x = 2
        p_ocp = :p_ocp

        # boundary constraint
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, 0, :(x[1](0) + x[1](1)), 0, :boundary, :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! initial" begin
        println("p_constraint_exa! initial (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa initial test"
        p.x = :x
        p.t0 = 0
        p.dim_x = 2
        p_ocp = :p_ocp

        # initial constraint with range
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, [0, 0], :(x(0)), [0, 0], (:initial, nothing), :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! final" begin
        println("p_constraint_exa! final (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa final test"
        p.x = :x
        p.tf = 1
        p.dim_x = 2
        p_ocp = :p_ocp

        # final constraint
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, [0, 0], :(x(1)), [0, 0], (:final, nothing), :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! variable_range" begin
        println("p_constraint_exa! variable_range (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa variable_range test"
        p.v = :v
        p.dim_v = 2
        p.l_v = :l_v
        p.u_v = :u_v
        p.box_v = :(LineNumberNode(0, "box"))
        p_ocp = :p_ocp

        # variable_range constraint
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, [0, 0], :(v), [1, 1], (:variable_range, nothing), :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! state_range" begin
        println("p_constraint_exa! state_range (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa state_range test"
        p.x = :x
        p.dim_x = 2
        p.l_x = :l_x
        p.u_x = :u_x
        p.box_x = :(LineNumberNode(0, "box"))
        p_ocp = :p_ocp

        # state_range constraint
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, [0, 0], :(x(t)), [1, 1], (:state_range, nothing), :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! control_range" begin
        println("p_constraint_exa! control_range (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa control_range test"
        p.u = :u
        p.dim_u = 1
        p.l_u = :l_u
        p.u_u = :u_u
        p.box_u = :(LineNumberNode(0, "box"))
        p_ocp = :p_ocp

        # control_range constraint
        ex = CTParser.p_constraint_exa!(
            p, p_ocp, [0], :(u(t)), [1], (:control_range, nothing), :c1
        )
        @test ex isa Expr
    end

    @testset "p_constraint_exa! mixed" begin
        println("p_constraint_exa! mixed (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa mixed test"
        p.x = :x
        p.u = :u
        p.v = :v
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.dt = :dt
        p_ocp = :p_ocp

        # mixed constraint (state + control)
        ex = CTParser.p_constraint_exa!(p, p_ocp, 0, :(x[1](t) + u[1](t)), 1, :mixed, :c1)
        @test ex isa Expr
    end

    @testset "p_constraint_exa! code generation returns Expr" begin
        println("p_constraint_exa! returns Expr (bis)")

        p = CTParser.ParsingInfo()
        p.lnum = 1
        p.line = "constraint exa code gen test"
        p.x = :x
        p.u = :u
        p.t = :t
        p.t0 = 0
        p.tf = 1
        p.dt = :dt
        p_ocp = :p_ocp

        # All valid constraint types should return an Expr
        # (We don't eval because it requires grid_size etc.)
        ex = CTParser.p_constraint_exa!(p, p_ocp, 0, :(x[1](t)), 1, :state_fun, :c1)
        @test ex isa Expr

        ex = CTParser.p_constraint_exa!(p, p_ocp, 0, :(u[1](t)), 1, :control_fun, :c1)
        @test ex isa Expr

        ex = CTParser.p_constraint_exa!(p, p_ocp, 0, :(x[1](0) + v), 1, :variable_fun, :c1)
        @test ex isa Expr
    end
end
