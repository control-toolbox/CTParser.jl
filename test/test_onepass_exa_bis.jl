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
end
