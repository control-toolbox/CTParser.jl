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
    end
end
