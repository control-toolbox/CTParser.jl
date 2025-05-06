# test onepass, exa parsing (aka parsing towards ExaModels)
# todo:

parsing_backend!(:exa)

function test_onepass_exa()

    @testset "alias" begin

        o = @def begin
                a = x + y
            end
        @test o() == nothing
        @test o(; grid_size=300) == nothing
        @test o(; backend=:foo) == nothing
        @test o(; init=.4) == nothing

    end

end