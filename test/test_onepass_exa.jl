# test onepass, exa parsing (aka parsing towards ExaModels)
# todo:

parsing_backend!(:exa)

function test_onepass_exa()

    @testset "alias" begin

        o = @def begin
                a = x + y
            end
        @test o() == nothing

    end

end