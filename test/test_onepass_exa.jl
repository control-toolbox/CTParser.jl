# test onepass, exa parsing (aka parsing towards ExaModels)
# todo:

parsing_backend!(:exa)

function test_onepass_exa()

    @testset "alias" begin

        o = @def begin
                # debug add a var and a cost (cannot build an ExaModel otherwise)
                a = x + y
            end
        @test o() isa ExaModels.ExaModel
        @test o(; grid_size = 100) isa ExaModels.ExaModel
        @test o(; backend = nothing) isa ExaModels.ExaModel
        @test o(; init = (0., 0., 0., 0., 0.)) isa ExaModels.ExaModel
        @test o(; base_type = Float32) isa ExaModels.ExaModel

    end

end