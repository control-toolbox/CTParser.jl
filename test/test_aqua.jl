function test_aqua()
    @testset "Aqua.jl" begin
        # Full Aqua test suite - now with zero type piracy!
        # All operations dispatch on our wrapper types (SymNumber, SymVector, SymMatrix)
        # instead of on ExaModels.AbstractNode, completely eliminating type piracy.
        Aqua.test_all(CTParser)
    end
end
