function test_aqua()
    @testset "Aqua.jl" begin
        Aqua.test_all(
            CTParser;
            ambiguities=true, # also tests submodules
            #stale_deps=(ignore=[:MLStyle],),
            deps_compat=(ignore=[:LinearAlgebra, :Unicode],),
            piracies=true,
        )
    end
end
