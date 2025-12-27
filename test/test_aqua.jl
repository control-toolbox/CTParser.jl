function test_aqua()
    @testset "Aqua.jl" begin
        Aqua.test_all(
            CTParser;
            ambiguities=false,
            #stale_deps=(ignore=[:MLStyle],),
            deps_compat=(ignore=[:LinearAlgebra, :Unicode],),
            # Disable piracy test: CTParser intentionally commits type piracy for Base.:*,
            # LinearAlgebra.dot, Base.adjoint, Base.transpose, LinearAlgebra.norm, and
            # LinearAlgebra.norm_sqr when working with ExaModels.AbstractNode arrays.
            # This is a deliberate design choice using holy traits internally for clean
            # architecture while providing operator overloads for familiar syntax.
            # Wrapper types were rejected because they break direct indexing (v[i] must
            # return AbstractNode, not a wrapper).
            piracies=false,
        )
        # Test ambiguities separately, allowing some from the symbolic linear algebra
        # Note: Method ambiguities from dot() and other symbolic ops are expected
        # due to interaction with SparseArrays and other LinearAlgebra types
        Aqua.test_ambiguities(CTParser; broken=true)
    end
end
