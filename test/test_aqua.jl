function test_aqua()
    @testset "Aqua.jl" begin
        Aqua.test_all(
            CTParser;
            ambiguities=false, # checked below with an explicit exclusion list
            #stale_deps=(ignore=[:MLStyle],),
            deps_compat=(ignore=[:Unicode],),
            # src/exa_linalg.jl is a deliberate, temporary port of ExaModels'
            # unregistered linear-algebra extension (#325): every method there
            # extends a Base/LinearAlgebra function on ExaModels node types, so
            # it is type piracy by construction. Listing the node types as our
            # own keeps the whitelist precise — a new pirated method on a
            # signature without one of these types still fails. The file goes
            # away with the upstream fix, together with this exception.
            piracies=(
                treat_as_own=[
                    ExaModels.Null,
                    ExaModels.AbstractNode,
                    ExaModels.AbstractAdjointNode,
                    ExaModels.AbstractSecondAdjointNode,
                    ExaModels.SecondAdjointNull,
                ],
            ),
        )

        # Ecosystem pattern (CTModels, CTSolvers, CTDirect): the ambiguity check
        # runs on the package alone, out of test_all. `exclude` skips the stdlib
        # functions extended by the temporary port in src/exa_linalg.jl — the 80+
        # ambiguities they carry are upstream's, ported verbatim, and leave with
        # the file when #325 is resolved. Every other function of the package is
        # still checked.
        Aqua.test_ambiguities(
            CTParser;
            exclude=[
                Base.:*,
                Base.:+,
                Base.:-,
                Base.sum,
                Base.adjoint,
                Base.transpose,
                Base.promote_rule,
                Base.convert,
                Base.zero,
                Base.one,
                Base.inv,
                Base.abs,
                Base.sqrt,
                Base.cbrt,
                Base.abs2,
                Base.exp,
                Base.log,
                Base.sin,
                Base.cos,
                Base.tan,
                LinearAlgebra.dot,
                LinearAlgebra.tr,
                LinearAlgebra.diag,
                LinearAlgebra.diagm,
                LinearAlgebra.det,
                LinearAlgebra.norm,
                LinearAlgebra.cross,
            ],
        )
    end
end
