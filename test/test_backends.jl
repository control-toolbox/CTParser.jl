# test_backends
# Backend activation and parser defaults. The prefix machinery this file used to
# exercise (PREFIX_FUN / PREFIX_EXA / E_PREFIX / INIT_PREFIX and their setters)
# was removed in #356: the generated code now interpolates the FUN / EXA / ERR
# module objects directly, so there is nothing left to configure.

function test_backends()
    @testset "backend activation (happy path)" begin
        println("backend activation")

        # By design, :fun should always be active
        @test is_active_backend(:fun)

        # Assume :exa is available in PARSING_BACKENDS and can be toggled
        # First ensure it is deactivated, then activate and deactivate again
        deactivate_backend(:exa)
        @test !is_active_backend(:exa)

        activate_backend(:exa)
        @test is_active_backend(:exa)

        deactivate_backend(:exa)
        @test !is_active_backend(:exa)

        # Reset to a clean state for other tests
        activate_backend(:exa)
    end

    @testset "backend activation (errors)" begin
        println("backend activation errors")

        # Unknown backend should throw an error (String from `throw("...")`)
        @test_throws CTBase.IncorrectArgument activate_backend(:unknown_backend)
        @test_throws CTBase.IncorrectArgument deactivate_backend(:unknown_backend)

        # :fun is always active, so trying to (de)activate it should fail
        @test_throws CTBase.PreconditionError activate_backend(:fun)
        @test_throws CTBase.PreconditionError deactivate_backend(:fun)
    end

    @testset "backend activation (twisted sequences)" begin
        println("backend activation twisted")

        # Start from a clean state where :exa is active (default)
        activate_backend(:exa)
        @test is_active_backend(:fun)
        @test is_active_backend(:exa)

        # Multiple deactivations should be idempotent (no error, still false)
        deactivate_backend(:exa)
        @test !is_active_backend(:exa)
        deactivate_backend(:exa)
        @test !is_active_backend(:exa)

        # Multiple activations should be idempotent (no error, still true)
        activate_backend(:exa)
        @test is_active_backend(:exa)
        activate_backend(:exa)
        @test is_active_backend(:exa)

        # is_active_backend on unknown backend should throw
        @test_throws CTBase.IncorrectArgument is_active_backend(:unknown_backend)

        # Sanity check: :fun is never affected by toggling :exa
        @test is_active_backend(:fun)

        # Final reset for subsequent tests: keep :exa active by convention
        activate_backend(:exa)
        @test is_active_backend(:exa)
    end

    @testset "generated-code modules are the packages themselves" begin
        println("FUN / EXA / ERR contract")

        # The expressions emitted by @def and @init interpolate these objects:
        # a module value resolves to a GlobalRef at macro-expansion time, which
        # is what makes the generated code independent of the caller's bindings
        # (#356). Guard the identity so no name-based prefix can come back.
        @test CTParser.FUN === CTModels
        @test CTParser.EXA === ExaModels
        @test CTParser.ERR === CTBase

        # Regression guard: the removed prefix API must not reappear
        for sym in (
            :prefix_fun,
            :prefix_fun!,
            :prefix_exa,
            :prefix_exa!,
            :e_prefix,
            :e_prefix!,
            :init_prefix,
            :init_prefix!,
            :PREFIX_FUN,
            :PREFIX_EXA,
            :E_PREFIX,
            :INIT_PREFIX,
            :__default_prefix_fun,
            :__default_prefix_exa,
            :__default_e_prefix,
            :__default_init_prefix,
        )
            @test !isdefined(CTParser, sym)
        end
    end

    @testset "default values consistency" begin
        println("default values")

        # Check all default values from defaults.jl
        @test CTParser.__default_parsing_backend() == :fun
        @test CTParser.__default_scheme_exa() == :midpoint
        @test CTParser.__default_grid_size_exa() == 250
        @test CTParser.__default_backend_exa() === nothing
        @test CTParser.__default_init_exa() == (0.1, 0.1, 0.1)
        @test CTParser.__default_base_type_exa() == Float64
    end
end
