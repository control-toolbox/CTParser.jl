# test_prefix_bis

function test_prefix_bis()
    @testset "backend activation (happy path)" begin
        println("backend activation (bis)")

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
        println("backend activation errors (bis)")

        # Unknown backend should throw an error (String from `throw("...")`)
        @test_throws String activate_backend(:unknown_backend)
        @test_throws String deactivate_backend(:unknown_backend)

        # :fun is always active, so trying to (de)activate it should fail
        @test_throws String activate_backend(:fun)
        @test_throws String deactivate_backend(:fun)
    end

    @testset "backend activation (twisted sequences)" begin
        println("backend activation twisted (bis)")

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
        @test_throws String is_active_backend(:unknown_backend)

        # Sanity check: :fun is never affected by toggling :exa
        @test is_active_backend(:fun)

        # Final reset for subsequent tests: keep :exa active by convention
        activate_backend(:exa)
        @test is_active_backend(:exa)
    end

    @testset "prefix setters (fun, exa, e_prefix)" begin
        println("prefix setters (bis)")

        # Save original values
        orig_fun = prefix_fun()
        orig_exa = prefix_exa()
        orig_e = e_prefix()

        # prefix_fun!
        @test prefix_fun() == CTParser.__default_prefix_fun()
        prefix_fun!(:TestModule)
        @test prefix_fun() == :TestModule
        prefix_fun!(:AnotherModule)
        @test prefix_fun() == :AnotherModule
        prefix_fun!(orig_fun)  # restore
        @test prefix_fun() == orig_fun

        # prefix_exa!
        @test prefix_exa() == CTParser.__default_prefix_exa()
        prefix_exa!(:MyExaModule)
        @test prefix_exa() == :MyExaModule
        prefix_exa!(orig_exa)  # restore
        @test prefix_exa() == orig_exa

        # e_prefix!
        @test e_prefix() == CTParser.__default_e_prefix()
        e_prefix!(:CustomExceptions)
        @test e_prefix() == :CustomExceptions
        e_prefix!(orig_e)  # restore
        @test e_prefix() == orig_e
    end

    @testset "prefix setters (pathological values)" begin
        println("prefix setters pathological (bis)")

        # Save original values
        orig_fun = prefix_fun()
        orig_exa = prefix_exa()
        orig_e = e_prefix()

        # Setting to unusual but valid symbols
        prefix_fun!(:_)
        @test prefix_fun() == :_
        prefix_fun!(Symbol("with-dash"))
        @test prefix_fun() == Symbol("with-dash")
        prefix_fun!(Symbol("with.dot"))
        @test prefix_fun() == Symbol("with.dot")

        # Restore
        prefix_fun!(orig_fun)
        prefix_exa!(orig_exa)
        e_prefix!(orig_e)

        # Verify restoration
        @test prefix_fun() == orig_fun
        @test prefix_exa() == orig_exa
        @test e_prefix() == orig_e
    end

    @testset "default values consistency" begin
        println("default values (bis)")

        # Check all default values from defaults.jl
        @test CTParser.__default_parsing_backend() == :fun
        @test CTParser.__default_scheme_exa() == :midpoint
        @test CTParser.__default_grid_size_exa() == 250
        @test CTParser.__default_backend_exa() === nothing
        @test CTParser.__default_init_exa() == (0.1, 0.1, 0.1)
        @test CTParser.__default_base_type_exa() == Float64
        @test CTParser.__default_prefix_fun() == :CTModels
        @test CTParser.__default_prefix_exa() == :ExaModels
        @test CTParser.__default_e_prefix() == :CTBase
    end
end
