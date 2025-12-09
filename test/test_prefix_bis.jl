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
end
