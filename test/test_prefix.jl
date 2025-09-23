# test prefixes

function test_prefix()

    test_name = "defaults"
    @testset "$test_name" begin println(test_name)

        @test CTParser.__default_parsing_backend() == :fun
        @test CTParser.__default_scheme_exa() == :trapeze
        @test CTParser.__default_grid_size_exa() == 250
        @test CTParser.__default_backend_exa() == nothing
        @test CTParser.__default_init_exa() == (0.1, 0.1, 0.1) # default init for v, x, u
        @test CTParser.__default_base_type_exa() == Float64 
        @test CTParser.__default_prefix_fun() == :CTModels
        @test CTParser.__default_prefix_exa() == :ExaModels
        @test CTParser.__default_e_prefix() == :CTBase

    end

    test_name = "prefixes"
    @testset "$test_name" begin println(test_name)

        @test prefix_fun() == CTParser.__default_prefix_fun() 
        prefix_fun!(:foo)
        @test prefix_fun() == :foo 
        prefix_fun!(CTParser.__default_prefix_fun()) # reset to default

        @test prefix_exa() == CTParser.__default_prefix_exa() 
        prefix_exa!(:foo)
        @test prefix_exa() == :foo 
        prefix_exa!(CTParser.__default_prefix_exa()) # reset to default

        @test e_prefix() == CTParser.__default_e_prefix() 
        e_prefix!(:foo)
        @test e_prefix() == :foo 
        e_prefix!(CTParser.__default_e_prefix()) # reset to default

    end

end
