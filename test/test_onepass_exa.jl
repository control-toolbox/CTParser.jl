# test onepass, exa parsing (aka parsing towards ExaModels)
# tests with @def_exa are @test_throw's that bypass :fun parsing since parsing with :fun throws an error before one can be detected by :exa.

activate_backend(:exa) # nota bene: needs to be executed before @def are expanded

# Mock up of CTDirect.discretise for tests

function discretise_exa(
    ocp;
    scheme=CTParser.__default_scheme_exa(),
    grid_size=CTParser.__default_grid_size_exa(),
    backend=CTParser.__default_backend_exa(),
    init=CTParser.__default_init_exa(),
    base_type=CTParser.__default_base_type_exa(),
)
    build_exa = CTModels.get_build_examodel(ocp)
    return build_exa(;
        scheme=scheme, grid_size=grid_size, backend=backend, init=init, base_type=base_type
    )[1]
end

function discretise_exa_full(
    ocp;
    scheme=CTParser.__default_scheme_exa(),
    grid_size=CTParser.__default_grid_size_exa(),
    backend=CTParser.__default_backend_exa(),
    init=CTParser.__default_init_exa(),
    base_type=CTParser.__default_base_type_exa(),
)
    build_exa = CTModels.get_build_examodel(ocp)
    return build_exa(;
        scheme=scheme, grid_size=grid_size, backend=backend, init=init, base_type=base_type
    )
end

# Tests

function test_onepass_exa()
    #l_scheme = [:euler, :euler_implicit, :midpoint, :trapeze]
    l_scheme = [:midpoint] # debug
    for scheme ∈ l_scheme
        __test_onepass_exa(; scheme=scheme)
        CUDA.functional() && __test_onepass_exa(CUDABackend(); scheme=scheme)
    end
end

function __test_onepass_exa(
    backend=nothing; scheme=CTParser.__default_scheme_exa(), tolerance=1e-8, kwargs...
)
    backend_name = isnothing(backend) ? "CPU" : "GPU"

    @ignore begin # debug
    test_name = "min ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            c = v₁ + b + x₁(0) + 2cos(x₃(1))
            c → min
        end
        m = discretise_exa(o)
        @test NLPModels.get_minimize(m) == true
        @test criterion(o) == :min
    end

    test_name = "max ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            c = v₁ + b + x₁(0) + 2cos(x₃(1))
            c → max
        end
        m = discretise_exa(o)
        @test NLPModels.get_minimize(m) == false
        @test criterion(o) == :max
    end

    test_name = "auxiliary functions ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        @test CTParser.is_range(1) == false
        @test CTParser.is_range(1:2) == true
        @test CTParser.is_range(1:2:5) == true
        @test CTParser.is_range(:(x:y:z)) == true
        @test CTParser.as_range(1) == :((1):(1))
        @test CTParser.as_range(1:2) == 1:2
        @test CTParser.as_range(:x) == :(x:x)
        @test CTParser.as_range(:(x + 1)) == :((x + 1):(x + 1))
    end

    test_name = "bare symbols and ranges - costs ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        # Test: Lagrange with sum over all state components
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            x(0) == [1, 2, 3]
            x(1) == [4, 5, 6]
            ∫(sum(x(t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Lagrange with sum over range of states
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            ∫(sum(x[1:2](t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Lagrange with sum over all controls
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∫(sum(u(t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Lagrange with sum over range of controls
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∫(sum(u[1:2](t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mayer with sum over all states at t0
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(0))^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mayer with sum over all states at tf
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(1))^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mayer with sum over range at t0
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[1:2](0))^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mayer with sum over range at tf
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[2:3](1))^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Bolza cost with bare symbols
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            (sum(x(0))^2 + sum(x(1))^2) + ∫(sum(u(t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Bolza cost with ranges
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            (sum(x[1:2](0)) + sum(x[2:3](1))) + ∫(sum(u[1:2](t))) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "bare symbols and ranges - constraints ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        # Test: Initial constraint with bare symbol
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(0)) == 6
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Initial constraint with range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[1:2](0)) == 3
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Final constraint with bare symbol
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(1)) == 15
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Final constraint with range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[2:3](1)) == 11
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Boundary constraint combining t0 and tf with bare symbols
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(0)) + sum(x(1)) == 21
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Boundary constraint with ranges
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[1:2](0)) - sum(x[2:3](1)) == -8
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Path constraint with bare state symbol
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(t))^2 ≤ 100
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Path constraint with state range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x[1:2](t)) ≤ 10
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Path constraint with bare control symbol
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            sum(u(t))^2 ≤ 5
            ∫(x₁(t)^2 + x₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Path constraint with control range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            sum(u[1:2](t)) ≤ 3
            ∫(x₁(t)^2 + x₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mixed constraint with bare symbols
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            sum(x(t)) + sum(u(t)) ≤ 15
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Mixed constraint with ranges
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R³, control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₃(t)
            sum(x[1:2](t)) + sum(u[2:3](t)) ≤ 8
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "bare symbols and ranges - dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        # Test: Dynamics with sum over all states
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == sum(x(t))
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == u₂(t)
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Dynamics with sum over state range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == sum(x[2:3](t))
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == u₂(t)
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Dynamics with sum over all controls
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == sum(u(t))
            ∂(x₂)(t) == u₁(t)
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Dynamics with sum over control range
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x₁)(t) == sum(u[1:2](t))
            ∂(x₂)(t) == u₃(t)
            ∫(u₁(t)^2 + u₂(t)^2 + u₃(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Dynamics with mixed bare symbols
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == sum(x(t)) + sum(u(t))
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == u₂(t)
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Dynamics with mixed ranges
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R³, control
            ∂(x₁)(t) == sum(x[1:2](t)) + sum(u[2:3](t))
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == u₂(t)
            ∫(u₁(t)^2 + u₂(t)^2 + u₃(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "user-defined functions with ranges ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        # Define user functions outside @def
        f(x, u) = x[1] * x[3] + u[1]^2 * cos(u[2])
        g(x) = x[1] + 2 * x[2]
        h(u) = u[1]^2 + sin(u[2])

        # Test: User-defined function in Lagrange cost
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            ∫(f(x(t), u(t))^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in Mayer cost at t0
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(0), [0, 0])^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in Mayer cost at tf
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(1), [0, 0])^2 → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in Bolza cost
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            (f(x(0), [0, 0]) + f(x(1), [0, 0])) + ∫(h(u(t))) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in initial constraint
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(0), [0, 0]) == 5
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in final constraint
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(1), [0, 0]) == 10
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in boundary constraint
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(0), [0, 0]) + f(x(1), [0, 0]) == 15
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in path constraint
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)
            f(x(t), u(t)) ≤ 10
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: User-defined function in dynamics
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == g(x[1:2](t))
            ∫(u₁(t)^2 + u₂(t)^2) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel

        # Test: Multiple user-defined functions
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x₁)(t) == g(x[1:2](t))
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == u₂(t)
            h(u(t)) ≤ 5
            (f(x(0), [0, 0])) + ∫(f(x(t), u(t))) → min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "pragma ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R⁴, control
            PRAGMA(println("Barracuda sors de ce corps !"))
            PRAGMA(println("grid_size = ", grid_size))
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            c = v₁ + b + x₁(0) + 2cos(x₃(1))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "alias ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            c = v₁ + b + x₁(0) + 2cos(x₃(1))
            c → min
        end
        @test discretise_exa(o; scheme=scheme) isa ExaModels.ExaModel
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
        @test discretise_exa(o; grid_size=100, scheme=scheme) isa ExaModels.ExaModel
        @test discretise_exa(o; init=(1.0, 2.0, 3.0, 4.0, 5.0), scheme=scheme) isa
            ExaModels.ExaModel
        @test discretise_exa(o; base_type=Float32, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(0) + 2cos(x(1))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "time ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v = (a, b) ∈ R², variable
            t ∈ [v, 0], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(a) + 2cos(x(1))
            c → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def_exa begin
            v = (a, b) ∈ R², variable
            t ∈ [0, v], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(a) + 2cos(x(1))
            c → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [a, 1], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(a) + 2cos(x(1))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [0, b], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(0) + 2cos(x(b))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            v = (a, b) ∈ R², variable
            t ∈ [a, b], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = v₁ + b + x(a) + 2cos(x(b))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R, state
            u ∈ R⁴, control
            ∂(x₁)(t) == x₁(t)
            c = tf + x(0) + 2cos(x(tf))
            c → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        t0 = 0.0
        o = @def begin
            t ∈ [t0, 1], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            x₁(t0) + 2cos(x₂(1)) → min
        end

        tf = 1.0
        o = @def begin
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "constraint ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            x₁(0) == 1
            x₂(tf) == 2
            -1 ≤ x₂(0) + x₁(tf) + tf ≤ 1
            tf ≤ 5
            tf^2 ≥ 5
            x₁(t) ≤ 1
            -1 ≤ u(t) ≤ 1
            cos(x₁(t)) ≤ 1
            cos(u(t)) ≤ 1
            x₁(t) + u(t) == 1
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def_exa begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            -1 ≤ x₂(0) + x₁(tf) + tf ≤ [1, 2]
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test_throws String o(; backend=backend)

        o = @def_exa begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            tf^2 ≥ [1, 5] # wrong dim
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test_throws String o(; backend=backend)

        o = @def_exa begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            cos(x₁(t)) ≤ [1, 2] # wrong dim
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test_throws String o(; backend=backend)

        o = @def_exa begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            cos(u(t)) ≤ [1, 2] # wrong dim
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test_throws String o(; backend=backend)

        o = @def_exa begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            x₁(t) + u(t) == [1, 2] # wrong dim
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test_throws String o(; backend=backend)

        o = @def begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R⁴, state
            u ∈ R, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) == 1
            x[1:2:3](0) == [1, 2]
            x[1:3](0) == [1, 2, 3]
            x(0) == [1, 2, 3, 4]
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x ∈ R⁴, state
            u ∈ R, control
            x₁(tf) == 1
            x[1:2:3](tf) == [1, 2]
            x[1:3](tf) == [1, 2, 3]
            x(tf) == [1, 2, 3, 4]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) + 2cos(x₂(tf)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            v ∈ R⁴, variable
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R, control
            v₁ == 1
            v[1:2:3] == [1, 2]
            v[1:3] == [1, 2, 3]
            v == [1, 2, 3, 4]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R, control
            x₁(t) == 1
            x[1:2:3](t) == [1, 2]
            x[1:3](t) == [1, 2, 3]
            x(t) == [1, 2, 3, 4]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            u₁(t) == 1
            u[2:2:4](t) == [1, 2]
            u[2:4](t) == [1, 2, 3]
            u(t) == [1, 2, 3, 4, 5]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel

        o = @def begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            v ≤ [1, 2, 3]
            v ≥ [1, 2, 3]
            v[1] ≤ 1
            v[1] ≥ 1 
            v[1:2] ≤ [1, 2]
            v[1:2] ≥ [1, 2]
            x[2](t) ≤ 1
            x[2:2:4](t) ≤ [1, 2]
            x[2:4](t) ≤ [1, 2, 3]
            x[2](t) ≥ 1
            x[2:2:4](t) ≥ [1, 2]
            x[2:4](t) ≥ [1, 2, 3]
            u[2](t) ≤ 1
            u[2:2:4](t) ≤ [1, 2]
            u[2:4](t) ≤ [1, 2, 3]
            u[2](t) ≥ 1
            u[2:2:4](t) ≥ [1, 2]
            u[2:4](t) ≥ [1, 2, 3]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "variable range ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v ∈ R⁵, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R⁴, control
            0 ≤ v[1] ≤ 1
            [0, 0] ≤ v[2:3] ≤ [1, 1]
            [0, 0] ≤ v[2:2:5] ≤ [1, 1]
            [0, 0, 0, 0, 0] ≤ v ≤ [1, 1, 1, 1, 1]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            x₁ → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "state range ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R⁵, state
            u ∈ R⁴, control
            0 ≤ x[1](t) ≤ 1
            [0, 0] ≤ x[2:3](t) ≤ [1, 1]
            [0, 0] ≤ x[2:2:5](t) ≤ [1, 1]
            [0, 0, 0, 0, 0] ≤ x(t) ≤ [1, 1, 1, 1, 1]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            ∂(x₅)(t) == x₁(t)
            x₁ → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "control range ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v ∈ R², variable
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            0 ≤ u[1](t) ≤ 1
            [0, 0] ≤ u[2:3](t) ≤ [1, 1]
            [0, 0] ≤ u[2:2:5](t) ≤ [1, 1]
            [0, 0, 0, 0, 0] ≤ u(t) ≤ [1, 1, 1, 1, 1]
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            ∂(x₄)(t) == x₁(t)
            x₁ → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            ẋ(t) == u[1:4](t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₂)(t) == x₁(t)
            ∂(x₃)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            ∂(x₁)(t) == x₁(t)
            ∂(x₁)(t) == x₁(t) # duplicate!
            ∂(x₃)(t) == x₁(t)
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            ∂(x[1:5])(t) == x₁(t) # not an integer
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        i = 1
        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R⁵, control
            ∂(x[i])(t) == x₁(t) # not an integer
            x₁(0) + 2cos(x₂(1)) → min
        end
        @test_throws ParsingError o(; backend=backend)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R, state
            u ∈ R⁵, control
            x(t) + u₂(t) + t == 1
            ẋ(t) == t + u₁(t)
            x(0) + 2cos(x(1)) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
        @test_throws String discretise_exa(o; scheme=:foo)
    end

    test_name = "lagrange cost ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            x(0) == [-1, 0]
            x(1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∫(0.5u(t)^2) → min
        end
        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
    end

    test_name = "scalar bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0)^2 == [-1, 0, 0] # should be scalar
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "variable bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            v[1:2] ≤ 3 # wrong dim
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "state bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](t) == [0, 0, 0] # wrong dim
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "control bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            u(t) ≤ [1, 2] # wrong dim 
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "path bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            3 ≤ x[1] + u(t) ≤ [1, 2] # wrong dim 
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "path bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            [3, 4] ≤ x[1] + u(t) ≤ [1, 2] # wrong dim 
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "initial bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0] # wrong dim
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "final bounds test"
    @testset "$test_name" begin
        println(test_name)

        o = @def_exa begin
            v ∈ R³, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(1) == [-1, 0] # wrong dim
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        @test_throws String o(; backend=backend)
    end

    test_name = "use case no. 1: simple example (mayer) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end

        @test discretise_exa(o; backend=backend, scheme=scheme) isa ExaModels.ExaModel
        m, _ = discretise_exa_full(o; backend=backend, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 6 atol = 1e-2
        N = 1000
        m, _ = discretise_exa_full(o; backend=backend, grid_size=N, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 6 atol = 1e-3
    end

    test_name = "use case no. 1: simple example (mayer), testing getters (1/2) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) → min
        end
        N = 1000
        m, getter = discretise_exa_full(o; backend=backend, grid_size=N, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test size(getter(s; val=:state)) == (3, N + 1)
        @test size(getter(s; val=:control)) == (1, N + 1)
        @test size(getter(s; val=:variable)) == (0,)
        @test size(getter(s; val=:costate)) == (3, N)
        @test size(getter(s; val=:state_l)) == (3, N + 1)
        @test size(getter(s; val=:state_u)) == (3, N + 1)
        @test size(getter(s; val=:control_l)) == (1, N + 1)
        @test size(getter(s; val=:control_u)) == (1, N + 1)
        @test size(getter(s; val=:variable_l)) == (0,)
        @test size(getter(s; val=:variable_u)) == (0,)
        @test_throws String getter(s; val=:foo)
    end

    test_name = "use case no. 1: simple example (mayer), testing getters (2/2) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            v ∈ R⁴, variable
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            v == [1, 2, 3, 4]
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u₁(t)
            ∂(x₃)(t) == 0.5(u₁(t)^2 + u₂(t)^2)
            x₃(1) → min
        end
        N = 1000
        m, getter = discretise_exa_full(o; backend=backend, grid_size=N, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test size(getter(s; val=:state)) == (3, N + 1)
        @test size(getter(s; val=:control)) == (2, N + 1)
        @test size(getter(s; val=:variable)) == (4,)
        @test size(getter(s; val=:costate)) == (3, N)
        @test size(getter(s; val=:state_l)) == (3, N + 1)
        @test size(getter(s; val=:state_u)) == (3, N + 1)
        @test size(getter(s; val=:control_l)) == (2, N + 1)
        @test size(getter(s; val=:control_u)) == (2, N + 1)
        @test size(getter(s; val=:variable_l)) == (4,)
        @test size(getter(s; val=:variable_u)) == (4,)
    end

    test_name = "use case no. 1: simple example (lagrange) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            x(0) == [-1, 0]
            x(1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∫(0.5u(t)^2) → min
        end

        m, _ = discretise_exa_full(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 6 atol = 1e-2
        m, _ = discretise_exa_full(o; backend=backend, grid_size=1000, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 6 atol = 1e-3
    end

    test_name = "use case no. 1: simple example (bolza) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            x(0) == [-1, 0, 0]
            x[1:2](1) == [0, 0]
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == u(t)
            ∂(x₃)(t) == 0.5u(t)^2
            x₃(1) + ∫(0.5u(t)^2) → min
        end

        m, _ = discretise_exa_full(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 2 * 6 atol = 1e-2
        m, _ = discretise_exa_full(o; backend=backend, grid_size=1000, scheme=scheme)
        s = madnlp(m; tol=tolerance, kwargs...)
        @test s.objective ≈ 2 * 6 atol = 1e-3
    end

    test_name = "use case no. 2: Goddard ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        r0 = 1.0
        v0 = 0.0
        m0 = 1.0
        vmax = 0.1
        mf = 0.6
        Cd = 310.0
        Tmax = 3.5
        β = 500.0
        b = 2.0

        o = @def begin
            tf ∈ R, variable
            t ∈ [0, tf], time
            x = (r, v, m) ∈ R³, state
            u ∈ R, control

            x(0) == [r0, v0, m0]
            m(tf) == mf
            0 ≤ u(t) ≤ 1
            r(t) ≥ r0
            0 ≤ v(t) ≤ vmax

            ∂(r)(t) == v(t)
            ∂(v)(t) ==
            -Cd * v(t)^2 * exp(-β * (r(t) - 1)) / m(t) - 1 / r(t)^2 + u(t) * Tmax / m(t)
            ∂(m)(t) == -b * Tmax * u(t)

            -r(tf) → min # todo: also add max when issue with GPU solved
        end

        tfs = 0.18761155665063417
        xs = [
            1.0 1.00105 1.00398 1.00751 1.01009 1.01124
            -1.83989e-40 0.056163 0.1 0.0880311 0.0492518 0.0123601
            1.0 0.811509 0.650867 0.6 0.6 0.6
        ]
        us = [0.599377 0.835887 0.387328 -5.87733e-9 -9.03538e-9 -8.62101e-9]
        N0 = length(us) - 1
        t = (tfs * 0):N0
        _xs = linear_interpolation(
            t, [xs[:, j] for j in 1:(N0 + 1)], extrapolation_bc=Line()
        )
        _us = linear_interpolation(
            t, [us[:, j] for j in 1:(N0 + 1)], extrapolation_bc=Line()
        )

        N = 200
        t = (tfs * 0):N
        xs = _xs.(t);
        xs = stack(xs[:])
        us = _us.(t);
        us = stack(us[:])

        m, _ = discretise_exa_full(
            o; backend=backend, grid_size=N, init=(tfs, xs, us), scheme=scheme
        )
        s = madnlp(m; tol=tolerance, kwargs...)
        __atol = scheme ∈ (:euler, :euler_implicit) ? 1e-3 : 1e-5
        @test s.objective ≈ -1.0125736217178989e+00 atol = __atol
    end

    test_name = "use case no. 3: quadrotor ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        T = 1
        g = 9.8
        r = 0.1

        o = @def begin
            t ∈ [0, T], time
            x ∈ R⁹, state
            u ∈ R⁴, control

            x(0) == zeros(9)

            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) ==
            u₁(t) * cos(x₇(t)) * sin(x₈(t)) * cos(x₉(t)) + u₁(t) * sin(x₇(t)) * sin(x₉(t))
            ∂(x₃)(t) == x₄(t)
            ∂(x₄)(t) ==
            u₁(t) * cos(x₇(t)) * sin(x₈(t)) * sin(x₉(t)) - u₁(t) * sin(x₇(t)) * cos(x₉(t))
            ∂(x₅)(t) == x₆(t)
            ∂(x₆)(t) == u₁(t) * cos(x₇(t)) * cos(x₈(t)) - g
            ∂(x₇)(t) == u₂(t) * cos(x₇(t)) / cos(x₈(t)) + u₃(t) * sin(x₇(t)) / cos(x₈(t))
            ∂(x₈)(t) == -u₂(t) * sin(x₇(t)) + u₃(t) * cos(x₇(t))
            ∂(x₉)(t) ==
            u₂(t) * cos(x₇(t)) * tan(x₈(t)) + u₃(t) * sin(x₇(t)) * tan(x₈(t)) + u₄(t)

            dt1 = sin(2π * t / T)
            df1 = 0
            dt3 = 2sin(4π * t / T)
            df3 = 0
            dt5 = 2t / T
            df5 = 2

            0.5∫(
                (x₁(t) - dt1)^2 +
                (x₃(t) - dt3)^2 +
                (x₅(t) - dt5)^2 +
                x₇(t)^2 +
                x₈(t)^2 +
                x₉(t)^2 +
                r * (u₁(t)^2 + u₂(t)^2 + u₃(t)^2 + u₄(t)^2),
            ) → min
        end

        N = 100
        m, _ = discretise_exa_full(o; grid_size=N, backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
        sol = madnlp(m; tol=tolerance, kwargs...)
        @test sol.status == MadNLP.SOLVE_SUCCEEDED
    end

    test_name = "use case no. 4: vectorised ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        f₁(x, u) = 2x[1] * u[1] + x[2] * u[2]
        f₂(x) = x[1] + 2x[2] - x[3]
        f₃(x0, xf) = x0[2]^2 + sum(xf)^2
        f₄(u) = sum(u.^2)
        
        o1 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            x[1:2:3](0) == [1, 3]
 
            ∂(x₁)(t) == sum(x(t))
            ∂(x₂)(t) == f₁(x(t), u(t))
            ∂(x₃)(t) == f₂(x(t)) 

            f₃(x(0), x(1)) + 0.5∫( f₄(u(t)) ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        o2 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            x[1:2:3](0) == [1, 3]
 
            ∂(x₁)(t) == x₁(t) + x₂(t) + x₃(t)
            ∂(x₂)(t) == 2x₁(t) * u₁(t) + x₂(t) * u₂(t)
            ∂(x₃)(t) == x₁(t) + 2x₂(t) - x₃(t)

            (x₂(0)^2 + (x₁(1) + x₂(1) + x₃(1))^2) + 0.5∫( u₁(t)^2 + u₂(t)^2 ) → min
        end

        m2, _ = discretise_exa_full(o2; grid_size=N, backend=backend, scheme=scheme)
        @test m2 isa ExaModels.ExaModel
        sol2 = madnlp(m2; tol=tolerance, max_iter=max_iter, kwargs...)
        obj2 = sol2.objective

        __atol = 1e-9
        @test obj1 - obj2 ≈ 0 atol = __atol
    end

    test_name = "use case no. 5: vectorised with ranges ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        g₁(x) = x[1]^2 + x[2]^2
        g₂(u) = u[1] * u[2]

        # Vectorised version using ranges
        o1 = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R², control

            x(0) == [0, .1, .2, .3]

            ∂(x₁)(t) == g₁(x[1:2](t))
            ∂(x₂)(t) == g₂(u(t))
            ∂(x₃)(t) == sum(x[2:4](t))
            ∂(x₄)(t) == u₁(t)

            sum(x[1:3](1))^2 + 0.5∫( sum(u(t).^2) ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        # Non-vectorised version using subscripts
        o2 = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R², control

            x(0) == [0, .1, .2, .3]

            ∂(x₁)(t) == x₁(t)^2 + x₂(t)^2
            ∂(x₂)(t) == u₁(t) * u₂(t)
            ∂(x₃)(t) == x₂(t) + x₃(t) + x₄(t)
            ∂(x₄)(t) == u₁(t)

            (x₁(1) + x₂(1) + x₃(1))^2 + 0.5∫( u₁(t)^2 + u₂(t)^2 ) → min
        end

        m2, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m2 isa ExaModels.ExaModel
        sol2 = madnlp(m2; tol=tolerance, max_iter=max_iter, kwargs...)
        obj2 = sol2.objective

        __atol = 1e-9
        @test obj1 - obj2 ≈ 0 atol = __atol
    end

    test_name = "use case no. 6: vectorised constraints ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        h₁(x) = x[1] + 2x[2] + 3x[3]
        h₂(u) = u[1]^2 + u[2]^2

        # Vectorised version
        o1 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            sum(x(0).^2) == 1.5 
            h₁(x(1)) ≤ 200

            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == sum(u(t))

            h₂(u(t)) ≤ 10

            sum(x(1))^2 + ∫( h₂(u(t)) ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        # Non-vectorised version
        o2 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            x₁(0)^2 + x₂(0)^2 + x₃(0)^2 == 1.5
            x₁(1) + 2x₂(1) + 3x₃(1) ≤ 200

            ∂(x₁)(t) == u₁(t)
            ∂(x₂)(t) == u₂(t)
            ∂(x₃)(t) == u₁(t) + u₂(t)

            u₁(t)^2 + u₂(t)^2 ≤ 10

            (x₁(1) + x₂(1) + x₃(1))^2 + ∫( u₁(t)^2 + u₂(t)^2 ) → min
        end

        m2, _ = discretise_exa_full(o2; grid_size=N, backend=backend, scheme=scheme)
        @test m2 isa ExaModels.ExaModel
        sol2 = madnlp(m2; tol=tolerance, max_iter=max_iter, kwargs...)
        obj2 = sol2.objective

        __atol = 1e-9
        @test obj1 - obj2 ≈ 0 atol = __atol
    end 

    # todo: test below inactived on GPU because run is unstable
    if isnothing(backend) test_name = "use case no. 7: mixed vectorisation ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        # User-defined functions
        p₁(x, u) = x[1] * u[1] + x[2] * u[2]
        p₂(x) = x[1]^2 + x[2]^2 + x[3]^2

        # Vectorised version with mixed patterns
        o1 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            x[1:2](0) == [0, 0.1]
            -0.1 ≤ x₃(0) ≤ 0.1
            sum(x(1)) == 0.2

            ∂(x₁)(t) == p₁(x[1:2](t), u(t))
            ∂(x₂)(t) == sum(u(t))
            ∂(x₃)(t) == x₁(t)

            p₂(x(t)) ≤ 50

            (p₂(x(0)) + sum(x[1:2](1))^2) + 0.5∫( sum(u(t).^2) ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        # Non-vectorised version
        o2 = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control

            x₁(0) == 0
            x₂(0) == 0.1
            -0.1 ≤ x₃(0) ≤ 0.1
            x₁(1) + x₂(1) + x₃(1) == 0.2 

            ∂(x₁)(t) == x₁(t) * u₁(t) + x₂(t) * u₂(t)
            ∂(x₂)(t) == u₁(t) + u₂(t)
            ∂(x₃)(t) == x₁(t)

            x₁(t)^2 + x₂(t)^2 + x₃(t)^2 ≤ 50

            (x₁(0)^2 + x₂(0)^2 + x₃(0)^2 + (x₁(1) + x₂(1))^2) + 0.5∫( u₁(t)^2 + u₂(t)^2 ) → min
        end

        m2, _ = discretise_exa_full(o2; grid_size=N, backend=backend, scheme=scheme)
        @test m2 isa ExaModels.ExaModel
        sol2 = madnlp(m2; tol=tolerance, max_iter=max_iter, kwargs...)
        obj2 = sol2.objective

        __atol = 1e-9
        @test obj1 - obj2 ≈ 0 atol = __atol
    end end
    end # debug

    test_name = "use case no. 8: vectorised dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)

        tf = 5
        x0 = [0, 1]
        A = [0 1; -1 0]
        B = [0, 1]
        Q = [1 0; 0 1]
        R = 1

        # Vectorised version
        o1 = @def begin
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            x(0) == x0
            ∂(x₁)(t) == dot(A[1, :], x(t)) + u(t) * B[1]
            #∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == dot(A[2, :], x(t)) + u(t) * B[2]
            #∂(x₂)(t) == -x₁(t) + u(t) 
            0.5∫( x(t)' * Q * x(t) + u(t)' * R * u(t) ) → min
            #0.5∫( x₁(t)^2 + x₂(t)^2 + u(t)^2 ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        # Non-vectorised version
        o2 = @def begin
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            x(0) == x0
            ∂(x₁)(t) == x₂(t)
            ∂(x₂)(t) == -x₁(t) + u(t) 
            0.5∫( x₁(t)^2 + x₂(t)^2 + u(t)^2 ) → min
        end
        
        m2, _ = discretise_exa_full(o2; grid_size=N, backend=backend, scheme=scheme)
        @test m2 isa ExaModels.ExaModel
        sol2 = madnlp(m2; tol=tolerance, max_iter=max_iter, kwargs...)
        obj2 = sol2.objective

        __atol = 1e-9
        @test obj1 - obj2 ≈ 0 atol = __atol
    end 
end
