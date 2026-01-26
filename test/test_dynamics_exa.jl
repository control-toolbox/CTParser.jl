# test vector-form dynamics for exa backend
# Tests for p_dynamics_exa! which allows defining all state dynamics in one expression: ∂(x)(t) == [e1, e2, ...]

activate_backend(:exa)

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

function test_dynamics_exa()
    l_scheme = [:euler, :euler_implicit, :midpoint, :trapeze]
    for scheme ∈ l_scheme
        __test_dynamics_exa(; scheme=scheme)
        CUDA.functional() && __test_dynamics_exa(CUDABackend(); scheme=scheme)
    end
end

function __test_dynamics_exa(
    backend=nothing; scheme=CTParser.__default_scheme_exa(), tolerance=1e-8, kwargs...
)
    backend_name = isnothing(backend) ? "CPU" : "GPU"

    # ============================================================================
    # Basic vector-form dynamics tests
    # ============================================================================

    test_name = "simple constant dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            ∂(x)(t) == [x₁(t), x₁(t), x₁(t)]
            x₁(0) == 1
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "simple control dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [u₁(t), u₂(t), u₁(t) + u₂(t)]
            x(0) == [0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with sum ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [sum(x(t)), u₁(t), u₂(t)]
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with partial sums ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R³, control
            ∂(x)(t) == [sum(x[2:3](t)), u₁(t), u₂(t)]
            x(0) == [0, 1, 2]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with control sum ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R², control
            ∂(x)(t) == [sum(u(t)), u₁(t)]
            x(0) == [0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with partial control sum ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R³, control
            ∂(x)(t) == [sum(u[1:2](t)), u₃(t)]
            x(0) == [0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with state and control sums ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [sum(x(t)) + sum(u(t)), u₁(t), u₂(t)]
            x(0) == [1, 2, 3]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with partial state and control sums ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R³, control
            ∂(x)(t) == [sum(x[1:2](t)) + sum(u[2:3](t)), u₁(t), u₂(t)]
            x(0) == [0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Tests with user-defined functions
    # ============================================================================

    test_name = "dynamics with user function ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        g(x, u) = x[1] + x[2] - u[1] * u[2]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [sum(x(t)), g(x(t), u(t)), u₂(t)]
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with slice user functions ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        g(x) = x[1] + x[2]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [u₁(t), u₂(t), g(x[1:2](t))]
            x(0) == [0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with multiple user functions ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        f₁(x, u) = x[1] + u[1] * u[2]
        f₂(x) = x[1] + x[2] + x[3]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [sum(x(t)), f₁(x(t), u(t)), f₂(x(t))]
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # High dimension dynamics
    # (Note: Single dimension R¹ states should use coordinate-wise form ∂(x)(t) == expr)
    # ============================================================================

    test_name = "5D dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁵, state
            u ∈ R, control
            ∂(x)(t) == [x₁(t), x₁(t), x₁(t), x₁(t), x₁(t)]
            x(0) == [1, 0, 0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Nonlinear dynamics
    # ============================================================================

    test_name = "nonlinear dynamics with products ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [x₁(t) + x₂(t) + x₃(t), 2x₁(t) * u₁(t) + x₂(t) * u₂(t), x₁(t) + 2x₂(t) - x₃(t)]
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "nonlinear dynamics with squares ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R², control
            ∂(x)(t) == [x₁(t)^2 + x₂(t)^2, u₁(t) * u₂(t), x₂(t) + x₃(t) + x₄(t), u₁(t)]
            x(0) == [1, 0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Double integrator (classic optimal control problem)
    # ============================================================================

    test_name = "double integrator ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ∂(x)(t) == [x₂(t), u(t)]
            x(0) == [1, 0]
            x(1) == [0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "double integrator with Mayer cost ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R, control
            ∂(x)(t) == [x₂(t), u(t), 0.5u(t)^2]
            x(0) == [1, 0, 0]
            x₁(1) == 0
            x₂(1) == 0
            x₃(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Mixed expressions with slices
    # ============================================================================

    test_name = "dynamics with state slices ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        g₁(x) = x[1] + x[2]
        g₂(u) = u[1] * u[2]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁴, state
            u ∈ R², control
            ∂(x)(t) == [g₁(x[1:2](t)), g₂(u(t)), sum(x[2:4](t)), u₁(t)]
            x(0) == [1, 0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Time-dependent (non-autonomous) dynamics
    # ============================================================================

    test_name = "time-dependent dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ∂(x)(t) == [x₂(t), u(t) + sin(t)]
            x(0) == [0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Complex realistic example: drone dynamics (9D)
    # ============================================================================

    test_name = "9D drone dynamics ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        g = 9.81
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R⁹, state
            u ∈ R³, control
            ∂(x)(t) == [
                x₂(t),
                u₁(t) * sin(x₇(t)) * sin(x₉(t)) + u₁(t) * cos(x₇(t)) * sin(x₈(t)) * cos(x₉(t)),
                x₄(t),
                u₁(t) * sin(x₇(t)) * cos(x₉(t)) - u₁(t) * cos(x₇(t)) * sin(x₈(t)) * sin(x₉(t)),
                x₆(t),
                u₁(t) * cos(x₇(t)) * cos(x₈(t)) - g,
                u₂(t) * cos(x₇(t)) / cos(x₈(t)) + u₃(t) * sin(x₇(t)) / cos(x₈(t)),
                -u₂(t) * sin(x₇(t)) + u₃(t) * cos(x₇(t)),
                u₂(t) * cos(x₇(t)) * tan(x₈(t)) + u₃(t) * sin(x₇(t)) * tan(x₈(t))
            ]
            x(0) == [0, 0, 0, 0, 0, 0, 0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # With external parameters
    # ============================================================================

    test_name = "dynamics with external parameters ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        p₁(x, u) = x[1] + x[2] + u[1] * u[2]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [p₁(x[1:2](t), u(t)), sum(u(t)), x₁(t)] # todo: add test with full external call ẋ(t) == f(x(t), u(t)), see tutos with vector fields, etc.
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "dynamics with products of functions and states ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [x₁(t) * u₁(t) + x₂(t) * u₂(t), u₁(t) + u₂(t), x₁(t)]
            x(0) == [1, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Linear systems with matrix notation
    # ============================================================================

    test_name = "linear system with dot product ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        A = [0 1; -1 0]
        B = [0, 1]
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ∂(x)(t) == [dot(A[1, :], x(t)) + u(t) * B[1], dot(A[2, :], x(t)) + u(t) * B[2]] # todo: add full matrix test (see use case no. 8)
            x(0) == [1, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "simple linear system ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R², state
            u ∈ R, control
            ∂(x)(t) == [x₂(t), -x₁(t) + u(t)]
            x(0) == [1, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    # ============================================================================
    # Multiple controls
    # ============================================================================

    test_name = "dynamics with multiple controls ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [u₁(t), u₂(t), sum(u(t))]
            x(0) == [0, 0, 0]
            x₁(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end

    test_name = "double integrator with Mayer cost (2 controls) ($backend_name, $scheme)"
    @testset "$test_name" begin
        println(test_name)
        o = @def begin
            t ∈ [0, 1], time
            x ∈ R³, state
            u ∈ R², control
            ∂(x)(t) == [x₂(t), u₁(t), 0.5(u₁(t)^2 + u₂(t)^2)]
            x(0) == [0, 0, 0]
            x₃(1) => min
        end
        m = discretise_exa(o; backend=backend, scheme=scheme)
        @test m isa ExaModels.ExaModel
    end
end

test_dynamics_exa()
