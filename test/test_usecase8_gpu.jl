# Temporary test file for use case no. 8 with GPU backend only
# Testing inlined expressions (not using dot) to verify GPU error is in MadNLP

activate_backend(:exa)

# Mock up of CTDirect.discretise for tests
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

function test_usecase8_gpu()
    if !CUDA.functional()
        @test_skip "CUDA not functional, skipping GPU test"
        return
    end

    backend = CUDABackend()
    backend_name = "GPU"
    scheme = :midpoint
    tolerance = 1e-8
    kwargs = ()

    test_name = "use case no. 8: vectorised dynamics ($backend_name, $scheme) - INLINED EXPRESSIONS"
    @testset "$test_name" begin
        println(test_name)

        tf = 5
        x0 = [0, 1]
        A = [0 1; -1 0]
        B = [0, 1]
        Q = [1 0; 0 1]
        R = 1

        # Vectorised version with INLINED expressions (not using dot)
        # Original dot expressions are commented out to show the equivalence
        o1 = @def begin
            t ∈ [0, tf], time
            x ∈ R², state
            u ∈ R, control
            x(0) == x0
            #∂(x₁)(t) == dot(A[1, :], x(t)) + u(t) * B[1]  # = 0*x₁(t) + 1*x₂(t) = x₂(t)
            ∂(x₁)(t) == x₂(t)
            #∂(x₂)(t) == dot(A[2, :], x(t)) + u(t) * B[2]  # = -1*x₁(t) + 0*x₂(t) + u(t)*1
            ∂(x₂)(t) == -x₁(t) + u(t)
            #0.5∫( x(t)' * Q * x(t) + u(t)' * R * u(t) ) → min
            0.5∫( x₁(t)^2 + x₂(t)^2 + u(t)^2 ) → min
        end

        N = 250
        max_iter = 10
        m1, _ = discretise_exa_full(o1; grid_size=N, backend=backend, scheme=scheme)
        @test m1 isa ExaModels.ExaModel

        # This is where the GPU error should occur (in MadNLP, not in our code)
        sol1 = madnlp(m1; tol=tolerance, max_iter=max_iter, kwargs...)
        obj1 = sol1.objective

        # Non-vectorised version (for comparison)
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
