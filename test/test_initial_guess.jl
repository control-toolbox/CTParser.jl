# test_initial_guess

function test_initial_guess()
    # Problem definitions
    ocp_fixed = @def begin
        t ∈ [0, 1], time
        x = (q, v) ∈ R², state
        u ∈ R, control
        x(0) == [-1, 0]
        x(1) == [0, 0]
        ẋ(t) == [v(t), u(t)]
        ∫(0.5u(t)^2) → min
    end

    ocp_var = @def begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        x = (q, v) ∈ R², state
        u ∈ R, control
        -1 ≤ u(t) ≤ 1
        q(0) == -1
        v(0) == 0
        q(tf) == 0
        v(tf) == 0
        ẋ(t) == [v(t), u(t)]
        tf → min
    end

    ocp_var2 = @def begin
        w = (tf, a) ∈ R², variable
        t ∈ [0, 1], time
        x ∈ R, state
        u ∈ R, control
        ẋ(t) == u(t)
        (tf + a) → min
    end

    @testset "minimal control function on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            u(t) := t
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        ufun = CTModels.control(ig)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "empty and alias-only blocks delegate to defaults" begin
        # Empty block: should behave like a plain call to build_initial_guess(ocp, ())
        ig_empty = @init ocp_fixed begin end
        @test ig_empty isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig_empty)

        # Alias-only block: aliases are executed, but no init specs should still
        # delegate to build_initial_guess(ocp, ()).
        ig_alias_only = @init ocp_fixed begin
            c = 1.0
        end
        @test ig_alias_only isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig_alias_only)
    end

    @testset "simple alias constant on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            a = 1.0
            v(t) := a
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        x0 = xfun(0.0)
        x1 = xfun(1.0)

        @test x0[2] ≈ 1.0
        @test x1[2] ≈ 1.0
    end

    @testset "simple alias for variable on variable-horizon OCP" begin
        ig = @init ocp_var begin
            a = 1.0
            tf := a
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)
    end

    @testset "2D variable block and components" begin
        # Full variable block
        ig_block = @init ocp_var2 begin
            w := [1.0, 2.0]
        end
        @test ig_block isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var2, ig_block)
        v_block = CTModels.variable(ig_block)
        @test length(v_block) == 2
        @test v_block[1] ≈ 1.0
        @test v_block[2] ≈ 2.0

        # Only the tf component
        ig_tf = @init ocp_var2 begin
            tf := 1.0
        end
        @test ig_tf isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var2, ig_tf)
        v_tf = CTModels.variable(ig_tf)
        @test length(v_tf) == 2
        @test v_tf[1] ≈ 1.0
        @test v_tf[2] ≈ 0.1

        # Only the a component
        ig_a = @init ocp_var2 begin
            a := 0.5
        end
        @test ig_a isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var2, ig_a)
        v_a = CTModels.variable(ig_a)
        @test length(v_a) == 2
        @test v_a[1] ≈ 0.1
        @test v_a[2] ≈ 0.5

        # Both components
        ig_both = @init ocp_var2 begin
            tf := 1.0
            a := 0.5
        end
        @test ig_both isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var2, ig_both)
        v_both = CTModels.variable(ig_both)
        @test length(v_both) == 2
        @test v_both[1] ≈ 1.0
        @test v_both[2] ≈ 0.5
    end

    @testset "per-component functions on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            q(t) := sin(t)
            v(t) := 1.0
            u(t) := t
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ sin(0.0)
        @test x1[1] ≈ sin(1.0)
        @test x0[2] ≈ 1.0
        @test x1[2] ≈ 1.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "state block function on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            x(t) := [sin(t), 1.0]
            u(t) := t
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ sin(0.0)
        @test x1[1] ≈ sin(1.0)
        @test x0[2] ≈ 1.0
        @test x1[2] ≈ 1.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "block time-grid init on fixed-horizon OCP" begin
        T = [0.0, 0.5, 1.0]
        X = [[-1.0, 0.0], [0.0, 0.5], [0.0, 0.0]]
        U = [0.0, 0.0, 1.0]

        ig = @init ocp_fixed begin
            x(T) := X
            u(T) := U
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ -1.0
        @test x0[2] ≈ 0.0
        @test x1[1] ≈ 0.0
        @test x1[2] ≈ 0.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "block matrix time-grid init on fixed-horizon OCP" begin
        T = [0.0, 0.5, 1.0]
        Xmat = [
            -1.0 0.0;
            0.0 0.5;
            0.0 0.0
        ]
        U = [0.0, 0.0, 1.0]

        ig = @init ocp_fixed begin
            x(T) := Xmat
            u(T) := U
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ -1.0
        @test x0[2] ≈ 0.0
        @test x1[1] ≈ 0.0
        @test x1[2] ≈ 0.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "block (T, nothing) init on fixed-horizon OCP" begin
        T = [0.0, 0.5, 1.0]

        ig = @init ocp_fixed begin
            x(T) := nothing
            u(T) := nothing
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)
    end

    @testset "component time-grid init on fixed-horizon OCP" begin
        Tq = [0.0, 0.5, 1.0]
        Dq = [-1.0, -0.5, 0.0]
        Tv = [0.0, 1.0]
        Dv = [0.0, 0.0]
        Tu = [0.0, 1.0]
        Du = [0.0, 1.0]

        ig = @init ocp_fixed begin
            q(Tq) := Dq
            v(Tv) := Dv
            u(Tu) := Du
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ -1.0
        @test x1[1] ≈ 0.0
        @test x0[2] ≈ 0.0
        @test x1[2] ≈ 0.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "partial init on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            q(t) := sin(t)
            v(t) := 1.0
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)
    end

    @testset "constant init on fixed-horizon OCP" begin
        ig = @init ocp_fixed begin
            q := -1.0
            v := 0.0
            u := 0.1
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig)
    end

    @testset "variable-only init on variable-horizon OCP" begin
        ig = @init ocp_var begin
            tf := 1.0
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)
    end

    @testset "logging option does not change semantics" begin
        # Reference without logging
        ig_plain = @init ocp_fixed begin
            u(t) := t
        end
        @test ig_plain isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig_plain)

        # Same DSL but with log = true, while redirecting stdout to avoid polluting test logs
        ig_log = Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                u(t) := t
            end log = true
        end
        @test ig_log isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_fixed, ig_log)

        # Compare behaviour at a few sample points
        ufun_plain = CTModels.control(ig_plain)
        ufun_log = CTModels.control(ig_log)
        for τ in (0.0, 0.5, 1.0)
            @test ufun_plain(τ) ≈ ufun_log(τ)
        end
    end

    @testset "per-component functions on variable-horizon OCP" begin
        ig = @init ocp_var begin
            tf := 1.0
            q(t) := sin(t)
            v(t) := 1.0
            u(t) := t
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ sin(0.0)
        @test x1[1] ≈ sin(1.0)
        @test x0[2] ≈ 1.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "(T, nothing) init on variable-horizon OCP" begin
        T = [0.0, 0.5, 1.0]

        ig = @init ocp_var begin
            tf := 1.0
            x(T) := nothing
            u(T) := nothing
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)
    end

    @testset "block time-grid init on variable-horizon OCP" begin
        T = [0.0, 0.5, 1.0]
        X = [[-1.0, 0.0], [0.0, 0.5], [0.0, 0.0]]
        U = [0.0, 0.0, 1.0]

        ig = @init ocp_var begin
            tf := 1.0
            x(T) := X
            u(T) := U
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ -1.0
        @test x0[2] ≈ 0.0
        @test x1[1] ≈ 0.0
        @test x1[2] ≈ 0.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "component time-grid init on variable-horizon OCP" begin
        Tq = [0.0, 0.5, 1.0]
        Dq = [-1.0, -0.5, 0.0]
        Tv = [0.0, 1.0]
        Dv = [0.0, 0.0]
        Tu = [0.0, 1.0]
        Du = [0.0, 1.0]

        ig = @init ocp_var begin
            tf := 1.0
            q(Tq) := Dq
            v(Tv) := Dv
            u(Tu) := Du
        end

        @test ig isa CTModels.AbstractOptimalControlInitialGuess
        CTModels.validate_initial_guess(ocp_var, ig)

        xfun = CTModels.state(ig)
        ufun = CTModels.control(ig)

        x0 = xfun(0.0)
        x1 = xfun(1.0)
        u0 = ufun(0.0)
        u1 = ufun(1.0)

        @test x0[1] ≈ -1.0
        @test x1[1] ≈ 0.0
        @test x0[2] ≈ 0.0
        @test x1[2] ≈ 0.0
        @test u0 ≈ 0.0
        @test u1 ≈ 1.0
    end

    @testset "invalid component vector without time (fixed horizon)" begin
        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                q := [0.0, 1.0]
            end
        end
    end

    @testset "time-grid length mismatch on component (fixed horizon)" begin
        T = [0.0, 0.5, 1.0]
        Dq_bad = [-1.0, 0.0]

        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                q(T) := Dq_bad
            end
        end
    end

    @testset "mixing state block and component (fixed horizon)" begin
        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                x(t) := [sin(t), 1.0]
                q(t) := 0.0
            end
        end
    end

    @testset "unknown component name (fixed horizon)" begin
        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                z(t) := 1.0
            end
        end
    end

    @testset "invalid variable dimension (variable horizon)" begin
        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_var begin
                tf := [1.0, 2.0]
            end
        end
    end

    @testset "time-grid length mismatch on component (variable horizon)" begin
        Tq = [0.0, 0.5, 1.0]
        Dq_bad = [-1.0, 0.0]

        @test_throws CTBase.IncorrectArgument Base.redirect_stdout(Base.devnull) do
            @init ocp_var begin
                tf := 1.0
                q(Tq) := Dq_bad
            end
        end
    end

    @testset "invalid DSL left-hand side" begin
        # Non-symbol lhs in constant form should be rejected at macro level
        @test_throws CTBase.ParsingError Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                (q + v) := 1.0
            end
        end

        # Non-symbol lhs in time-dependent form should also be rejected
        @test_throws CTBase.ParsingError Base.redirect_stdout(Base.devnull) do
            @init ocp_fixed begin
                (q + v)(t) := 1.0
            end
        end
    end

    @testset "init_prefix: getter and setter" begin
        old_pref = CTParser.init_prefix()
        CTParser.init_prefix!(:MyBackend)
        @test CTParser.init_prefix() == :MyBackend
        CTParser.init_prefix!(old_pref)
        @test CTParser.init_prefix() == old_pref
    end
end
