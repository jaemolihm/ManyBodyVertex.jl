using mfRG
using Test

@testset "SIAM parquet KF" begin
    # Test whether run_parquet runs. Does not check the correctness of the result.
    e = 0
    Δ = 10.0
    temperature = 0.1
    U = 0.5 * Δ

    # Very coarse parameters for debugging
    vgrid_1p = get_nonequidistant_grid(10, 11) .* Δ;
    vgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    wgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    wgrid_k2 = get_nonequidistant_grid(10, 3) .* Δ;
    vgrid_k2 = get_nonequidistant_grid(10, 5) .* Δ;

    basis_w_k1 = LinearSplineAndTailBasis(1, 3, wgrid_k1)
    basis_w_k2 = LinearSplineAndTailBasis(1, 3, wgrid_k2)
    basis_aux = LinearSplineAndTailBasis(1, 0, vgrid_k2)

    basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
    basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
    basis_v_bubble, basis_w_bubble = mfRG.basis_for_bubble(basis_v_bubble_tmp, basis_w_k1)

    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, temperature)

    # Run parquet calculation
    Γ, Σ = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w_k2,
        basis_aux, basis_1p; max_class=3, max_iter=3)
    @test Γ isa mfRG.AsymptoticVertex{:KF, ComplexF64}
    @test Σ isa Green2P{:KF, ComplexF64}
    @test Γ.basis_k1_b === (; freq=basis_w_k1)
    @test Γ.basis_k2_b === (; freq=basis_w_k2)
    @test Γ.basis_k3_b === (; freq=basis_w_k2)
    @test Γ.basis_k2_f === (; freq=basis_aux)
    @test Γ.basis_k3_f === (; freq=basis_aux)

    # Test SBE decomposition
    vs = vgrid_k2
    ws = wgrid_k2
    Γ_sbe = mfRG.asymptotic_to_sbe(Γ)
    @time for ispin in 1:2
        Γ_K123 = (Γ.K1_A[ispin], Γ.K2_A[ispin], Γ.K2p_A[ispin], Γ.K3_A[ispin])
        ∇, M = Γ_sbe.∇_A[ispin], Γ_sbe.M_A[ispin]

        z1 = sum([V(v, vp, w) for v in vs, vp in vs, w in ws] for V in Γ_K123)
        z2 = sum([V(v, vp, w) for v in vs, vp in vs, w in ws] for V in [∇, M])
        @test z1 ≈ z2 rtol=1e-4

        for C in (:A, :P, :T)
            z1 = sum(to_matrix(V, 0.5, basis_aux, basis_aux, C) for V in Γ_K123)
            z2 = sum(to_matrix(V, 0.5, basis_aux, basis_aux, C) for V in [∇, M])
            @test z1 ≈ z2 rtol=2e-2
        end
    end
end

@testset "SIAM parquet w/o irr MF" begin
    nmax = 5
    basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
    basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
    basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)

    temperature = 2.0
    U = 1.0

    # Reference system
    e₀ = 0.0
    Δ₀ = 0.7
    D₀ = 10
    G0₀ = SIAMLazyGreen2P{:MF}(; e=e₀, Δ=Δ₀, temperature, D=D₀)
    @time Γ₀, Σ₀ = run_parquet(G0₀, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
                        basis_v_aux, basis_1p; max_iter=20, reltol=1e-3, temperature)
    G₀ = solve_Dyson(G0₀, Σ₀)
    Π₀ = mfRG.setup_bubble_SU2(G₀, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble=false)

    # Target system
    e = 0.1
    Δ = 1.4
    D = 20
    G0 = SIAMLazyGreen2P{:MF}(; e, Δ, temperature, D)
    @time Γ_exact, Σ_exact = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
                        basis_v_aux, basis_1p; max_iter=20, reltol=1e-3, temperature)

    # Run parquet without the fully irreducible vertex
    @time ΔΓ, Σ = run_parquet_without_irreducible(G0, Π₀, Γ₀, basis_1p;
                        max_iter=15, reltol=1e-3, temperature);

    # Test vertex
    x = vertex_to_vector(Γ_exact)
    y = vertex_to_vector(Γ₀) .+ vertex_to_vector(ΔΓ)
    @test isapprox(x, y; rtol=1e-4)
    @test !isapprox(x, vertex_to_vector(Γ₀); rtol=1e-4)  # Check ΔΓ term is needed

    # Test self-energy
    @test Σ_exact.data ≈ Σ.data rtol=1e-5
    @test Σ_exact.offset ≈ Σ.offset rtol=1e-5

    # Test SBE decomposition
    vs = -nmax:nmax
    ws = -nmax:nmax
    Γ₀_SBE = mfRG.asymptotic_to_sbe(Γ₀)
    @time for ispin in 1:2
        Γ_K123 = (Γ₀.K1_A[ispin], Γ₀.K2_A[ispin], Γ₀.K2p_A[ispin], Γ₀.K3_A[ispin])
        ∇, M = Γ₀_SBE.∇_A[ispin], Γ₀_SBE.M_A[ispin]
        z1 = sum([V(v, vp, w) for v in vs, vp in vs, w in ws] for V in Γ_K123)
        z2 = sum([V(v, vp, w) for v in vs, vp in vs, w in ws] for V in [∇, M])
        @test z1 ≈ z2

        for C in (:A, :P, :T)
            z1 = sum(to_matrix(V, 1, basis_v_aux, basis_v_aux, C) for V in Γ_K123)
            z2 = sum(to_matrix(V, 1, basis_v_aux, basis_v_aux, C) for V in [∇, M])
            @test z1 ≈ z2
        end
    end
end
