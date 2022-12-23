using mfRG
using Test

@testset "SIAM parquet KF" begin
    # Test whether run_parquet runs. Does not check the correctness of the result.
    e = 0
    Δ = 10.0
    t = 0.1
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

    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, t)

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
end

@testset "SIAM parquet w/o irr MF" begin
    nmax = 5
    basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
    basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
    basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)

    t = 2.0
    U = 1.0

    # Reference system
    e₀ = 0.0
    Δ₀ = 0.7
    D₀ = 10
    G0₀ = SIAMLazyGreen2P{:MF}(; e=e₀, Δ=Δ₀, t, D=D₀)
    @time Γ₀, Σ₀ = run_parquet(G0₀, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
                        basis_v_aux, basis_1p; max_iter=20, reltol=1e-2, temperature=t)
    G₀ = solve_Dyson(G0₀, Σ₀)
    Π₀ = mfRG.setup_bubble_SU2(G₀, basis_v_bubble, basis_w_bubble; temperature=t, smooth_bubble=false)

    # Target system
    e = 0.1
    Δ = 1.4
    D = 20
    G0 = SIAMLazyGreen2P{:MF}(; e, Δ, t, D)
    @time Γ_exact, Σ_exact = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
                        basis_v_aux, basis_1p; max_iter=20, reltol=1e-2, temperature=t)

    # Run parquet without the fully irreducible vertex
    @time ΔΓ, Σ = run_parquet_without_irreducible(G0, Π₀, Γ₀, basis_1p;
                        max_iter=20, reltol=1e-2, temperature=t);

    # Test vertex
    x = vertex_to_vector(Γ_exact)
    y = vertex_to_vector(Γ₀) .+ vertex_to_vector(ΔΓ)
    @test isapprox(x, y; rtol=1e-4)
    @test !isapprox(x, vertex_to_vector(Γ₀); rtol=1e-4)  # Check ΔΓ term is needed

    # Test self-energy
    @test Σ_exact.data ≈ Σ.data rtol=1e-5
    @test Σ_exact.offset ≈ Σ.offset rtol=1e-5
end

@testset "SIAM parquet linear response" begin
    # Test linear response for parquet calculation
    Δ = 2.0
    t = 0.5
    D = Inf
    U = 1.0
    results = Dict()

    for F in (:MF, :KF)
        if F === :MF
            nmax = 4
            basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)
            basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 4 * nmax)
            basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
            basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
            basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
            basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
        elseif F === :KF
            vgrid_1p = get_nonequidistant_grid(8, 51) .* Δ;
            vgrid_k1 = get_nonequidistant_grid(8, 7) .* Δ;
            wgrid_k1 = get_nonequidistant_grid(8, 7) .* Δ;
            vgrid_k3 = get_nonequidistant_grid(8, 5) .* Δ;

            basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
            basis_w = LinearSplineAndTailBasis(1, 3, wgrid_k1)
            basis_w_k1 = LinearSplineAndTailBasis(1, 3, wgrid_k1)
            basis_v_aux = LinearSplineAndTailBasis(1, 0, vgrid_k3)
            basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
            basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v_bubble_tmp, basis_w)
        end

        function do_parquet(μ)
            # Run parquet calculation with given chemical potential μ.
            G0 = SIAMLazyGreen2P{F}(; e=-μ, Δ, t, D)
            Γ, Σ, Π = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
                basis_v_aux, basis_1p; max_iter=15, reltol=1e-3, temperature=t, mixing_coeff=1.);
            G = solve_Dyson(G0, Σ)
            op_suscep_L, op_suscep_R = susceptibility_operator_SU2(Val(F))
            chi = compute_response_SU2(op_suscep_L, op_suscep_R, Γ, Π.A)
            n = compute_occupation(G, t)
            (; Γ, Σ, Π, G, chi, n)
        end

        # Test μ = 0 gives half filling
        res_half_filling = do_parquet(0.)
        @test res_half_filling.n ≈ 0.5
        @test norm(res_half_filling.Σ.offset) ≈ 0 atol=sqrt(eps(Float64))

        # Test consistency between the interacting charge susceptibility computed from finite
        # differences and linear response.
        μ = 0.5
        δμ = 1e-3
        res = do_parquet(μ)
        chi_lr_vertex = res.chi
        chi_fd = (do_parquet(μ + δμ).n - do_parquet(μ - δμ).n) / 2 / δμ
        if F === :MF
            chi_lr = chi_lr_vertex.total[1](0, 0, 0)[1, 1]
            chi_lr_dis = chi_lr_vertex.disconnected[1](0, 0, 0)[1, 1]
            @test chi_fd ≈ chi_lr rtol=1e-3
            @test !isapprox(chi_lr, chi_lr_dis; rtol=1e-3)
        elseif F === :KF
            chi_lr = chi_lr_vertex.total[1](0, 0, 0)[1, 2]
            chi_lr_dis = chi_lr_vertex.disconnected[1](0, 0, 0)[1, 2]
            @test chi_fd ≈ chi_lr rtol=5e-3
            @test !isapprox(chi_lr, chi_lr_dis; rtol=5e-3)
        end
        results[(F, "n")] = res.n
        results[(F, "fd")] = chi_fd
        results[(F, "lr")] = chi_lr
        results[(F, "lr_dis")] = chi_lr_dis
    end

    # Check MF and KF gives the same results
    @test results[(:MF, "n")] ≈ results[(:KF, "n")] rtol=1e-3
    @test results[(:MF, "fd")] ≈ results[(:KF, "fd")] rtol=5e-3
    @test results[(:MF, "lr")] ≈ results[(:KF, "lr")] rtol=5e-3
    @test results[(:MF, "lr_dis")] ≈ results[(:KF, "lr_dis")] rtol=5e-3
end
