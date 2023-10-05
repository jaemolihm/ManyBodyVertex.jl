using ManyBodyVertex
using Test

@testset "analytic continuation" begin
    # Analytic continuation from KF to MF
    e = 0.5
    Δ = 1.0
    temperature = 0.5
    G0 = SIAMLazyGreen2P{:MF}(; e, Δ, temperature)
    basis_MF = ImagGridAndTailBasis(:Fermion, 1, 3, 20)
    G_MF = ManyBodyVertex.green_lazy_to_explicit(G0, basis_MF)

    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, temperature)
    basis_KF = LinearSplineAndTailBasis(1, 3, get_nonequidistant_grid(6, 151))
    G_KF = ManyBodyVertex.green_lazy_to_explicit(G0, basis_KF)

    vs = -12:11
    G_MF_ac = analytic_continuation_KF_to_MF(G_KF, basis_MF, temperature, :Green)
    @test G_MF_ac.(vs) ≈ G_MF.(vs) rtol=1e-4
end

@testset "analytic continuation SIAM parquet" begin
    # Test analytic continuation from KF to MF of the SIAM parquet result
    e = 0.5
    Δ = 1.0
    temperature = 0.5
    U = 2.0
    function do_parquet(F)
        if F === :MF
            nmax = 4
            basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)
            basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 4 * nmax)
            basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
            basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
            basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
            basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
        elseif F === :KF
            vgrid_1p = get_nonequidistant_grid(8, 51) .* Δ
            vgrid_k1 = get_nonequidistant_grid(8, 11) .* Δ
            wgrid_k1 = get_nonequidistant_grid(8, 11) .* Δ
            vgrid_k3 = get_nonequidistant_grid(8, 11) .* Δ

            basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
            basis_w = LinearSplineAndTailBasis(1, 3, wgrid_k1)
            basis_w_k1 = LinearSplineAndTailBasis(1, 3, wgrid_k1)
            basis_v_aux = LinearSplineAndTailBasis(1, 0, vgrid_k3)
            basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
            basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v_bubble_tmp, basis_w)
        end
        G0 = SIAMLazyGreen2P{F}(; e, Δ, temperature)
        Γ, Σ, Π = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w,
            basis_v_aux, basis_1p; max_iter=30, reltol=1e-3, temperature, mixing_coeff=1.0);
        G = solve_Dyson(G0, Σ)
        op_suscep_L, op_suscep_R = susceptibility_operator_SU2(Val(F))
        chi = compute_response_SU2(op_suscep_L, op_suscep_R, Γ, Π.A, (; freq=basis_v_aux))
        n = compute_occupation(G, temperature)
        (; Γ, Σ, Π, G, chi, n, basis_1p, basis_w)
    end

    res_MF = do_parquet(:MF)
    res_KF = do_parquet(:KF)

    @test res_MF.n ≈ res_KF.n rtol=5e-3

    vs = -12:11
    G_MF_ac = analytic_continuation_KF_to_MF(res_KF.G, res_MF.basis_1p, temperature, :Green)
    Σ_MF_ac = analytic_continuation_KF_to_MF(res_KF.Σ, res_MF.basis_1p, temperature, :Vertex)
    @test G_MF_ac.(vs) ≈ res_MF.G.(vs) rtol=5e-3
    @test Σ_MF_ac.(vs) ≈ res_MF.Σ.(vs) rtol=5e-3
    @test Σ_MF_ac(10000) ≈ res_MF.Σ(10000) rtol=5e-2  # Hartree term has larger error

    ws = -10:10
    basis_w_ac = ImagGridAndTailBasis(:Boson, 1, 3, 8)
    for key in (:total, :disconnected, :connected), i in 1:2
        chi_MF = getproperty(res_MF.chi, key)[i]
        chi_KF = getproperty(res_KF.chi, key)[i]
        chi_MF_ac = analytic_continuation_KF_to_MF(chi_KF, basis_w_ac, temperature, :Green)
        @test chi_MF_ac.(ws) ≈ chi_MF.(ws) rtol=2e-1  # susceptibility has larger error
        @test maximum(abs.(getindex.(chi_MF_ac.(ws) .- chi_MF.(ws), 1, 1))) < 0.05
    end
end
