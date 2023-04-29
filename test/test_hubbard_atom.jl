using Test
using mfRG

@testset "Hubbard atom" begin
    U = 1.0
    temperature = 0.3

    nmax = 5
    basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 4 * nmax)
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, 2 * nmax)
    basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
    basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)

    @time Γ = mfRG.hubbard_atom_asymptotic_vertex(U, temperature, basis_w_k1, basis_w, basis_v_aux)

    G_HA = HubbardAtomLazyGreen2P{:MF}(; U, temperature)
    @time ΠA, ΠP = mfRG.setup_bubble_SU2(G_HA, basis_v_bubble, basis_w_bubble; temperature);
    @time Σ_HA = mfRG.compute_self_energy_SU2(Γ, G_HA, ΠA, ΠP, (; freq=basis_1p); temperature)

    # Check SDE is satisfied for exact HA vertex
    G0 = SIAMLazyGreen2P{:MF}(; e=0., Δ=0.0, temperature)
    G_Hubbard = solve_Dyson(G0, Σ_HA)
    vs = -10:9
    @test getindex.(G_Hubbard.(vv), 1, 1) ≈ getindex.(G_HA.(vv), 1, 1) atol=1e-4
end
