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
    vertex, Σ = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w_k2,
        basis_aux, basis_1p; max_class=3, max_iter=3)
    @test vertex isa mfRG.AsymptoticVertex{:KF, ComplexF64}
    @test Σ isa Green2P{:KF, ComplexF64}
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
    @time ΔΓ, ΔΣ = run_parquet_without_irreducible(G0, Π₀, Γ₀, basis_1p;
                        max_iter=20, reltol=1e-2, temperature=t);

    # Test Γ_exact ≈ Γ₀ + ΔΓ
    x = vertex_to_vector(Γ_exact)
    y = vertex_to_vector(Γ₀) .+ vertex_to_vector(ΔΓ)
    @test isapprox(x, y; rtol=1e-4)
    @test !isapprox(x, vertex_to_vector(Γ₀); rtol=1e-4)  # Check ΔΓ term is needed
end
