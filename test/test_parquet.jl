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
