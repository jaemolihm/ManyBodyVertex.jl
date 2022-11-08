using mfRG
using Test

@testset "SIAM parquet KF" begin
    # Test whether run_parquet runs. Does not check the correctness of the result.
    using mfRG: siam_get_bubble_improved
    e = 0
    Δ = 10.0
    t = 0.1
    U = 0.5 * Δ

    # Very coarse parameters for debugging
    vgrid_1p = get_nonequidistant_grid(10, 31) .* Δ;
    vgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    wgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    vgrid_k3 = get_nonequidistant_grid(10, 5) .* Δ;

    basis_w = LinearSplineAndTailBasis(1, 3, wgrid_k1)
    basis_aux = LinearSplineAndTailBasis(1, 0, vgrid_k3)

    basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
    basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
    basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v_bubble_tmp, basis_w)
    ΠA_ = siam_get_bubble_improved(basis_v_bubble, basis_w_bubble, basis_1p, Val(:KF), Val(:A); e, Δ, t)
    ΠP_ = siam_get_bubble_improved(basis_v_bubble, basis_w_bubble, basis_1p, Val(:KF), Val(:P); e, Δ, t)
    ΠA = (ΠA_, ΠA_)
    ΠP = (ΠP_, ΠP_ * 2)

    # Run parquet calculation
    vertex = run_parquet(U, ΠA, ΠP, basis_w, basis_aux; max_class=3, max_iter=3)
    @test vertex isa mfRG.AsymptoticVertex{:KF, ComplexF64}
end;
