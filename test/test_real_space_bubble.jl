using mfRG
using Test

@testset "compute_bubble_nonlocal" begin
    # Test consistency between the bubble calculated with the real-space method and direct
    # momentum-space integration (convolution).
    nmax = 2
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, nmax)
    basis_v = ImagGridAndTailBasis(:Fermion, 2, 4, nmax)
    nmax_1p = maximum(abs.(get_fitting_points(basis_v))) + maximum(abs.(get_fitting_points(basis_w)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 0, nmax_1p)

    temperature = 0.2
    G0 = mfRG.HubbardLazyGreen2P{:MF}(; temperature, t=1., t2=0.2, μ=0.1)

    lattice = SMatrix{2, 2}([1. 0; 0 1])
    positions = [SVector(0., 0.)]
    bonds_L = [(1, 1, SVector(0, 0))]
    bonds_R = [(1, 1, SVector(0, 0))]
    qgrid = (3, 3)
    nk_1p = 4

    rbasis = RealSpaceBasis(lattice, positions, bonds_L, bonds_R, qgrid)
    rbasis_1p = RealSpaceBasis2P(lattice, positions, (nk_1p, nk_1p))

    G = mfRG.green_lazy_to_explicit(G0, (; freq=basis_1p, r=rbasis_1p));
    b0 = (1, 1, SVector(0, 0))

    for C in (:A, :P, :T)
        Π = mfRG.compute_bubble_nonlocal_real_space(G, basis_v, basis_w, Val(C), rbasis; temperature);

        for xq in rbasis.qpts
            Πq_ref = mfRG.compute_bubble_nonlocal(G, basis_v, basis_w, Val(C), xq, 2 * nk_1p; temperature)
            @test mfRG.interpolate_to_q(Π, xq, b0, b0).data ≈ Πq_ref.data
        end
    end
end


