using Test
using mfRG

# TODO: Add test with finite D

@testset "model SIAM" begin
    using StaticArrays
    using mfRG: siam_get_green_function
    e = 0.5
    Δ = 1.0
    t = 0.1
    v = 10.0
    @test siam_get_green_function(v, Val(:MF); e, Δ, t) isa Number
    @test siam_get_green_function(v, Val(:KF); e, Δ, t) isa SMatrix{2,2}
    for F in (:MF, :KF)
        @inferred siam_get_green_function(v, Val(F); e, Δ, t)
    end

    # Test SIAMLazyGreen2P
    for F in (:MF, :KF)
        D = F === :MF ? 5.0 : Inf
        G0 = SIAMLazyGreen2P{F}(; e, Δ, t, D)
        @test G0 isa mfRG.AbstractFrequencyVertex{F, ComplexF64}
        v = rand()
        @test all(G0(v) .≈ siam_get_green_function(v, Val(F); e, Δ, t, D))
        @inferred G0(v)
    end

    # Test SIAM bubble
    basis_f = LinearSplineAndTailBasis(2, 4, -3:0.6:3)
    basis_b = LinearSplineAndTailBasis(1, 3, -2:0.5:2)
    for F in (:MF, :KF), C in (:A, :P, :T)
        G0 = SIAMLazyGreen2P{F}(; e, Δ, t)
        @test compute_bubble(G0, basis_f, basis_b, Val(C); temperature=t) isa Bubble{F, C, ComplexF64}
    end

end

@testset "compute_bubble_smoothed" begin
    # With compute_bubble_smoothed, sanity checks for the bubble is satisfied to high
    # accuracy even for a coarse basis_f.
    e = 0.5
    Δ = 1.5
    t = 0.1
    basis_f = LinearSplineAndTailBasis(2, 4, range(-10, 10, step=5.0))
    basis_b = LinearSplineAndTailBasis(1, 0, -3.:3:3)

    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, t)
    for c in (:A, :P, :T)
        Π = compute_bubble_smoothed(G0, basis_f, basis_b, Val(c); temperature=t)
        for w in basis_b.grid
            Π_w = reshape(to_matrix(Π, w, ConstantBasis(), ConstantBasis()), 2, 2, 2, 2)

            # ∫ dv Gᴿ(v) Gᴿ(v + w) = 0 due to analyticity
            @test 0 < abs(Π_w[1, 2, 1, 2]) < 1e-5
            @test 0 < abs(Π_w[2, 1, 2, 1]) < 1e-5

            if w ≈ 0
                # 1 / (2π * im) * ∫ dv Gᴿ(v) Gᴬ(v) = - im / 2Δ
                if c === :A
                    @test abs(Π_w[1, 1, 2, 2] + im / 2 / Δ) < 1e-3
                    @test abs(Π_w[2, 2, 1, 1] + im / 2 / Δ) < 1e-3
                elseif c === :T
                    # -1 due to the bubble prefactor of T channel
                    @test abs(Π_w[1, 1, 2, 2] - im / 2 / Δ) < 1e-3
                    @test abs(Π_w[2, 2, 1, 1] - im / 2 / Δ) < 1e-3
                end
            end
        end
    end

    # Test compute_bubble and compute_bubble_smoothed with Lazy G0 and explicit G0 gives
    # similar results. (The two results should be identical for infinitely dense basis_1p.)
    for F in (:KF, :MF)
        D = F == :KF ? Inf : 5.0
        G0 = SIAMLazyGreen2P{F}(; e, Δ, t, D)

        # Compute Green2P object for G0
        if F === :KF
            basis_f = LinearSplineAndTailBasis(2, 4, -3.:3:3)
            basis_b = LinearSplineAndTailBasis(1, 0, -5.:5:5)
            basis_1p = LinearSplineAndTailBasis(1, 3, get_nonequidistant_grid(7, 121))
        else
            basis_f = ImagGridAndTailBasis(:Fermion, 2, 4, 2)
            basis_b = ImagGridAndTailBasis(:Boson, 1, 0, 3)
            basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, 32)
        end
        nind = mfRG.get_nind(G0)
        vs = get_fitting_points(basis_1p)
        green_data_tmp = G0.(vs)
        green_data = reshape(reduce(hcat, green_data_tmp), nind, nind, length(vs))
        data = mfRG.fit_basis_coeff(green_data, basis_1p, vs, 3)
        G0_basis = Green2P{F}(basis_1p, 1, data)

        for C in (:A, :P, :T)
            Π1 = compute_bubble(G0, basis_f, basis_b, Val(C); temperature=t)
            Π2 = compute_bubble(G0_basis, basis_f, basis_b, Val(C); temperature=t)
            @test norm(Π1.data - Π2.data) / norm(Π1.data) < 4e-3

            Π1 = compute_bubble_smoothed(G0, basis_f, basis_b, Val(C); temperature=t)
            Π2 = compute_bubble_smoothed(G0_basis, basis_f, basis_b, Val(C); temperature=t)
            @test norm(Π1.data - Π2.data) / norm(Π1.data) < 4e-3
        end
    end
end
