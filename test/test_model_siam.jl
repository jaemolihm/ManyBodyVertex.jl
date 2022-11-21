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
        @test G0(v) ≈ siam_get_green_function(v, Val(F); e, Δ, t, D)
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

@testset "siam_get_bubble_improved" begin
    using mfRG: siam_get_bubble_improved
    # With siam_get_bubble_improved, sanity checks for the bubble is satisfied to high
    # accuracy even for a coarse basis_f.
    e = 0.5
    Δ = 1.5
    t = 0.1
    basis_1p = LinearSplineAndTailBasis(1, 3, range(-10, 10, step=0.1))
    basis_f = LinearSplineAndTailBasis(2, 4, range(-10, 10, step=5.0))
    basis_b = LinearSplineAndTailBasis(1, 0, -3.:3:3)

    for c in (:A, :P, :T)
        Π = siam_get_bubble_improved(basis_f, basis_b, basis_1p, Val(:KF), Val(c); e, Δ, t)
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
end
