using Test
using mfRG

@testset "model SIAM" begin
    e = 0.5
    Δ = 1.0
    t = 0.1
    v = 10.0
    @test mfRG.siam_get_green_function(v, e, Δ, t, Val(:MF)) isa Number
    @test mfRG.siam_get_green_function(v, e, Δ, t, Val(:KF)) isa SMatrix{2,2}
    for F in (:MF, :KF)
        @inferred mfRG.siam_get_green_function(v, e, Δ, t, Val(F))
    end

    basis_f = LinearSplineAndTailBasis(2, 4, -3:0.6:3)
    basis_b = LinearSplineAndTailBasis(1, 3, -2:0.5:2)
    for F in (:MF, :KF), C in (:A, :P, :T)
        @test mfRG.siam_get_bubble(basis_f, basis_b, Val(F), Val(C); e, Δ, t) isa Bubble{F, C, ComplexF64}
    end
end
