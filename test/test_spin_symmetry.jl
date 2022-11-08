using mfRG
using Test

@testset "Vertex SU2" begin
    using mfRG: su2_convert_spin_channel, su2_bare_vertex, su2_apply_crossing
    # A and T channel uses (d, m) parametrization, P channel uses (p, m) parametrization
    basis1 = LinearSplineAndTailBasis(0, 2, [-1., 1.])
    basis2 = LinearSplineAndTailBasis(0, 1, [-2., 2.])
    Γ_A = [Vertex4P{:KF, :A}(Float64, basis1, basis2, basis2), Vertex4P{:KF, :A}(Float64, basis1, basis2, basis2)]
    Γ_A[1].data .= rand(size(Γ_A[1].data)...)
    Γ_A[2].data .= rand(size(Γ_A[2].data)...)
    @test su2_convert_spin_channel(:A, Γ_A)[1].data ≈ Γ_A[1].data
    @test su2_convert_spin_channel(:A, Γ_A)[2].data ≈ Γ_A[2].data
    @test su2_convert_spin_channel(:T, Γ_A)[1].data ≈ Γ_A[1].data
    @test su2_convert_spin_channel(:T, Γ_A)[2].data ≈ Γ_A[2].data
    @test su2_convert_spin_channel(:P, Γ_A)[1].data ≈ (Γ_A[1].data .- Γ_A[2].data) ./ 2
    @test su2_convert_spin_channel(:P, Γ_A)[2].data ≈ Γ_A[2].data

    Γ_P = [Vertex4P{:KF, :P}(Float64, basis1, basis2, basis2), Vertex4P{:KF, :P}(Float64, basis1, basis2, basis2)]
    Γ_P[1].data .= rand(size(Γ_P[1].data)...)
    Γ_P[2].data .= rand(size(Γ_P[2].data)...)
    @test su2_convert_spin_channel(:A, Γ_P)[1].data ≈ 2 .* Γ_P[1].data .- Γ_P[2].data
    @test su2_convert_spin_channel(:A, Γ_P)[2].data ≈ Γ_P[2].data
    @test su2_convert_spin_channel(:T, Γ_P)[1].data ≈ 2 .* Γ_P[1].data .- Γ_P[2].data
    @test su2_convert_spin_channel(:T, Γ_P)[2].data ≈ Γ_P[2].data
    @test su2_convert_spin_channel(:P, Γ_P)[1].data ≈ Γ_P[1].data
    @test su2_convert_spin_channel(:P, Γ_P)[2].data ≈ Γ_P[2].data

    # Test su2_bare_vertex
    U = 5.0
    for F in (:KF, :MF)
        Γ0_A = su2_bare_vertex(U, Val(F), Val(:A))
        Γ0_P = su2_bare_vertex(U, Val(F), Val(:P))
        Γ0_T = su2_bare_vertex(U, Val(F), Val(:T))
        @test Γ0_A[1].data ≈ Γ0_A[1].data .* +1
        @test Γ0_A[2].data ≈ Γ0_A[1].data .* -1
        @test Γ0_P[1].data ≈ Γ0_A[1].data .* 0
        @test Γ0_P[2].data ≈ Γ0_A[1].data .* -1
        @test Γ0_T[1].data ≈ Γ0_A[1].data .* +1
        @test Γ0_T[2].data ≈ Γ0_A[1].data .* -1
        @test Γ0_T[1].data ≈ su2_apply_crossing(Γ0_A)[1].data
        @test Γ0_T[2].data ≈ su2_apply_crossing(Γ0_A)[2].data
    end
end
