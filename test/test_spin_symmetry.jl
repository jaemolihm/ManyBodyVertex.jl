using mfRG
using Test

@testset "Vertex SU2" begin
    using mfRG: su2_convert_spin_channel, su2_bare_vertex, su2_apply_crossing
    basis1 = LinearSplineAndTailBasis(0, 2, [-1., 1.])
    basis2 = LinearSplineAndTailBasis(0, 1, [-2., 2.])
    Γ_A = [Vertex4P{:KF, :A}(Float64, basis1, basis2, basis2), Vertex4P{:KF, :A}(Float64, basis1, basis2, basis2)]
    Γ_A[1].data .= rand(size(Γ_A[1].data)...)
    Γ_A[2].data .= rand(size(Γ_A[2].data)...)
    @test su2_convert_spin_channel(:A, Γ_A)[1].data ≈ Γ_A[1].data
    @test su2_convert_spin_channel(:A, Γ_A)[2].data ≈ Γ_A[2].data
    @test su2_convert_spin_channel(:T, Γ_A)[1].data ≈ (Γ_A[1].data .+ 3 .* Γ_A[2].data) ./ 2
    @test su2_convert_spin_channel(:T, Γ_A)[2].data ≈ (Γ_A[1].data - Γ_A[2].data) ./ 2
    @test su2_convert_spin_channel(:P, Γ_A)[1].data ≈ (Γ_A[1].data .- 3 .* Γ_A[2].data) ./ 2
    @test su2_convert_spin_channel(:P, Γ_A)[2].data ≈ (Γ_A[1].data .+ Γ_A[2].data) ./ 2

    Γ_P = [Vertex4P{:KF, :P}(Float64, basis1, basis2, basis2), Vertex4P{:KF, :P}(Float64, basis1, basis2, basis2)]
    Γ_P[1].data .= rand(size(Γ_P[1].data)...)
    Γ_P[2].data .= rand(size(Γ_P[2].data)...)
    @test su2_convert_spin_channel(:A, Γ_P)[1].data ≈ (Γ_P[1].data .+ 3 .* Γ_P[2].data) ./ 2
    @test su2_convert_spin_channel(:A, Γ_P)[2].data ≈ (.-Γ_P[1].data .+ Γ_P[2].data) ./ 2
    @test su2_convert_spin_channel(:T, Γ_P)[1].data ≈ (.-Γ_P[1].data .+ 3 .* Γ_P[2].data) ./ 2
    @test su2_convert_spin_channel(:T, Γ_P)[2].data ≈ (Γ_P[1].data .+ Γ_P[2].data) ./ 2
    @test su2_convert_spin_channel(:P, Γ_P)[1].data ≈ Γ_P[1].data
    @test su2_convert_spin_channel(:P, Γ_P)[2].data ≈ Γ_P[2].data

    # P -> A -> P
    Γ_P_in_A = [Vertex4P{:KF, :A}(basis1, basis2, basis2, 1, x.data) for x in su2_convert_spin_channel(:A, Γ_P)]
    Γ_P_2 = su2_convert_spin_channel(:P, Γ_P_in_A)
    @test Γ_P_2[1].data ≈ Γ_P[1].data
    @test Γ_P_2[2].data ≈ Γ_P[2].data

    # Test su2_bare_vertex
    U = 5.0
    for F in (:KF, :MF)
        Γ0_A = su2_bare_vertex(Val(F), Val(:A), U)
        Γ0_P = su2_bare_vertex(Val(F), Val(:P), U)
        Γ0_T = su2_bare_vertex(Val(F), Val(:T), U)
        @test Γ0_A[1].data ≈ Γ0_A[1].data .* +1
        @test Γ0_A[2].data ≈ Γ0_A[1].data .* -1
        @test Γ0_P[1].data ≈ Γ0_A[1].data .* 2
        @test Γ0_P[2].data ≈ Γ0_A[1].data .* 0
        @test Γ0_T[1].data ≈ Γ0_A[1].data .* -1
        @test Γ0_T[2].data ≈ Γ0_A[1].data .* +1
        @test Γ0_T[1].data ≈ su2_apply_crossing(Γ0_A)[1].data
        @test Γ0_T[2].data ≈ su2_apply_crossing(Γ0_A)[2].data
        @test Γ0_A[1].data ≈ su2_convert_spin_channel(:A, Γ0_P)[1].data
        @test Γ0_A[2].data ≈ su2_convert_spin_channel(:A, Γ0_P)[2].data
        @test Γ0_P[1].data ≈ su2_convert_spin_channel(:P, Γ0_A)[1].data
        @test Γ0_P[2].data ≈ su2_convert_spin_channel(:P, Γ0_A)[2].data
        @test Γ0_T[1].data ≈ su2_convert_spin_channel(:T, Γ0_A)[1].data
        @test Γ0_T[2].data ≈ su2_convert_spin_channel(:T, Γ0_A)[2].data
    end
end
