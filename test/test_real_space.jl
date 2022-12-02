using mfRG
using Test

@testset "real space channels" begin
    using LinearAlgebra
    using StaticArrays
    using mfRG: lattice_vectors_to_standard, lattice_vectors_to_channel, frequency_to_channel
    R = SVector{3}(rand(-5:5, 3))
    Rp = SVector{3}(rand(-5:5, 3))
    R_B = SVector{3}(rand(-5:5, 3))
    for C in (:A, :P, :T)
        R1234 = lattice_vectors_to_standard(Val(C), R, Rp, R_B)
        @test lattice_vectors_to_channel(Val(C), R1234) == (; R, Rp, R_B)
    end

    R1234 = Tuple(SVector{3}(rand(-5:5, 3)) for _ in 1:4)
    for C in (:A, :P, :T)
        R, Rp, R_B = lattice_vectors_to_channel(Val(C), R1234)
        @test lattice_vectors_to_standard(Val(C), R, Rp, R_B) == R1234 .- Ref(R1234[4])
    end

    # Test consistency with the momentum representation (which is same as the frequency representation)
    k1234 = rand(4)
    R1234 = rand(4)
    k1234 .-= sum(k1234) / 4
    k_dot_R = dot(k1234, R1234)
    for C in (:A, :P, :T)
        k, kp, q = frequency_to_channel(Val(:KF), Val(C), k1234)
        R, Rp, R_B = lattice_vectors_to_channel(Val(C), R1234)
        @test k * R - kp * Rp + q * (R_B + (R - Rp) / 2) ≈ -k_dot_R
    end
end
