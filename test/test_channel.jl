using Test
using mfRG

@testset "channels" begin
    for c in (:A, :P, :T)
        inds_standard = mfRG.indices_to_standard(Val(c), (1, 2, 3, 4))
        @test mfRG.indices_to_channel(Val(c), inds_standard) == (1, 2, 3, 4)

        vvw = rand(3)
        v1234 = mfRG.frequency_to_standard(Val(:ZF), Val(c), vvw...)
        @test all(mfRG.frequency_to_channel(Val(:ZF), Val(c), v1234...) .≈ vvw)
    end
end
