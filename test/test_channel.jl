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

    for f in (:MF, :ZF, :KF)
        for c in (:A, :P, :T)
            for v1 in -2:2, v2 in -2:2, w in -2:2
                x = mfRG.frequency_to_standard(Val(f), Val(c), v1, v2, w)
                y = mfRG.frequency_to_channel(Val(f), Val(c), x...)
                @test sum(x) == (f === :MF ? -2 : 0)
                @test y == (v1, v2, w)
            end
        end
    end
end
