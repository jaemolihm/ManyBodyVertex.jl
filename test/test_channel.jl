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

    # Test _bubble_frequencies_inv
    for f in (:MF, :ZF, :KF), c in (:A, :P, :T)
        for (v, w) in [(10, 2), (-5, 2), (6, -3), (-5, -11)]
            v1, v2 = mfRG._bubble_frequencies(Val(f), Val(c), v, w)
            @test all((v, w) .≈ mfRG._bubble_frequencies_inv(Val(f), Val(c), v1, v2))
        end
        for (v1, v2) in [(10, 2), (-5, 2), (6, -3), (-5, -11)]
            v, w = mfRG._bubble_frequencies_inv(Val(f), Val(c), v1, v2)
            @test all((v1, v2) .≈ mfRG._bubble_frequencies(Val(f), Val(c), v, w))
        end
    end

    # Check vertex and bubble parametrizations are consistent
    for c in (:A, :P, :T)
        # Vertex indices 3 and 4 belong to the left vertex, and 1 and 2 to the right.
        inds_vertex = mfRG.indices_to_standard(Val(c), (1, 2, 3, 4))
        inds_bubble = mfRG._bubble_indices(Val(c), (3, 4, 1, 2))

        # First and third indices of inds_bubble are incoming, second and fourth are
        # outgoing in the vertex standard representation.
        @test iseven(inds_vertex[inds_bubble[1]])
        @test isodd(inds_vertex[inds_bubble[2]])
        @test iseven(inds_vertex[inds_bubble[3]])
        @test isodd(inds_vertex[inds_bubble[4]])

        # Frequencies of the vertex. Multiply -1 for incoming legs.
        for f in (:MF, :ZF, :KF)
            for (vL, vB, vR, w) in ((1, 3, -6, 23), (0, -6, 4, 22))
                v1234_L = mfRG.frequency_to_standard(Val(f), Val(c), vL, vB, w) .* (1, -1, 1, -1)
                v1234_R = mfRG.frequency_to_standard(Val(f), Val(c), vB, vR, w) .* (1, -1, 1, -1)
                if f === :MF
                    # For imaginary frequencies, -v requires additional subtraction of 1.
                    v1234_L = v1234_L .- (0, 1, 0, 1)
                    v1234_R = v1234_R .- (0, 1, 0, 1)
                end
                v1, v2 = mfRG._bubble_frequencies(Val(f), Val(c), vB, w)

                # inds_bubble[2] -- v1 --> inds_bubble[1]
                for i in 1:2
                    v1234_vertex = inds_bubble[i] <= 2 ? v1234_R : v1234_L
                    @test v1 == v1234_vertex[inds_vertex[inds_bubble[i]]]
                end

                # inds_bubble[4] -- v2 --> inds_bubble[3]
                for i in 3:4
                    v1234_vertex = inds_bubble[i] <= 2 ? v1234_R : v1234_L
                    @test v2 == v1234_vertex[inds_vertex[inds_bubble[i]]]
                end
            end
        end
    end

    # Check whether all MF frequencies are counted exactly once by the bubble parametrization
    for C in (:A, :P, :T)
        v12 = vec([mfRG._bubble_frequencies.(Val(:MF), Val(C), i, j) for i in -5:5, j in -5:5])
        @test allunique(v12)
        @test all((i, j) ∈ v12 for i in -2:2, j in -2:2)
    end
end
