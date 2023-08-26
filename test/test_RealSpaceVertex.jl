using mfRG
using Test

@testset "RealSpaceVertex" begin
    using mfRG: real_space_channel, real_space_convert_channel, apply_crossing

    lattice = SMatrix{2, 2}([1. 0; 0 1])
    positions = [SVector(0., 0.)]
    bonds_L = [(1, 1, SVector(0, 0))]
    bonds_R = [(1, 1, SVector(0, 0))]
    qgrid = (2, 2)
    rbasis = RealSpaceBasis(lattice, positions, bonds_L, bonds_R, qgrid);

    basis_v = ImagConstantBasis()
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 5)

    x = RealSpaceVertex(:A, rbasis, typeof(Vertex4P{:MF}(:P, basis_v, basis_v, basis_w)); channel=:P)
    @test real_space_channel(x) == :A
    # TODO: We need a channel field in RealSpaceVertex
    @test get_channel(x) == :P
    @test mfRG.get_formalism(x) == :MF

    for ind in get_indices(rbasis)
        y = Vertex4P{:MF}(:P, basis_v, basis_v, basis_w)
        y.data .= rand(eltype(y), size(y.data))
        insert!(x.vertices_R, ind, y)
    end

    for C in (:A, :P, :T)
        y = real_space_convert_channel(x, rbasis, C)
        @test real_space_channel(y) == C
        @test get_channel(y) == :P
        if C == real_space_channel(x)
            @test getproperty.(x.vertices_R, :data).values ≈ getproperty.(y.vertices_R, :data).values
        end
    end

    # Test apply_crossing
    x = RealSpaceVertex(:A, rbasis, typeof(Vertex4P{:MF}(:A, basis_v, basis_v, basis_w)))
    for ind in get_indices(rbasis)
        y = Vertex4P{:MF}(:A, basis_v, basis_v, basis_w)
        y.data .= rand(eltype(y), size(y.data))
        insert!(x.vertices_R, ind, y)
    end
    y = apply_crossing(x)
    @test real_space_channel(y) == :T
    @test get_channel(y) == :T
    @test y.vertices_R[(1, 1, 1)] isa typeof(apply_crossing(x.vertices_R[(1, 1, 1)]))
    @test y.vertices_R[(1, 1, 1)].data ≈ apply_crossing(x.vertices_R[(1, 1, 1)]).data

    # Test cache_vertex_matrix
    basis_v2 = ImagGridAndTailBasis(:Fermion, 1, 0, 3)
    x = RealSpaceVertex(:A, rbasis, typeof(Vertex4P{:MF}(:A, basis_v, basis_v, basis_w)))
    for ind in get_indices(rbasis)
        y = Vertex4P{:MF}(:A, basis_v, basis_v, basis_w)
        y.data .= rand(eltype(y), size(y.data))
        insert!(x.vertices_R, ind, y)
    end
    y = mfRG.cache_vertex_matrix(x, :P, [-1, 0, 1], (; freq=basis_v2, r=rbasis))
    @test real_space_channel(y) == :P
    @test get_channel(y) == :P
    @test y isa RealSpaceVertex{:MF}
    @test eltype(y.vertices_R) <: mfRG.CachedVertex4P{:MF}
end
