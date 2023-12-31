using ManyBodyVertex
using Test

@testset "ManyBodyVertex.jl" begin
    include("test_nonequidistant_grid.jl")
    include("test_basis.jl")
    include("test_basis_shiftedspline.jl")
    include("test_real_space.jl")

    include("test_channel.jl")
    include("test_vertex4P.jl")
    include("test_bubble.jl")
    include("test_spin_symmetry.jl")
    include("test_RealSpaceVertex.jl")
    include("test_real_space_bubble.jl")

    include("test_BSE.jl")
    include("test_model_siam.jl")
    include("test_hubbard_atom.jl")

    include("test_parquet.jl")
    include("test_response.jl")
    include("test_analytic_continuation.jl")
end
