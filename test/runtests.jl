using mfRG
using Test

@testset "mfRG.jl" begin
    include("test_nonequidistant_grid.jl")
    include("test_basis.jl")
    include("test_channel.jl")
    include("test_vertex4P.jl")
    include("test_bubble.jl")
    include("test_spin_symmetry.jl")

    include("test_BSE.jl")
    include("test_model_siam.jl")
end
