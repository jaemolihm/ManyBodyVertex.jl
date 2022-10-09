using mfRG
using Test

@testset "mfRG.jl" begin
    include("test_basis.jl")
    include("test_vertex4P.jl")
end
