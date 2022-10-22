module mfRG
export ConstantBasis, LinearSplineAndTailBasis
export get_fitting_points
export basis_integral

export Vertex4P, Bubble, to_matrix, bubble_to_matrix, vertex_keldyshview, apply_crossing
export fit_bosonic_basis_coeff!

export solve_BSE

include("interval_iterable.jl")
include("basis_functions.jl")
include("basis_integral.jl")

include("channel.jl")
include("Vertex.jl")
include("Bubble.jl")
include("BSE.jl")
end
