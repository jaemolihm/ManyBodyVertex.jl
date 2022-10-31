module mfRG
export ConstantBasis, LinearSplineAndTailBasis
export ImagConstantBasis, ImagGridAndTailBasis
export get_fitting_points
export basis_integral

export Vertex4P, Bubble, to_matrix, vertex_keldyshview, apply_crossing
export fit_bosonic_basis_coeff!

export solve_BSE

include("interval_iterable.jl")
include("basis_functions.jl")
include("basis_functions_imaginary.jl")
include("basis_integral.jl")

include("channel.jl")
include("Vertex.jl")
include("Bubble.jl")
include("ScreenedBubble.jl")
include("BSE.jl")
end
