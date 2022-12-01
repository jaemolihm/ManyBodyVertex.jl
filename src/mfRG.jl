module mfRG
export get_nonequidistant_grid

export nbasis
export ConstantBasis, LinearSplineAndTailBasis, ShiftedSplineBasis
export ImagConstantBasis, ImagGridAndTailBasis
export get_fitting_points
export basis_integral
export basis_for_bubble

export Green2P, solve_Dyson
export Vertex4P, to_matrix, vertex_keldyshview, apply_crossing
export Bubble, compute_bubble, compute_bubble_smoothed
export fit_bosonic_basis_coeff!
export get_bare_vertex
export vertex_to_vector, vector_to_vertex

export vertex_bubble_integral
export solve_BSE, solve_BSE_left
export run_parquet
export run_parquet_without_irreducible

export SIAMLazyGreen2P

include("utility.jl")
include("acceleration.jl")

include("basis/basis_functions.jl")
include("basis/basis_functions_imaginary.jl")
include("basis/basis_integral.jl")
include("basis/basis_integral_imaginary.jl")
include("basis/ShiftedSplineBasis.jl")
include("basis/basis_integral_self_energy.jl")

include("channel.jl")
include("spin_symmetry_su2.jl")

include("AbstractFrequencyVertex.jl")
include("Green.jl")
include("Vertex.jl")
include("CachedVertex.jl")
include("AsymptoticVertex.jl")
include("Bubble.jl")
include("ScreenedBubble.jl")
include("compute_bubble.jl")

include("BSE.jl")
include("parquet_su2.jl")
include("parquet_su2_wo_irreducible.jl")

include("models.jl")
end
