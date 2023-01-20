module mfRG
using TimerOutputs

export get_nonequidistant_grid

export nbasis, get_formalism, get_nind
export constant_basis, LinearSplineAndTailBasis, ShiftedSplineBasis
export ImagConstantBasis, ImagGridAndTailBasis
export get_fitting_points
export concat_constant_basis
export basis_integral
export basis_for_bubble

export Green2P, solve_Dyson
export Vertex4P, to_matrix, keldyshview, apply_crossing
export Bubble, compute_bubble, compute_bubble_smoothed
export fit_bosonic_basis_coeff!
export get_bare_vertex
export vertex_to_vector, vector_to_vertex

export vertex_bubble_integral
export solve_BSE, solve_BSE_left
export run_parquet
export run_parquet_without_irreducible

export RealSpaceBasis, RealSpaceBasis2P, get_indices
export RealSpaceGreen2P
export RealSpaceVertex
export RealSpaceBubble
export interpolate_to_q
export run_parquet_nonlocal

export analytic_continuation_KF_to_MF
export susceptibility_operator_SU2, compute_response_SU2
export compute_occupation_matrix, compute_occupation

export SIAMLazyGreen2P
export HubbardLazyGreen2P

"""TimerOutput object used to store mfRG timings."""
const timer = TimerOutput()

include("utility.jl")
include("acceleration.jl")

include("basis/basis_functions.jl")
include("basis/basis_functions_imaginary.jl")
include("basis/concatenated_basis.jl")
include("basis/basis_integral.jl")
include("basis/basis_integral_imaginary.jl")
include("basis/ShiftedSplineBasis.jl")
include("basis/basis_integral_self_energy.jl")
include("basis/basis_integral_bubble.jl")

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
include("self_energy.jl")
include("single_boson_exchange.jl")

include("real_space/real_space_basis.jl")
include("real_space/RealSpaceGreen.jl")
include("real_space/RealSpaceVertex.jl")
include("real_space/RealSpaceBubble.jl")

include("BSE.jl")
include("parquet_su2.jl")
include("parquet_su2_wo_irreducible.jl")
include("analytic_continuation.jl")
include("response.jl")

include("real_space/self_energy.jl")
include("real_space/real_space_BSE.jl")
include("real_space/parquet.jl")

include("models/siam.jl")
include("models/hubbard.jl")
end
