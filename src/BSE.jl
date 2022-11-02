using LinearAlgebra
# using LinearMaps
# using IterativeSolvers

"""
    vertex_bubble_integral(ΓL, Π, ΓR, basis_w; basis_aux)
Compute the bubble integral ``Γ = ΓL * Π * ΓR`` on the bosonic basis basis_w in the channel
of `Π`. If `ΓL` or `ΓR` are in a different channel, `basis_aux` must be provided.
"""
function vertex_bubble_integral(
        ΓL::AbstractVertex4P{F, CL, T},
        Π::AbstractBubble{F, CB, T},
        ΓR::AbstractVertex4P{F, CR, T},
        basis_w;
        basis_aux=nothing
    ) where {F, T, CL, CB, CR}

    if (CL != CB || CR != CB) && basis_aux === nothing
        error("For vertex and bubble in different channels, basis_aux must be set.")
    end

    ws = get_fitting_points(basis_w)
    nind2 = get_nind(ΓL)^2
    basis_L1, basis_L2 = (CL == CB) ? (ΓL.basis_f1, ΓL.basis_f2) : (basis_aux, basis_aux)
    basis_R1, basis_R2 = (CR == CB) ? (ΓR.basis_f1, ΓR.basis_f2) : (basis_aux, basis_aux)
    Γ_mat = zeros(T, size(basis_L1, 2) * nind2, size(basis_R2, 2) * nind2, length(ws))

    # Compute the bubble integral on a grid of w (bosonic frequency)
    for (iw, w) in enumerate(ws)
        ΓL_mat = to_matrix(ΓL, w, basis_L1, basis_L2, Val(CB))
        Π_mat = to_matrix(Π, w, basis_L2, basis_R1)
        ΓR_mat = to_matrix(ΓR, w, basis_R1, basis_R2, Val(CB))
        Γ_mat[:, :, iw] .= ΓL_mat * (Π_mat * ΓR_mat)
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F, CB}(T, basis_L1, basis_R2, basis_w, ΓL.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end

"""
    compute_bse_matrix(Γ, Π, C, basis_L1, basis_L2, basis_R1)
Return the matrix represetntation of ``I - Γ * Π``
"""
function compute_bse_matrix(Γ, Π, w, basis_L1, basis_L2, basis_R1)
    basis_L1 === basis_R1 || error("basis_L1 and basis_R1 must be identical")
    Γ_mat = to_matrix(Γ, w, basis_L1, basis_L2, Val(channel(Π)))
    Π_mat = to_matrix(Π, w, basis_L2, basis_R1)
    I - (Γ_mat * Π_mat)
end

"""
    solve_BSE(Γ1, Π, Γ0, basis_w, basis_aux=nothing)
Solve the BSE ``Γ = (I - Γ1 * Π)⁻¹ * Γ0 - Γ0``.

If the channel of `Γ1` is different from that of `Π`, use `basis_aux` as the basis for `Γ1`.
"""
function solve_BSE(Γ1, Π::AbstractBubble{F, C, T}, Γ0, basis_w; basis_aux=nothing) where {F, C, T}
    if channel(Γ1) != C
        basis_aux === nothing && error("Vertex and bubble have different channels. basis_aux must be set.")
        basis_Γ1_1 = basis_aux
        basis_Γ1_2 = basis_aux
    else
        basis_Γ1_1 = Γ1.basis_f1
        basis_Γ1_2 = Γ1.basis_f2
    end
    # 1st-order solution: Γ_1st = Γ1 * Π * Γ0
    Γ_1st = vertex_bubble_integral(Γ1, Π, Γ0, basis_w; basis_aux)

    # Solve the BSE on a grid of w (bosonic frequency)
    # BSE: Γ = (I - Γ1 * Π)⁻¹ * Γ_1st
    ws = get_fitting_points(basis_w)
    Γ_mat = zeros(T, size(Γ_1st.data)[1:2]..., length(ws))
    @views for (iw, w) in enumerate(ws)
        Γ_1st_w = to_matrix(Γ_1st, w)
        bse_mat = compute_bse_matrix(Γ1, Π, w, basis_Γ1_1, basis_Γ1_2, basis_Γ1_1)
        Γ_mat[:, :, iw] .= bse_mat \ Γ_1st_w
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F, C}(T, basis_Γ1_1, Γ_1st.basis_f2, basis_w, Γ_1st.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end

# abstract type AbstractBSEMap{F, C, T} <: LinearMaps.LinearMap{T} end

# """
#     BSEMap(Γ, Π, Γ_in) <: AbstractBSEMap{F, T} <: LinearMaps.LinearMap{T}
# ``Γ_out = BSEMap(Γ, Π) * Γ_in = Γ_in - Γ * Π * Γ_in``.

# Before using a BSEMap, one should call `set_bosonic_frequency!(bsemap, w)` with the bosonic
# frequency to use.
# """
# mutable struct BSEMap{F, C, T, FT, VT <: Vertex4P{F, C, T}, BT <: Bubble{F, T}} <: AbstractBSEMap{F, C, T}
#     Γ::VT
#     Π::BT
#     overlap::Array{FT, 3}
#     ΓΠ_mat::Matrix{T}
#     size_Γ_in::NTuple{2, Int}
#     function BSEMap(Γ::VT, Π::BT, Γ_in) where {VT <: Vertex4P{F, C, T}, BT} where {F, C, T}
#         size_Γ_in = size(Γ_in.data)[1:2]
#         overlap = basis_integral(Γ.basis_f2, Γ_in.basis_f1, Π.basis_f)
#         FT = eltype(overlap)
#         ΓΠ_mat = zeros(T, size_Γ_in[1], size_Γ_in[1])
#         new{F, C, T, FT, VT, BT}(Γ, Π, overlap, ΓΠ_mat, size_Γ_in)
#     end
# end
# Base.size(bsemap::BSEMap{F}) where {F} = (prod(bsemap.size_Γ_in), prod(bsemap.size_Γ_in))

# function set_bosonic_frequency!(bsemap::BSEMap, w)
#     Π_mat = to_matrix(bsemap.Π, w, bsemap.overlap)
#     Γ_mat = to_matrix(bsemap.Γ, w)
#     mul!(bsemap.ΓΠ_mat, Γ_mat, Π_mat)
# end

# function LinearAlgebra.mul!(Γ_out_vec::AbstractVecOrMat, bsemap::BSEMap, Γ_in_vec::AbstractVector)
#     # Calculate the Bubble integral. ΓΠ_mat is precomputed in set_bosonic_frequency!.
#     # Inputs are vectors, reshape them into matrix form and them do matrix multiplication.
#     Γ_in_mat = reshape(Γ_in_vec, bsemap.size_Γ_in)
#     Γ_out_vec .= Γ_in_vec .- vec(bsemap.ΓΠ_mat * Γ_in_mat)
#     Γ_out_vec
# end

# """
#     solve_BSE(Γ1, Π, Γ0, basis_w)
# Solve the BSE ``Γ = Γ1 * Π * Γ0 + Γ1 * Π * Γ``.
# """
# function solve_BSE(Γ1::AbstractVertex4P{F, C, T}, Π, Γ0, basis_w) where {F, C, T}
#     # 1st-order solution: Γ1 * Π * Γ0
#     Γ_1st = vertex_bubble_integral(Γ1, Π, Γ0, basis_w)

#     # Solve the BSE on a grid of w (bosonic frequency)
#     # BSE: Γ = Γ_1st + Γ1 * Π * Γ.
#     ws = get_fitting_points(basis_w)
#     Γ_mat = zeros(T, size(Γ_1st.data)[1:2]..., length(ws))
#     bsemap = BSEMap(Γ1, Π, Γ_1st)
#     @views for (iw, w) in enumerate(ws)
#         set_bosonic_frequency!(bsemap, w)
#         Γ_1st_w = to_matrix(Γ_1st, w)
#         Γ_mat[:, :, iw] .= Γ_1st_w
#         IterativeSolvers.gmres!(vec(Γ_mat[:, :, iw]), bsemap, vec(Γ_1st_w));
#     end

#     # Fit the data on the w grid to the basis and store in Vertex4P object
#     Γ = Vertex4P{F, C}(T, Γ_1st.basis_f1, Γ_1st.basis_f2, basis_w, Γ_1st.norb)
#     fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
#     Γ
# end
