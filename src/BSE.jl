using LinearAlgebra
using LinearMaps
using IterativeSolvers

"""
    vertex_bubble_integral(Γ1, Π, Γ2, basis_w)
Compute the bubble integral ``Γ = Γ1 * Π * Γ2`` on the bosonic basis basis_w.
"""
function vertex_bubble_integral(Γ1::AbstractVertex4P{F, C, T}, Π, Γ2, basis_w) where {F, C, T}
    ws = get_fitting_points(basis_w)
    Γ_mat = zeros(T, size(Γ1.data, 1), size(Γ2.data, 2), length(ws))
    overlap = basis_integral(Γ1.basis_f2, Γ2.basis_f1, Π.basis_f)

    # Compute the bubble integral on a grid of w (bosonic frequency)
    for (iw, w) in enumerate(ws)
        Γ1_mat = to_matrix(Γ1, w)
        Π_mat = to_matrix(Π, w, overlap)
        Γ2_mat = to_matrix(Γ2, w)
        Γ_mat[:, :, iw] .= Γ1_mat * Π_mat * Γ2_mat
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F, C}(T, Γ1.basis_f1, Γ2.basis_f2, basis_w, Γ1.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end

abstract type AbstractBSEMap{F, C, T} <: LinearMaps.LinearMap{T} end

"""
    BSEMap(Γ, Π, Γ_in) <: AbstractBSEMap{F, T} <: LinearMaps.LinearMap{T}
``Γ_out = BSEMap(Γ, Π) * Γ_in = Γ_in - Γ * Π * Γ_in``.

Before using a BSEMap, one should call `set_bosonic_frequency(bsemap, w)` with the bosonic
frequency to use.
"""
mutable struct BSEMap{F, C, T, FT, VT <: Vertex4P{F, C, T}, BT <: Bubble{F, T}} <: AbstractBSEMap{F, C, T}
    Γ::VT
    Π::BT
    overlap::Array{FT, 3}
    ΓΠ_mat::Matrix{T}
    size_Γ_in::NTuple{2, Int}
    function BSEMap(Γ::VT, Π::BT, Γ_in) where {VT <: Vertex4P{F, C, T}, BT} where {F, C, T}
        size_Γ_in = size(Γ_in.data)[1:2]
        overlap = basis_integral(Γ.basis_f2, Γ_in.basis_f1, Π.basis_f)
        FT = eltype(overlap)
        ΓΠ_mat = zeros(T, size_Γ_in[1], size_Γ_in[1])
        new{F, C, T, FT, VT, BT}(Γ, Π, overlap, ΓΠ_mat, size_Γ_in)
    end
end
Base.size(bsemap::BSEMap{F}) where {F} = (prod(bsemap.size_Γ_in), prod(bsemap.size_Γ_in))

function set_bosonic_frequency(bsemap::BSEMap, w)
    Π_mat = to_matrix(bsemap.Π, w, bsemap.overlap)
    Γ_mat = to_matrix(bsemap.Γ, w)
    mul!(bsemap.ΓΠ_mat, Γ_mat, Π_mat)
end

function LinearAlgebra.mul!(Γ_out_vec::AbstractVecOrMat, bsemap::BSEMap, Γ_in_vec::AbstractVector)
    # Calculate the Bubble integral. ΓΠ_mat is precomputed in set_bosonic_frequency.
    # Inputs are vectors, reshape them into matrix form and them do matrix multiplication.
    Γ_in_mat = reshape(Γ_in_vec, bsemap.size_Γ_in)
    Γ_out_vec .= Γ_in_vec .- vec(bsemap.ΓΠ_mat * Γ_in_mat)
    Γ_out_vec
end

"""
    solve_BSE(Γ1, Π, Γ0, basis_w)
Solve the BSE ``Γ = Γ1 * Π * Γ0 + Γ1 * Π * Γ``.
"""
function solve_BSE(Γ1::AbstractVertex4P{F, C, T}, Π, Γ0, basis_w) where {F, C, T}
    # 1st-order solution: Γ1 * Π * Γ0
    Γ_1st = vertex_bubble_integral(Γ1, Π, Γ0, basis_w)

    # Solve the BSE on a grid of w (bosonic frequency)
    # BSE: Γ = Γ_1st + Γ1 * Π * Γ.
    ws = get_fitting_points(basis_w)
    Γ_mat = zeros(T, size(Γ_1st.data)[1:2]..., length(ws))
    bsemap = BSEMap(Γ1, Π, Γ_1st)
    @views for (iw, w) in enumerate(ws)
        set_bosonic_frequency(bsemap, w)
        Γ_1st_w = to_matrix(Γ_1st, w)
        Γ_mat[:, :, iw] .= Γ_1st_w
        IterativeSolvers.gmres!(vec(Γ_mat[:, :, iw]), bsemap, vec(Γ_1st_w));
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F, C}(T, Γ_1st.basis_f1, Γ_1st.basis_f2, basis_w, Γ_1st.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end
