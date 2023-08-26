using LinearAlgebra
# using LinearMaps
# using IterativeSolvers

"""
    vertex_bubble_integral(ΓL, Π, ΓR, basis_w; basis_aux)
Compute the bubble integral ``Γ = ΓL * Π * ΓR`` on the bosonic basis basis_w in the channel
of `Π`. If `ΓL` or `ΓR` are in a different channel, `basis_aux` must be provided.

- FIXME: Currently caching assumes basis_aux is used. Possible fix is to create a
`CachedVertex` type that contains basis information.
"""
function vertex_bubble_integral(
        ΓL::AbstractVertex4P{F},
        Π::AbstractBubble{F},
        ΓR::AbstractVertex4P{F},
        basis_w;
        basis_aux=nothing
    ) where {F}
 
    T = eltype(Π)
    CL = get_channel(ΓL)
    CB = get_channel(Π)
    CR = get_channel(ΓR)

    if (CL != CB || CR != CB) && basis_aux === nothing
        error("For vertex and bubble in different channels, basis_aux must be set.")
    end

    ws = get_fitting_points(basis_w)
    nind2 = get_nind(Π)^2
    basis_L1, basis_L2 = (CL == CB) ? (ΓL.basis_f1, ΓL.basis_f2) : (basis_aux, basis_aux)
    basis_R1, basis_R2 = (CR == CB) ? (ΓR.basis_f1, ΓR.basis_f2) : (basis_aux, basis_aux)

    # Compute the bubble integral on a grid of w (bosonic frequency)
    Γ_mat = zeros(T, size(basis_L1, 2) * nind2, size(basis_R2, 2) * nind2, length(ws))
    cache_and_load_overlaps(Π, basis_L2, basis_R1)
    Threads.@threads for iw in eachindex(ws)
        w = ws[iw]
        ΓL_mat = to_matrix(ΓL, w, basis_L1, basis_L2, CB)
        Π_mat = to_matrix(Π, w, basis_L2, basis_R1)
        ΓR_mat = to_matrix(ΓR, w, basis_R1, basis_R2, CB)
        Γ_mat[:, :, iw] .= ΓL_mat * (Π_mat * ΓR_mat)
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F}(CB, T, basis_L1, basis_R2, basis_w, Π.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end

function vertex_bubble_integral(
        ΓL::AbstractVertex4P,
        Π::AbstractBubble,
        ΓR::AbstractVertex4P,
        basis_w::NamedTuple{(:freq,), Tuple{T}} where {T<:Basis};
        basis_aux=nothing
    )
    basis_aux_ = basis_aux === nothing ? nothing : basis_aux.freq
    vertex_bubble_integral(ΓL, Π, ΓR, basis_w.freq; basis_aux=basis_aux_)
end

"""
    compute_bse_matrix(Γ, Π, C, basis_L1, basis_L2, basis_R1)
Return the matrix represetntation of ``I - Γ * Π``
"""
function compute_bse_matrix(Γ, Π, w, basis_L1, basis_L2, basis_R1)
    basis_L1 === basis_R1 || error("basis_L1 and basis_R1 must be identical")
    Γ_mat = to_matrix(Γ, w, basis_L1, basis_L2, get_channel(Π))
    Π_mat = to_matrix(Π, w, basis_L2, basis_R1)
    I - (Γ_mat * Π_mat)
end

"""
    solve_BSE(Γ1, Π, Γ0, basis_w, basis_aux=nothing)
Solve the BSE ``Γ = (I - Γ1 * Π)⁻¹ * Γ0 - Γ0``.

If the channel of `Γ1` is different from that of `Π`, use `basis_aux` as the basis for `Γ1`.
"""
function solve_BSE(Γ1, Π::AbstractBubble{F, T}, Γ0, basis_w; basis_aux=nothing) where {F, T}
    C = get_channel(Π)
    if get_channel(Γ1) != get_channel(Π)
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
    Γ = Vertex4P{F}(C, T, basis_Γ1_1, Γ_1st.basis_f2, basis_w, Γ_1st.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end


"""
        compute_bse_matrix_left(Γ, Π, w, basis_L2, basis_R1, basis_R2)
Return the matrix represetntation of ``I - Π * Γ``
"""
function compute_bse_matrix_left(Γ, Π, w, basis_L2, basis_R1, basis_R2)
    basis_L2 === basis_R2 || error("basis_L2 and basis_R2 must be identical")
    Γ_mat = to_matrix(Γ, w, basis_R1, basis_R2, get_channel(Π))
    Π_mat = to_matrix(Π, w, basis_L2, basis_R1)
    I - (Π_mat * Γ_mat)
end

"""
    solve_BSE_left(Γ1, Π, Γ0, basis_w, basis_aux=nothing)
Solve the BSE ``Γ = Γ0 * (I - Π * Γ1)⁻¹ - Γ0``.

If the channel of `Γ1` is different from that of `Π`, use `basis_aux` as the basis for `Γ1`.
"""
function solve_BSE_left(Γ1, Π::AbstractBubble{F, T}, Γ0, basis_w; basis_aux=nothing) where {F, T}
    C = get_channel(Π)
    if get_channel(Γ1) != C
        basis_aux === nothing && error("Vertex and bubble have different channels. basis_aux must be set.")
        basis_Γ1_1 = basis_aux
        basis_Γ1_2 = basis_aux
    else
        basis_Γ1_1 = Γ1.basis_f1
        basis_Γ1_2 = Γ1.basis_f2
    end
    # 1st-order solution: Γ_1st = Γ0 * Π * Γ1
    Γ_1st = vertex_bubble_integral(Γ0, Π, Γ1, basis_w; basis_aux)

    # Solve the BSE on a grid of w (bosonic frequency)
    # BSE: Γ = Γ_1st * (I - Π * Γ1)⁻¹
    ws = get_fitting_points(basis_w)
    Γ_mat = zeros(T, size(Γ_1st.data)[1:2]..., length(ws))
    @views for (iw, w) in enumerate(ws)
        Γ_1st_w = to_matrix(Γ_1st, w)
        bse_mat = compute_bse_matrix_left(Γ1, Π, w, basis_Γ1_2, basis_Γ1_1, basis_Γ1_2)
        Γ_mat[:, :, iw] .= (bse_mat' \ Γ_1st_w')'
    end

    # Fit the data on the w grid to the basis and store in Vertex4P object
    Γ = Vertex4P{F}(C, T, Γ_1st.basis_f1, basis_Γ1_2, basis_w, Γ_1st.norb)
    fit_bosonic_basis_coeff!(Γ, Γ_mat, ws)
    Γ
end
