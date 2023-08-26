"""
    SBEReducibleVertex(U, W, Λb, Λ) => ∇
Vertex in the single-boson exchange (SBE) decomposition.
We define ``∇(v1, v2, w) = Λb(v1, w) * W(w) * Λ(v2, w) - U``.
"""
mutable struct SBEReducibleVertex{F, T, VT1, VT2, VT3, VT4} <: AbstractVertex4P{F, T}
    # Number of orbitals
    norb::Int
    U::VT1
    W::VT2
    Λb::VT3
    Λ::VT4
    function SBEReducibleVertex(U, W::AbstractVertex4P{F, T}, Λb, Λ) where {F, T}
        C = get_channel(W)
        get_channel(Λb) === C || throw(ArgumentError("Channel of Λb does not match with W"))
        get_channel(Λ) === C || throw(ArgumentError("Channel of Λ does not match with W"))
        new{F, T, typeof(U), typeof(W), typeof(Λb), typeof(Λ)}(U.norb, U, W, Λb, Λ)
    end
end

get_channel(∇::SBEReducibleVertex) = ∇.W.channel

function Base.getproperty(∇::SBEReducibleVertex, s::Symbol)
    if s === :basis_f1
        getfield(getfield(∇, :Λb), s)
    elseif s === :basis_f2
        getfield(getfield(∇, :Λ), s)
    else
        getfield(∇, s)
    end
end

_vertex_names(::SBEReducibleVertex) = (:W, :Λb, :Λ)

function Base.similar(∇::SBEReducibleVertex{F, T}) where {F, T}
    SBEReducibleVertex(similar(∇.U), similar(∇.W), similar(∇.Λb), similar(∇.Λ))
end

"""
    (∇::SBEReducibleVertex{F})(v1, v2, w, C_out::Symbol=get_channel(∇)) where {F}
Evaluate the SBE vertex at given frequencies in the channel parametrization.
"""
function (∇::SBEReducibleVertex{F})(v1, v2, w, C_out::Symbol=get_channel(∇)) where {F}
    C_∇ = get_channel(∇)
    v1234 = frequency_to_standard(Val(F), C_out, v1, v2, w)
    vvw = frequency_to_channel(Val(F), C_∇, v1234)
    U  = ∇.U(vvw...)
    W  = ∇.W(vvw...)
    Λb = ∇.Λb(vvw...)
    Λ  = ∇.Λ(vvw...)
    ∇_vvw = Λb * W * Λ .- U
    _permute_orbital_indices_matrix_4p(C_∇, C_out, ∇_vvw, get_nind(∇))
end

function (∇::SBEReducibleVertex{F, T})(v1::AbstractVector, v2::AbstractVector, w::AbstractVector,
        C_out::Symbol=get_channel(∇)) where {F, T}
    # Batched version of SBEReducibleVertex evaluation
    C_∇ = get_channel(∇)
    nfreq = length(v1)
    nind = get_nind(∇)
    U = ∇.U(0, 0, 0, C_out)

    W = zeros(T, nind^2, nind^2)
    Λb = zeros(T, nind^2, nind^2)
    Λ = zeros(T, nind^2, nind^2)
    tmp1 = zeros(T, nind^2, nind^2)
    tmp2 = zeros(T, nind^2, nind^2)
    W_buffer = _get_vertex_coeff_buffers(∇.W)
    Λb_buffer = _get_vertex_coeff_buffers(∇.Λb)
    Λ_buffer = _get_vertex_coeff_buffers(∇.Λ)

    ∇_vvw = zeros(T, nfreq, nind^2, nind^2)
    @views for ifreq in 1:nfreq
        v1234 = frequency_to_standard(Val(F), C_out, v1[ifreq], v2[ifreq], w[ifreq])
        vvw = frequency_to_channel(Val(F), C_∇, v1234)
        _evaluate_vertex_with_buffer!(W, ∇.W, vvw..., W_buffer...)
        _evaluate_vertex_with_buffer!(Λb, ∇.Λb, vvw..., Λb_buffer...)
        _evaluate_vertex_with_buffer!(Λ, ∇.Λ, vvw..., Λ_buffer...)
        mul!(tmp2, mul!(tmp1, Λb, W), Λ)
        ∇_vvw[ifreq, :, :] .= tmp2 .- U
    end
    _permute_orbital_indices_matrix_4p_keep_dim1(C_∇, C_out, ∇_vvw, nind)
end

function to_matrix(∇::SBEReducibleVertex{F, T}, w, basis1=∇.basis_f1, basis2=∇.basis_f2, C_out::Symbol=get_channel(∇)) where {F, T}
    C_∇ = get_channel(∇)
    if C_out === C_∇ && basis1 === ∇.basis_f1 && basis2 === ∇.basis_f2
        # Same channel, same basis. Just need to contract the bosonic frequency basis.
        to_matrix(∇.Λb, w) * to_matrix(∇.W, w) * to_matrix(∇.Λ, w)
    else
        nind = get_nind(∇)
        vs1 = get_fitting_points(basis1)
        vs2 = get_fitting_points(basis2)

        v1_ = vec(ones(length(vs2))' .* vs1)
        v2_ = vec(vs2' .* ones(length(vs1)))
        w_ = fill(w, length(v1_))
        ∇_w_data = reshape(∇(v1_, v2_, w_, C_out), (length(vs1), length(vs2), nind^2, nind^2))

        ∇_tmp1 = fit_basis_coeff(∇_w_data, basis1, vs1, 1)
        ∇_tmp2 = fit_basis_coeff(∇_tmp1, basis2, vs2, 2)

        ∇_w = reshape(permutedims(∇_tmp2, (1, 3, 2, 4)), size(basis1, 2) * nind^2, size(basis2, 2) * nind^2)
        ∇_w
    end
end

function su2_apply_crossing(∇::NTuple{2, SBEReducibleVertex})
    C = get_channel(∇[1])
    C === :A || error("su2_apply_crossing implemented only for A -> T")
    U_new = apply_crossing.(getproperty.(∇, :U))
    W_new = apply_crossing.(getproperty.(∇, :W))
    @. SBEReducibleVertex(U_new, W_new, getproperty(∇, :Λb), getproperty(∇, :Λ))
end


"""
Ref: Sec. 4.3 of M. Gievers et al, Eur. Phys. J. B. 95, 108 (2022)
- ``W = U + K1``
- ``Λb = 1 + K2 * W⁻¹``
- ``Λ = 1 + W⁻¹ * K2p``
"""
function asymptotic_to_sbe_single_boson(U, K1::AbstractVertex4P{F, T}, K2, K2p) where {F, T}
    C = get_channel(K1)
    @assert get_channel(K2) == C
    @assert get_channel(K2p) == C

    norb = K1.norb
    nind = get_nind(K1)
    basis_const = K1.basis_f1

    # W = K1 + U
    W = sbe_vertex_W_add_U(K1, U)

    # Λb = 1 + K2 * W⁻¹
    vs = get_fitting_points(K2.basis_f1)
    ws  = get_fitting_points(K2.basis_b)
    Λb_data = zeros(T, length(vs), nind^2, nind^2, length(ws))
    for (iw, w) in enumerate(ws)
        inv_W = pinv(W(0, 0, w), rtol=sqrt(eps(real(T))))
        for (iv, v) in enumerate(vs)
            Λb_data[iv, :, :, iw] .= K2(v, 0, w) * inv_W
        end
    end
    Λb_minus_I = Vertex4P{F}(C, T, K2.basis_f1, basis_const, K2.basis_b, norb)
    Λb_minus_I.data .= reshape(
        fit_basis_coeff(fit_basis_coeff(Λb_data, K2.basis_f1, vs, 1), K2.basis_b, ws, 4),
        size(Λb_minus_I.data)
    )
    Λb = sbe_vertex_Λb_add_I(Λb_minus_I)

    # Λ  = 1 + W⁻¹ * K2p
    vs = get_fitting_points(K2p.basis_f2)
    ws  = get_fitting_points(K2p.basis_b)
    Λ_data = zeros(T, nind^2, length(vs), nind^2, length(ws))
    for (iw, w) in enumerate(ws)
        inv_W = pinv(W(0, 0, w), rtol=sqrt(eps(real(T))))
        for (iv, v) in enumerate(vs)
            Λ_data[:, iv, :, iw] .= inv_W * K2p(0, v, w)
        end
    end
    Λ_minus_I = Vertex4P{F}(C, T, basis_const, K2p.basis_f2, K2.basis_b, norb)
    Λ_minus_I.data .= reshape(
        fit_basis_coeff(fit_basis_coeff(Λ_data, K2p.basis_f2, vs, 2), K2p.basis_b, ws, 4),
        size(Λ_minus_I.data)
    )
    Λ = sbe_vertex_Λ_add_I(Λ_minus_I)

    SBEReducibleVertex(U, W, Λb, Λ)
end

"""
Ref: Sec. 4.3 of M. Gievers et al, Eur. Phys. J. B. 95, 108 (2022)
- ``M = K3 - K2 * W⁻¹ * K2p``
"""
function asymptotic_to_sbe_multi_boson(U, K1::AbstractVertex4P{F, T}, K2, K2p, K3) where {F, T}
    C = get_channel(K1)
    get_channel(K2) === C || throw(ArgumentError("Channel mismatch between K1 and K2"))
    get_channel(K2p) === C || throw(ArgumentError("Channel mismatch between K1 and K2p"))
    get_channel(K3) === C || throw(ArgumentError("Channel mismatch between K1 and K3"))
    nind = get_nind(K1)
    # M = K3 - K2 * (K1 + U)⁻¹ * K2p
    # M_subtract_data[v1, v2, w] = K2[v1, w] * W[w]⁻¹ * K2p[v2, w]
    M = similar(K3)
    vs1 = get_fitting_points(M.basis_f1)
    vs2 = get_fitting_points(M.basis_f2)
    ws  = get_fitting_points(M.basis_b)
    M_subtract_data = zeros(T, length(vs1), nind^2, length(vs2), nind^2, length(ws))
    K2_data = zeros(T, nind^2, nind^2, length(vs1))
    K2p_data = zeros(T, nind^2, nind^2, length(vs2))
    @views for (iw, w) in enumerate(ws)
        inv_W = pinv(K1(0, 0, w) + U(0, 0, w, C), rtol=sqrt(eps(real(T))))
        for (iv1, v1) in enumerate(vs1)
            K2_data[:, :, iv1] .= K2(v1, 0, w)
        end
        for (iv2, v2) in enumerate(vs2)
            K2p_data[:, :, iv2] .= K2p(0, v2, w)
        end
        for iv2 in axes(vs2, 1), iv1 in axes(vs1, 1)
            M_subtract_data[iv1, :, iv2, :, iw] .= K2_data[:, :, iv1] * inv_W * K2p_data[:, :, iv2]
        end
    end

    tmp1 = fit_basis_coeff(M_subtract_data, M.basis_f1, vs1, 1)
    tmp2 = fit_basis_coeff(tmp1, M.basis_f2, vs2, 3)
    M.data .= K3.data .- reshape(fit_basis_coeff(tmp2, M.basis_b, ws, 5), size(M.data))
    M
end

function sbe_vertex_W_add_U(W_minus_U::Vertex4P{F, T}, U) where {F, T}
    C = get_channel(W_minus_U)
    basis_b = concat_constant_basis(W_minus_U.basis_b)
    W = Vertex4P{F}(C, T, W_minus_U.basis_f1, W_minus_U.basis_f2, basis_b, W_minus_U.norb)
    W.data[:, :, 1] .= U(0, 0, 0, C)
    W.data[:, :, 2:end] .= W_minus_U.data
    W
end

function sbe_vertex_Λb_add_I(Λb_minus_I::Vertex4P{F, T}) where {F, T}
    # Concatenate bases of Λb_minus_I and basis_const, store Λb_minus_I and 1 in the corresponding blocks
    C = get_channel(Λb_minus_I)
    basis_f1 = concat_constant_basis(Λb_minus_I.basis_f1)
    basis_f2 = Λb_minus_I.basis_f2
    basis_b  = concat_constant_basis(Λb_minus_I.basis_b)
    Λb = Vertex4P{F}(C, T, basis_f1, basis_f2, basis_b, Λb_minus_I.norb)

    nind = get_nind(Λb)
    Λb_data_reshape = reshape(Λb.data, nb_f1(Λb), nind^2, 1, nind^2, nb_b(Λb))
    @views Λb_data_for_Λb_minus_I = Λb_data_reshape[2:end, :, 1, :, 2:end]
    @views Λb_data_for_I = Λb_data_reshape[1, :, 1, :, 1]
    Λb_data_for_Λb_minus_I .= reshape(Λb_minus_I.data, size(Λb_data_for_Λb_minus_I))
    Λb_data_for_I .= I(nind^2)
    Λb
end

function sbe_vertex_Λ_add_I(Λ_minus_I::Vertex4P{F, T}) where {F, T}
    # Concatenate bases of Λ_minus_I and basis_const, store Λ_minus_I and 1 in the corresponding blocks
    C = get_channel(Λ_minus_I)
    basis_f1 = Λ_minus_I.basis_f1
    basis_f2 = concat_constant_basis(Λ_minus_I.basis_f2)
    basis_b  = concat_constant_basis(Λ_minus_I.basis_b)
    Λ = Vertex4P{F}(C, T, basis_f1, basis_f2, basis_b, Λ_minus_I.norb)

    nind = get_nind(Λ)
    Λ_data_reshape = reshape(Λ.data, nb_f1(Λ), nind^2, nb_f2(Λ), nind^2, nb_b(Λ))
    @views Λ_data_for_Λ_minus_I = Λ_data_reshape[1, :, 2:end, :, 2:end]
    @views Λ_data_for_I = Λ_data_reshape[1, :, 1, :, 1]
    Λ_data_for_Λ_minus_I .= reshape(Λ_minus_I.data, size(Λ_data_for_Λ_minus_I))
    Λ_data_for_I .= I(nind^2)
    Λ
end

function sbe_3p_identity(U::AbstractVertex4P{F, T}) where {F, T}
    C = get_channel(U)
    basis_const = U.basis_f1
    id  = Vertex4P{F}(C, T, basis_const, basis_const, basis_const, U.norb)
    nind = get_nind(id)
    reshape(id.data, nind^2, nind^2) .= I(nind^2)
    id
end

function sbe_bare_reducible_vertex(U)
    SBEReducibleVertex(U, U, sbe_3p_identity(U), sbe_3p_identity(U))
end

"""
    sbe_get_WΛ(∇::SBEReducibleVertex) => SBEReducibleVertex
Return a `SBEReducibleVertex` with `Λ` replaced by the identity 3-point vertex: `∇ = Λb W`.
"""
sbe_get_ΛW(∇::SBEReducibleVertex) = SBEReducibleVertex(∇.U, ∇.W, ∇.Λb, sbe_3p_identity(∇.U))

"""
    sbe_get_WΛ(∇::SBEReducibleVertex) => SBEReducibleVertex
Return a `SBEReducibleVertex` with `Λb` replaced by the identity 3-point vertex: `∇ = W Λ`.
"""
sbe_get_WΛ(∇::SBEReducibleVertex) = SBEReducibleVertex(∇.U, ∇.W, sbe_3p_identity(∇.U), ∇.Λ)


"""
    SBEVertex
4-point vertex in the single-boson exchange decomposition.
"""
Base.@kwdef struct SBEVertexX5X{F, T} <: AbstractFrequencyVertex{F, T}
    ∇_A
    ∇_P
    ∇_T
    M_A = nothing
    M_P = nothing
    M_T = nothing
    # FIXME: remove basis2
    basis_k1_b = ∇_A[1].W.basis_b.basis2
    basis_k2_f = ∇_A[1].Λb.basis_f1.basis2
    basis_k2_b = ∇_A[1].Λb.basis_b.basis2
    basis_k3_f = M_A === nothing ? nothing : get_fermionic_basis_1(M_A[1])
    basis_k3_b = M_A === nothing ? nothing : get_bosonic_basis(M_A[1])
end

function asymptotic_to_sbe(Γ::AsymptoticVertex{F, T}) where {F, T}
    ∇_A = asymptotic_to_sbe_single_boson.(Γ.Γ0_A, Γ.K1_A, Γ.K2_A, Γ.K2p_A)
    M_A = asymptotic_to_sbe_multi_boson.(Γ.Γ0_A, Γ.K1_A, Γ.K2_A, Γ.K2p_A, Γ.K3_A)
    ∇_P = asymptotic_to_sbe_single_boson.(Γ.Γ0_P, Γ.K1_P, Γ.K2_P, Γ.K2p_P)
    M_P = asymptotic_to_sbe_multi_boson.(Γ.Γ0_P, Γ.K1_P, Γ.K2_P, Γ.K2p_P, Γ.K3_P)
    ∇_T = asymptotic_to_sbe_single_boson.(Γ.Γ0_T, Γ.K1_T, Γ.K2_T, Γ.K2p_T)
    M_T = asymptotic_to_sbe_multi_boson.(Γ.Γ0_T, Γ.K1_T, Γ.K2_T, Γ.K2p_T, Γ.K3_T)
    SBEVertexX5X{F, T}(; ∇_A, ∇_P, ∇_T, M_A, M_P, M_T)
end

_vertex_names(::SBEVertexX5X) = (:∇_A, :∇_P, :∇_T, :M_A, :M_P, :M_T)

get_vertices(Γ) = filter!(!isnothing, [getproperty(Γ, n) for n in _vertex_names(Γ)])

function get_irreducible_vertices(C, Γ::SBEVertexX5X)
    # FIXME: Currently SU2 is assumed.
    # FIXME: SU2 spin conversion is done in iterate_parquet_single_channel_sbe
    # su2_convert_spin_channel.(C, filter!(x -> channel(x[1]) != C, get_vertices(Γ)))
    filter!(x -> get_channel(x[1]) != C, get_vertices(Γ))
end

function get_reducible_vertices(C, Γ::SBEVertexX5X)
    C === :A && return (; ∇=Γ.∇_A, M=Γ.M_A)
    C === :P && return (; ∇=Γ.∇_P, M=Γ.M_P)
    C === :T && return (; ∇=Γ.∇_T, M=Γ.M_T)
    throw(ArgumentError("Wrong channel $C"))
end
