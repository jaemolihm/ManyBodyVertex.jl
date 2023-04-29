"""
    SBEReducibleVertex(U, W, Λb, Λ) => ∇
Vertex in the single-boson exchange (SBE) decomposition.
We define ``∇(v1, v2, w) = Λb(v1, w) * W(w) * Λ(v2, w) - U``.
"""
mutable struct SBEReducibleVertex{F, C, T, VT1, VT2, VT3, VT4} <: AbstractVertex4P{F, C, T}
    # Number of orbitals
    norb::Int
    U::VT1
    W::VT2
    Λb::VT3
    Λ::VT4
    function SBEReducibleVertex(U, W::AbstractVertex4P{F, C, T}, Λb, Λ) where {F, C, T}
        new{F, C, T, typeof(U), typeof(W), typeof(Λb), typeof(Λ)}(U.norb, U, W, Λb, Λ)
    end
end

"""
    (∇::SBEReducibleVertex{F, C_∇})(v1, v2, w, c_out::Val=Val(C_∇)) where {F, C_∇}
Evaluate the SBE vertex at given frequencies in the channel parametrization.
"""
function (∇::SBEReducibleVertex{F, C_∇})(v1, v2, w, c_out::Val=Val(C_∇)) where {F, C_∇}
    v1234 = frequency_to_standard(Val(F), c_out, v1, v2, w)
    vvw = frequency_to_channel(Val(F), Val(C_∇), v1234)
    U  = ∇.U(vvw...)
    W  = ∇.W(vvw...)
    Λb = ∇.Λb(vvw...)
    Λ  = ∇.Λ(vvw...)
    ∇_vvw = Λb * W * Λ .- U
    _permute_orbital_indices_matrix_4p(Val(C_∇), c_out, ∇_vvw, get_nind(∇))
end

function (∇::SBEReducibleVertex{F, C_∇, T})(v1::AbstractVector, v2::AbstractVector, w::AbstractVector,
        c_out::Val=Val(C_∇)) where {F, C_∇, T}
    # Batched version of SBEReducibleVertex evaluation
    nfreq = length(v1)
    nind = get_nind(∇)
    U = ∇.U(zero(eltype(v1)), zero(eltype(v2)), zero(eltype(w)), c_out)

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
        v1234 = frequency_to_standard(Val(F), c_out, v1[ifreq], v2[ifreq], w[ifreq])
        vvw = frequency_to_channel(Val(F), Val(C_∇), v1234)
        _evaluate_vertex_with_buffer!(W, ∇.W, vvw..., W_buffer...)
        _evaluate_vertex_with_buffer!(Λb, ∇.Λb, vvw..., Λb_buffer...)
        _evaluate_vertex_with_buffer!(Λ, ∇.Λ, vvw..., Λ_buffer...)
        mul!(tmp2, mul!(tmp1, Λb, W), Λ)
        ∇_vvw[ifreq, :, :] .= tmp2 .- U
    end
    _permute_orbital_indices_matrix_4p_keep_dim1(Val(C_∇), c_out, ∇_vvw, nind)
end

function to_matrix(∇::SBEReducibleVertex{F, C, T}, w, basis1=∇.basis_f1, basis2=∇.basis_f2, c::Val=Val(C)) where {F, C, T}
    nind = get_nind(∇)
    vs1 = get_fitting_points(basis1)
    vs2 = get_fitting_points(basis2)

    v1_ = vec(ones(length(vs2))' .* vs1)
    v2_ = vec(vs2' .* ones(length(vs1)))
    w_ = fill(w, length(v1_))
    ∇_w_data = reshape(∇(v1_, v2_, w_, c), (length(vs1), length(vs2), nind^2, nind^2))

    ∇_tmp1 = fit_basis_coeff(∇_w_data, basis1, vs1, 1)
    ∇_tmp2 = fit_basis_coeff(∇_tmp1, basis2, vs2, 2)

    ∇_w = reshape(permutedims(∇_tmp2, (1, 3, 2, 4)), size(basis1, 2) * nind^2, size(basis2, 2) * nind^2)
    ∇_w
end


"""
Ref: Sec. 4.3 of M. Gievers et al, Eur. Phys. J. B. 95, 108 (2022)
- ``W = U + K1``
- ``Λb = 1 + K2 * W⁻¹``
- ``Λ = 1 + W⁻¹ * K2p``
- ``M = K3 - K2 * W⁻¹ * K2p``
"""
function asymptotic_to_sbe(U, K1::AbstractVertex4P{F, C, T}, K2, K2p, K3) where {F, C, T}
    @assert channel(U) == C
    @assert channel(K2) == C
    @assert channel(K2p) == C
    @assert channel(K3) == C

    norb = K1.norb
    nind = get_nind(K1)
    basis_const = K1.basis_f1

    # W = K1 + U
    W = Vertex4P{F, C}(T, basis_const, basis_const, concat_constant_basis(K1.basis_b), norb)
    W.data[:, :, 1] .= U.data
    W.data[:, :, 2:end] .= K1.data

    # Λb = 1 + K2 * W⁻¹
    vs = get_fitting_points(K2.basis_f1)
    ws  = get_fitting_points(K2.basis_b)
    Λb_data = zeros(T, length(vs), nind^2, nind^2, length(ws))
    for (iw, w) in enumerate(ws)
        inv_W = pinv(W(0, 0, w), rtol=sqrt(eps(real(eltype(W)))))
        for (iv, v) in enumerate(vs)
            Λb_data[iv, :, :, iw] .= K2(v, 0, w) * inv_W
        end
    end
    Λb_minus_1 = fit_basis_coeff(fit_basis_coeff(Λb_data, K2.basis_f1, vs, 1), K2.basis_b, ws, 4)

    # Concatenate bases of K2 and basis_const, store Λb_minus_1 and 1 in the corresponding blocks
    Λb = Vertex4P{F, C}(T, concat_constant_basis(K2.basis_f1), basis_const, concat_constant_basis(K2.basis_b), norb)
    Λb_data_reshape = reshape(Λb.data, nb_f1(Λb), nind^2, 1, nind^2, nb_b(Λb))
    Λb_data_reshape[2:end, :, 1, :, 2:end] .= Λb_minus_1
    Λb_data_reshape[1,     :, 1, :, 1    ] .= I(nind^2)

    # Λ  = 1 + W⁻¹ * K2p
    vs = get_fitting_points(K2p.basis_f2)
    ws  = get_fitting_points(K2p.basis_b)
    Λ_data = zeros(T, nind^2, length(vs), nind^2, length(ws))
    for (iw, w) in enumerate(ws)
        inv_W = pinv(W(0, 0, w), rtol=sqrt(eps(real(eltype(W)))))
        for (iv, v) in enumerate(vs)
            Λ_data[:, iv, :, iw] .= inv_W * K2p(0, v, w)
        end
    end
    Λ_minus_1 = fit_basis_coeff(fit_basis_coeff(Λ_data, K2p.basis_f2, vs, 2), K2p.basis_b, ws, 4)

    # Concatenate bases of K2p and basis_const, store Λ_minus_1 and 1 in the corresponding blocks
    Λ  = Vertex4P{F, C}(T, basis_const, concat_constant_basis(K2p.basis_f2), concat_constant_basis(K2p.basis_b), norb)
    Λ_data_reshape = reshape(Λ.data, 1, nind^2, nb_f2(Λ), nind^2, nb_b(Λ))
    Λ_data_reshape[1, :, 2:end, :, 2:end] .= Λ_minus_1
    Λ_data_reshape[1, :, 1,     :, 1    ] .= I(nind^2)

    ∇ = SBEReducibleVertex(U, W, Λb, Λ)

    # M = K3 - K2 * W⁻¹ * K2p
    # M_subtract_data[v1, v2, w] = K2[v1, w] * W[w]⁻¹ * K2p[v2, w]
    M = similar(K3)
    vs1 = get_fitting_points(M.basis_f1)
    vs2 = get_fitting_points(M.basis_f2)
    ws  = get_fitting_points(M.basis_b)
    M_subtract_data = zeros(T, length(vs1), nind^2, length(vs2), nind^2, length(ws))
    K2_data = zeros(T, nind^2, nind^2, length(vs1))
    K2p_data = zeros(T, nind^2, nind^2, length(vs2))
    @views for (iw, w) in enumerate(ws)
        inv_W = pinv(W(0, 0, w), rtol=sqrt(eps(real(eltype(W)))))
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

    (; ∇, M)
end
