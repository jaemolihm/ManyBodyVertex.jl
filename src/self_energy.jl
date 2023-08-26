using OMEinsum

function _compute_self_energy(Γ, G, v, overlap=nothing; temperature=nothing)
    F = get_formalism(Γ)
    C = channel(Γ)
    nind = get_nind(G)
    nb_f1(Γ) == 1 || error("Γ.basis_f1 must be a constant basis")
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")
    if overlap === nothing
        overlap = basis_integral_self_energy(Γ.basis_f2, Γ.basis_b, G.basis, v, Val(C))
    end
    # (ab, icd, j) -> (a, b, c, d, ij)
    Γ_data = reshape(permutedims(reshape(Γ.data, (nind^2, nb_f2(Γ), nind^2, nb_b(Γ))), (1, 3, 2, 4)), nind^4, :)
    # (a, b, c, d, ij) * (ij, k) -> (a, b, c, d, k)
    Γ_data_2 = reshape(Γ_data * reshape(overlap, (:, nbasis(G.basis))), nind, nind, nind, nind, nbasis(G.basis))

    Σ_v = zeros(eltype(G), nind, nind)
    @views for b in 1:nind, d in 1:nind, c in 1:nind, a in 1:nind
        if C === :P
            Σ_v[a, d] += transpose(Γ_data_2[a, b, c, d, :]) * G.data[c, b, :]
        else
            Σ_v[a, d] += transpose(Γ_data_2[a, b, c, d, :]) * G.data[b, c, :]
        end
    end
    Σ_v .*= (integral_coeff(Val(F), temperature) * -1 / 2)
    Σ_v
end

function SU2_self_energy_coeff(::Val{C}) where {C}
    # For the A channel, the factor 2 accounts for the contribution of the T channel.
    # For the P channel, the factor 2 account for the factor of 1/2 in the bubble.
    if C === :A
        (1/2, 3/2) .* 2
    elseif C === :P
        (-1/2, 3/2) .* 2
    elseif C === :T
        error("Do not use the T-channel vertices for self-energy calculation under SU2
            symmetry. Use crossing symmetry and A-channel vertices instead.")
    else
        error("Wrong channel $C")
    end
end

function _compute_self_energy_SU2(Γs, G, basis; temperature=nothing)
    F = get_formalism(G)
    nind = get_nind(G)
    vs = get_fitting_points(basis.freq)
    Σ_data_iv = zeros(ComplexF64, nind, nind, length(vs))

    Base.Threads.@threads for iv in eachindex(vs)
        v = vs[iv]
        for Γ in Γs
            C = channel(Γ[1])
            overlap = basis_integral_self_energy(Γ[1].basis_f2, Γ[1].basis_b, G.basis, v, C)
            coeff = SU2_self_energy_coeff(C)
            Σ_data_iv[:, :, iv] .+= _compute_self_energy(Γ[1], G, v, overlap; temperature) .* coeff[1]
            Σ_data_iv[:, :, iv] .+= _compute_self_energy(Γ[2], G, v, overlap; temperature) .* coeff[2]
        end
    end
    Green2P{F}(basis.freq, 1, mfRG.fit_basis_coeff(Σ_data_iv, basis.freq, vs, 3))
end

"""
    compute_self_energy_SU2(Γ, G, ΠA, ΠP, basis=G.basis; temperature=nothing, exclude_UU=false)
Compute the self-energy by solving the Schwinger-Dyson equation.
``Σ[a, d](v) = ∫ dw K1[ab, cd](v+w/2, w) * G[b, c](v+w) * integral_coeff * -1/2``

# Input
- `exclude_UU=false`: if set to `true`, skip the UΠU term. Used for parquet without
    irreducible vertex.
"""
function compute_self_energy_SU2(Γ, G, ΠA, ΠP, basis=get_basis(G); temperature=nothing,
                                 exclude_UU=false)
    F = get_formalism(Γ)
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")

    # If G is lazily defined, compute G on the basis explicitly.
    G_ = green_lazy_to_explicit(G, basis)

    # U_Π_U term (the O(U²) term): to avoid double counting, multiply 1/3 to each channel.
    # A, T channel: (1/3 + 1/3 [A + T]) / 2 (multiplied in _compute_self_energy_SU2) = 1/3
    # P channel: (1/3 * 2 [_bubble_prefactor 1/2 in ΠP]) / 2 (multiplied in
    #            _compute_self_energy_SU2) = 1/3
    U_ΠA_U = vertex_bubble_integral.(Γ.Γ0_A, ΠA, Γ.Γ0_A, Ref(Γ.basis_k1_b)) .* (1/3)
    U_ΠA_K1K2 = _mapreduce_bubble_integrals([Γ.Γ0_A], ΠA, [Γ.K1_A, Γ.K2_A], Γ.basis_k1_b)
    U_ΠA_K2pK3 = _mapreduce_bubble_integrals([Γ.Γ0_A], ΠA, [Γ.K2p_A, Γ.K3_A], Γ.basis_k2_b)
    if Γ.Λ_A !== nothing
        U_ΠA_Λ = _mapreduce_bubble_integrals([Γ.Γ0_A], ΠA, [Γ.Λ_A], Γ.basis_k2_b) .* (1/3)
    else
        U_ΠA_Λ = nothing
    end

    U_ΠP_U = vertex_bubble_integral.(Γ.Γ0_P, ΠP, Γ.Γ0_P, Ref(Γ.basis_k1_b)) .* (1/3)
    U_ΠP_K1K2 = _mapreduce_bubble_integrals([Γ.Γ0_P], ΠP, [Γ.K1_P, Γ.K2_P], Γ.basis_k1_b)
    U_ΠP_K2pK3 = _mapreduce_bubble_integrals([Γ.Γ0_P], ΠP, [Γ.K2p_P, Γ.K3_P], Γ.basis_k2_b)
    if Γ.Λ_P !== nothing
        U_ΠP_Λ = _mapreduce_bubble_integrals([Γ.Γ0_P], ΠP, [Γ.Λ_P], Γ.basis_k2_b) .* (1/3)
    else
        U_ΠP_Λ = nothing
    end

    vertices_use = []
    push!(vertices_use, U_ΠA_K1K2, U_ΠA_K2pK3, U_ΠP_K1K2, U_ΠP_K2pK3, U_ΠA_Λ, U_ΠP_Λ)
    if !exclude_UU
        push!(vertices_use, U_ΠA_U, U_ΠP_U)
    end
    filter!(!isnothing, vertices_use)

    Σ = _compute_self_energy_SU2(vertices_use, G_, basis; temperature)

    # Add Hartree self-energy
    if !exclude_UU
        Σ.offset .= self_energy_hartree_SU2(Γ.Γ0_A, G_, temperature)
    end

    Σ
end


"""
    self_energy_hartree(U, G, temperature)
Compute the Hartree self-energy. Given the Hamiltonian, we shift the chemical potential
so that the Hartree self-energy is zero if all orbitals are half filled.
- `U`: Bare vertex in A channel
- `G`: Green function
- `temperature`: temperature, used only for MF.
"""
function self_energy_hartree(U, G, temperature)
    channel(U) === Val(:A) || error("Channel of the bare vertex must be :A")
    nind = get_nind(U)
    Uarr = reshape(U(0, 0, 0), nind, nind, nind, nind)
    n = compute_occupation_matrix(G, temperature) .- I(G.norb) / 2
    F = get_formalism(G)

    if F === :MF
        @ein Σ_H[a, d] := Uarr[a, b, c, d] * n[b, c]
        Σ_H = Σ_H::Matrix{eltype(G)} .* -1
    elseif F === :KF
        # Multiply 2 which was divided in compute_occupation_matrix.
        n .*= 2

        # Use only the (1, 2, 2, 2) component of the bare vertex
        Uarr_ = reshape(Uarr, G.norb, 2, G.norb, 2, G.norb, 2, G.norb, 2)
        Uarr_use = Uarr_[:, 1, :, 2, :, 2, :, 2]
        @ein Σ_H_orbital[a, d] := Uarr_use[a, b, c, d] * n[b, c]
        Σ_H_orbital = Σ_H_orbital::Matrix{eltype(G)} .* -1

        # Let only the (k1, k2) = (1, 2) and (2, 1) components of Σ_H be nonzero:
        # Σ_H[i, k1, j, k2] = Σ_H_[i, j] * X[k1, k2]
        Σ_H = zeros(eltype(G), nind, nind)
        Σ_H_reshape = Base.ReshapedArray(Σ_H, (G.norb, 2, G.norb, 2), ())
        Σ_H_reshape[:, 1, :, 2] .= Σ_H_orbital
        Σ_H_reshape[:, 2, :, 1] .= Σ_H_orbital
    else
        Σ_H = zeros(eltype(G), nind, nind)
        # error("Wrong formalism $F. Must be :KF or :MF.")
    end
    # Impose Hermiticity
    (Σ_H .+ Σ_H') ./ 2
end

function self_energy_hartree_SU2(U, G, temperature)
    (  self_energy_hartree(U[1], G, temperature) .* 1/2
    .+ self_energy_hartree(U[2], G, temperature) .* 3/2)
end
