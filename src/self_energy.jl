using OMEinsum

"""
    self_energy_hartree(U, G, temperature)
Compute the Hartree self-energy. Given the Hamiltonian, we shift the chemical potential
so that the Hartree self-energy is zero if all orbitals are half filled.
- `U`: Bare vertex in A channel
- `G`: Green function
- `temperature`: temperature, used only for MF.
"""
function self_energy_hartree(U, G, temperature)
    channel(U) === :A || error("Channel of the bare vertex must be :A")
    nind = get_nind(U)
    Uarr = reshape(U(0, 0, 0), nind, nind, nind, nind)
    n = compute_occupation_matrix(G, temperature) .- I(G.norb) / 2

    if get_formalism(G) === :MF
        @ein Σ_H[a, d] := Uarr[a, b, c, d] * n[b, c]
        Σ_H = Σ_H::Matrix{eltype(G)} .* -1
    elseif get_formalism(G) === :KF
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
        error("Wrong formalism $F. Must be :KF or :MF.")
    end
    # Impose Hermiticity
    (Σ_H .+ Σ_H') ./ 2
end

function self_energy_hartree_SU2(U, G, temperature)
    (  self_energy_hartree(U[1], G, temperature) .* 1/2
    .+ self_energy_hartree(U[2], G, temperature) .* 3/2)
end
