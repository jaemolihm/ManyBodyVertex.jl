using OMEinsum

"""
    self_energy_hartree(U, G, temperature)
Compute the Hartree self-energy. Given the Hamiltonian, we shift the chemical potential
so that the Hartree self-energy is zero if all orbitals are half filled.
- `U`: Bare vertex in A channel
- `G`: Green function
- `temperature`: temperature, needed only for MF.
"""
function self_energy_hartree(U, G, temperature)
    get_formalism(G) === :MF || error("Only MF implemented yet")
    channel(U) === :A || error("Channel of the bare vertex must be :A")
    nind = get_nind(U)
    Uarr = reshape(U(0, 0, 0), nind, nind, nind, nind)
    n = compute_occupation_matrix(G, temperature) .- I(G.norb) / 2
    @ein Σ_H[a, d] := Uarr[a, b, c, d] * n[b, c]
    Σ_H = Σ_H::Matrix{eltype(G)} .* -1
    # Impose Hermiticity
    Matrix(Hermitian(Σ_H))
end

function self_energy_hartree_SU2(U, G, temperature)
    (  self_energy_hartree(U[1], G, temperature) .* 1/2
    .+ self_energy_hartree(U[2], G, temperature) .* 3/2)
end
