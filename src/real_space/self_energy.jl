function _compute_self_energy_SU2(Γs, G::RealSpaceGreen2P{F}, basis; temperature=nothing) where {F}
    nind = get_nind(G)
    vs = get_fitting_points(basis.freq)
    Σ = RealSpaceGreen2P{F}(eltype(G), basis.freq, basis.r, G.norb)

    Σ_data_iv_iR = [zeros(eltype(G), nind, nind, length(vs), length(Rs)) for Rs in basis.r.R_replicas]

    for Γ in Γs
        # Function barrier
        _compute_self_energy_SU2_single_vertex!(Σ_data_iv_iR, Γ, G, vs, Σ.rbasis, temperature)
    end

    for iatm2 in 1:Σ.rbasis.natom, iatm1 in 1:Σ.rbasis.natom
        # Divide by R_ndegen (comes from Fourier transform k -> R)
        for (iR, ndegen) in enumerate(Σ.rbasis.R_ndegen[iatm1, iatm2])
            Σ_data_iv_iR[iatm1, iatm2][:, :, :, iR] ./= ndegen
        end
        Σ.data[iatm1, iatm2] .= fit_basis_coeff(Σ_data_iv_iR[iatm1, iatm2], basis.freq, vs, 3)
    end
    Σ
end

function _compute_self_energy_SU2_single_vertex!(Σ_data_iv_iR, Γ, G::RealSpaceGreen2P, vs, rbasis, temperature)
    C = get_channel(Γ[1])
    Threads.@threads for iv in eachindex(vs)
        v = vs[iv]
        overlap = basis_integral_self_energy(Γ[1].basis_f2, Γ[1].basis_b, G.basis, v, C)
        coeff = SU2_self_energy_coeff(C)
        for bR in Γ[1].rbasis.bonds_R, bL in Γ[1].rbasis.bonds_L
            iatm1, iatm2 = bL[1], bR[1]
            iatmG1, iatmG2 = (C === :A || C === :T) ? (bL[2], bR[2]) : (bR[2], bL[2])

            Rs = rbasis.R_replicas[iatm1, iatm2]
            for iR in eachindex(Rs)
                R = Rs[iR]
                Γ_R = (Γ[1][bL, bR, -R], Γ[2][bL, bR, -R])
                Γ_R === (nothing, nothing) && continue

                R_G = (C === :A || C === :T) ? bR[3] + R : -bR[3] - R
                G_R = G[iatmG1, iatmG2, R_G]
                G_R === nothing && continue

                Σ_data_iv_iR[iatm1, iatm2][:, :, iv, iR] .+= _compute_self_energy(
                        Γ_R[1], G_R, v, overlap; temperature) .* coeff[1]
                Σ_data_iv_iR[iatm1, iatm2][:, :, iv, iR] .+= _compute_self_energy(
                        Γ_R[2], G_R, v, overlap; temperature) .* coeff[2]
            end
        end
    end
    Σ_data_iv_iR
end

function self_energy_hartree(U, G::RealSpaceGreen2P, temperature)
    get_channel(U) === :A || error("Channel of the bare vertex must be :A")
    nind = get_nind(G)
    Σ_H_data = [zeros(eltype(G), nind, nind, length(Rs)) for Rs in G.rbasis.R_replicas]
    for bR in U.rbasis.bonds_R, bL in U.rbasis.bonds_L
        iatm1, iatm2 = bL[1], bR[1]
        iatmG1, iatmG2 = bL[2], bR[2]
        Rs = G.rbasis.R_replicas[iatm1, iatm2]

        for iR in eachindex(Rs)
            R = Rs[iR]
            U_R = U[bL, bR, -R]
            U_R === nothing && continue

            R_G = bR[3] + R
            G_R = G[iatmG1, iatmG2, R_G]
            G_R === nothing && continue

            Σ_H_data[iatm1, iatm2][:, :, iR] .+= self_energy_hartree(U_R, G_R, temperature)
        end
    end
    Σ_H_data
end
