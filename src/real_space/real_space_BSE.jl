"""
# Inputs
-`basis_b = (; freq=basis_w, r=rbasis)`. `basis_w` is the basis for the bosonic frequency,
    and `rbasis.qgrid` is the bosonic momentum grid to use for the output vertex.
"""
function vertex_bubble_integral(
        ΓL::RealSpaceVertex{F},
        Π::RealSpaceBubble{F},
        ΓR::RealSpaceVertex{F},
        basis_b;
        basis_aux=nothing
    ) where {F}

    RC = real_space_channel(Π)
    RC === real_space_channel(ΓL) || throw(ArgumentError("Real-space channel mismatch between Π and ΓL"))
    RC === real_space_channel(ΓR) || throw(ArgumentError("Real-space channel mismatch between Π and ΓR"))

    basis_w = basis_b.freq
    qgrid = basis_b.r.qgrid
    CB = get_channel(Π)

    rbasis = RealSpaceBasis(ΓL.rbasis.lattice, ΓL.rbasis.positions, ΓL.rbasis.bonds_L,
                               ΓR.rbasis.bonds_R, qgrid)

    basis_f1 = get_channel(ΓL) == CB ? ΓL.basis_f1 : basis_aux.freq
    basis_f2 = get_channel(ΓR) == CB ? ΓR.basis_f2 : basis_aux.freq
    norb = ΓL.norb
    nq = length(rbasis.qpts)

    Γ_iq = Vertex4P{F}(CB, eltype(Π), basis_f1, basis_f2, basis_w, norb)
    Γ = RealSpaceVertex(RC, rbasis, typeof(Γ_iq))

    # TODO: Precompute interpolate_to_q?
    # TODO: Use block matrix type?

    basis_Π_L = get_channel(ΓL) == CB ? ΓL.basis_f2 : basis_aux.freq
    basis_Π_R = get_channel(ΓR) == CB ? ΓR.basis_f1 : basis_aux.freq
    @timeit timer "overlap" cache_and_load_overlaps(Π, basis_Π_L, basis_Π_R)

    # TODO: Threads
    for iq in eachindex(rbasis.qpts)
        xq = rbasis.qpts[iq]
        for (ibL, bL) in enumerate(rbasis.bonds_L)
            for (ibR, bR) in enumerate(rbasis.bonds_R)
                Γ_iq.data .= 0

                # Calculate bubble integral, sum over intermediate bonds.
                for bpL in ΓL.rbasis.bonds_R
                    ΓL_qb = interpolate_to_q(ΓL, xq, bL, bpL)
                    ΓL_qb === nothing && continue
                    for bpR in ΓR.rbasis.bonds_L
                        ΓR_qb = interpolate_to_q(ΓR, xq, bpR, bR)
                        ΓR_qb === nothing && continue
                        Π_qb = interpolate_to_q(Π, xq, bpL, bpR)
                        Π_qb === nothing && continue
                        @timeit timer "bubble_int" begin
                            Γ_iq.data .+= vertex_bubble_integral(ΓL_qb, Π_qb, ΓR_qb, basis_w; basis_aux).data
                        end
                    end
                end

                # Fouirer transform xq -> R_B
                @timeit timer "q_to_R" for (iR_B, R_B) in enumerate(Γ.rbasis.R_B_replicas[ibL, ibR])
                    if !haskey(Γ.vertices_R, (ibL, ibR, iR_B))
                        insert!(Γ.vertices_R, (ibL, ibR, iR_B), zero(Γ_iq))
                    end
                    R, Rp = bL[3], bR[3]
                    ndegen = Γ.rbasis.R_B_ndegen[ibL, ibR][iR_B]
                    coeff = cispi(-2 * xq' * (R_B + (R - Rp) / 2)) / nq / ndegen
                    Γ.vertices_R[(ibL, ibR, iR_B)].data .+= Γ_iq.data .* coeff
                end
            end
        end
    end
    Γ
end
