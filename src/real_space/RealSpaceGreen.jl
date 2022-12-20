# TODOs
# [ ] Optimize by FFT
# [ ] Multiple atoms

struct RealSpaceGreen2P{F, T, BT, RBT <: RealSpaceBasis2P} <: AbstractFrequencyVertex{F, T}
    basis::BT
    rbasis::RBT
    norb::Int
    data::Matrix{Array{T, 4}}  # [iatm1, iatm2][i1, i2, ibasis, iR]
    offset::Matrix{Array{T, 3}}  # [iatm1, iatm2][i1, i2, iR]
    function RealSpaceGreen2P{F}(basis, rbasis, norb, data) where {F}
        T = eltype(eltype(data))
        nind = norb * nkeldysh(F)
        offset = [zeros(T, nind, nind, length(rs)) for rs in rbasis.R_replicas]
        new{F, T, typeof(basis), typeof(rbasis)}(basis, rbasis, norb, data, offset)
    end
end

function RealSpaceGreen2P{F}(::Type{T}, basis, rbasis, norb=1) where {F, T}
    nb = nbasis(basis)
    nk = nkeldysh(F)
    data = [zeros(T, norb * nk, norb * nk, nb, length(rs)) for rs in rbasis.R_replicas]
    RealSpaceGreen2P{F}(basis, rbasis, norb, data)
end
RealSpaceGreen2P{F}(basis, rbasis, norb=1) where {F} = RealSpaceGreen2P{F}(ComplexF64, basis, rbasis, norb)

get_basis(G::RealSpaceGreen2P) = (; freq=G.basis, r=G.rbasis)
green_lazy_to_explicit(G::RealSpaceGreen2P, basis) = G

function green_lazy_to_explicit(G_lazy::AbstractLazyGreen2P, basis::NamedTuple{(:freq, :r)})
    basis.r.natom > 1 && error("multiple atoms not implemented")
    F = get_formalism(G_lazy)
    nind = get_nind(G_lazy)

    G = RealSpaceGreen2P{F}(eltype(G_lazy), basis.freq, basis.r, G_lazy.norb);

    vs = get_fitting_points(basis.freq)
    Gq_data = zeros(eltype(G_lazy), nind, nind, length(vs))

    nq = length(G.rbasis.qpts)
    for iatm2 in 1:G.rbasis.natom, iatm1 in 1:G.rbasis.natom
        # FIXME: multiple natom
        for xq in G.rbasis.qpts
            # Compute Green function at xq
            @views for (iv, v) in enumerate(vs)
                Gq_data[:, :, iv] .= G_lazy(xq, v)
            end
            G_q = fit_basis_coeff(Gq_data, basis.freq, vs, 3)

            # Fouirer transform xq -> R
            @views for (iR, R) in enumerate(G.rbasis.R_replicas[iatm1, iatm2])
                ndegen = G.rbasis.R_ndegen[iatm1, iatm2][iR]
                coeff = cispi(-2 * xq' * R) / nq / ndegen
                G.data[iatm1, iatm2][:, :, :, iR] .+= G_q .* coeff
            end
        end
    end
    G
end

function interpolate_to_q(G::RealSpaceGreen2P, xq, iatm1::Integer, iatm2::Integer)
    G_q = Green2P{get_formalism(G)}(eltype(G), G.basis, G.norb)
    @views for (iR, R) in enumerate(G.rbasis.R_replicas[iatm1, iatm2])
        # Fouirer transform R_B -> xq
        coeff = cispi(2 * xq' * R)
        G_q.data .+= G.data[iatm1, iatm2][:, :, :, iR] .* coeff
        G_q.offset .+= G.offset[iatm1, iatm2][:, :, iR] .* coeff
    end
    G_q
end

function Base.getindex(G::RealSpaceGreen2P{F}, iatm1::Integer, iatm2::Integer, R) where {F}
    iR = findfirst(x -> x == R, G.rbasis.R_replicas[iatm1, iatm2])
    iR === nothing && return nothing
    data = view(G.data[iatm1, iatm2], :, :, :, iR)
    offset = view(G.offset[iatm1, iatm2], :, :, iR)
    Green2P{F}(G.basis, G.norb, data, offset)
end

"""
    solve_Dyson(G0, Σ::RealSpaceGreen2P{F}, basis=get_basis(Σ))
"""
function solve_Dyson(G0, Σ::RealSpaceGreen2P{F}, basis=get_basis(Σ)) where {F}
    # For multiple atoms, need to construct a matrix including all atoms and invert it.
    Σ.rbasis.natom > 1 && error("multiple atoms not implemented")

    G = RealSpaceGreen2P{F}(eltype(Σ), basis.freq, basis.r, Σ.norb);

    nq = length(G.rbasis.qpts)
    for iatm2 in 1:G.rbasis.natom, iatm1 in 1:G.rbasis.natom
        # FIXME: multiple natom
        for xq in G.rbasis.qpts
            G0_q = interpolate_to_q(G0, xq, iatm1, iatm2)
            Σ_q = interpolate_to_q(Σ, xq, iatm1, iatm2)
            G_q = solve_Dyson(G0_q, Σ_q, basis.freq)

            # Fouirer transform xq -> R
            @views for (iR, R) in enumerate(G.rbasis.R_replicas[iatm1, iatm2])
                ndegen = G.rbasis.R_ndegen[iatm1, iatm2][iR]
                coeff = cispi(-2 * xq' * R) / nq / ndegen
                G.data[iatm1, iatm2][:, :, :, iR] .+= G_q.data .* coeff
            end
        end
    end
    G
end
