using FFTW

# TODOs
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

function Base.similar(G::RealSpaceGreen2P{F}) where {F}
    RealSpaceGreen2P{F}(G.basis, G.rbasis, G.norb, zero.(G.data))
end
Base.zero(G::RealSpaceGreen2P) = similar(G)

function _check_basis_identity(A::RealSpaceGreen2P, B::RealSpaceGreen2P)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    A.rbasis === B.rbasis || error("Different rbasis")
    A.basis === B.basis || error("Different basis")
    A.norb === B.norb || error("Different norb")
end
data_fieldnames(::Type{<:RealSpaceGreen2P}) = (:data, :offset)

"""
    fft_Green2P!(G_R, G_q, rbasis::RealSpaceBasis2P, R_replicas, R_ndegen)
Fourier transform xq -> R: ``G_R(R) = ∑_q exp(-2πi*q*R) * G_q(q) / nq / ndegen(R)``.
`G_q` is destroyed in the output.
- `G_R`: Output. Array of size `(n1, n2, n3, nR)`, where `nR = length(R_replicas)`.
- `G_q`: Input. Array of size `(nq_1, ..., nq_Dim, n1, n2, n3)`, where `nq_i = qgrid[i]`.
"""
function fft_Green2P!(G_R, G_q, qgrid::NTuple{Dim}, R_replicas, R_ndegen) where {Dim}
    @assert size(G_R)[end] == length(R_replicas) == length(R_ndegen)
    @assert size(G_q)[1:Dim] == qgrid
    @assert size(G_R)[1:3] == size(G_q)[end-2:end]

    fft!(G_q, 1:Dim)
    nq = prod(qgrid)
    @views for (iR, (R, ndegen)) in enumerate(zip(R_replicas, R_ndegen))
        iR_grid = mod.(R, qgrid) .+ 1
        G_R[:, :, :, iR] .= G_q[iR_grid..., :, :, :] ./ (nq .* ndegen)
    end
end

function green_lazy_to_explicit(G_lazy::AbstractLazyGreen2P, basis::NamedTuple{(:freq, :r)})
    basis.r.natom > 1 && error("multiple atoms not implemented")
    F = get_formalism(G_lazy)
    nind = get_nind(G_lazy)

    qgrid = basis.r.qgrid
    @assert length(basis.r.qpts) === prod(qgrid)

    G = RealSpaceGreen2P{F}(eltype(G_lazy), basis.freq, basis.r, G_lazy.norb);

    vs = get_fitting_points(basis.freq)
    Gq_data = zeros(eltype(G_lazy), nind, nind, length(vs))
    Gq_collect = zeros(eltype(G_lazy), qgrid..., nind, nind, nbasis(basis.freq))

    for iatm2 in 1:G.rbasis.natom, iatm1 in 1:G.rbasis.natom
        # FIXME: multiple natom
        for iqs in CartesianIndices(qgrid)
            xq = SVector((iqs.I .- 1) ./ qgrid)
            # Compute Green function at xq
            for (iv, v) in enumerate(vs)
                Gq_data[:, :, iv] .= G_lazy(xq, v)
            end
            Gq_collect[iqs, :, :, :] .= fit_basis_coeff(Gq_data, basis.freq, vs, 3)
        end

        fft_Green2P!(G.data[iatm1, iatm2], Gq_collect, qgrid,
            G.rbasis.R_replicas[iatm1, iatm2], G.rbasis.R_ndegen[iatm1, iatm2])
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
    qgrid = basis.r.qgrid

    nind = get_nind(Σ)
    G = RealSpaceGreen2P{F}(eltype(Σ), basis.freq, basis.r, Σ.norb)
    Gq_collect = zeros(eltype(Σ), qgrid..., nind, nind, nbasis(basis.freq))

    for iatm2 in 1:G.rbasis.natom, iatm1 in 1:G.rbasis.natom
        # FIXME: multiple natom
        for iqs in CartesianIndices(qgrid)
            xq = SVector((iqs.I .- 1) ./ qgrid)

            G0_q = interpolate_to_q(G0, xq, iatm1, iatm2)
            Σ_q = interpolate_to_q(Σ, xq, iatm1, iatm2)
            G_q = solve_Dyson(G0_q, Σ_q, basis.freq)
            Gq_collect[iqs, :, :, :] .= G_q.data
        end

        fft_Green2P!(G.data[iatm1, iatm2], Gq_collect, qgrid,
            G.rbasis.R_replicas[iatm1, iatm2], G.rbasis.R_ndegen[iatm1, iatm2])
    end
    G
end
