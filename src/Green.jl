using OMEinsum

# TODO: Test for Green2P
# TODO: Test for Dyson

# Type for lazy Green function object
abstract type AbstractLazyGreen2P{F, T} <: AbstractFrequencyVertex{F, T} end

"""
    green_lazy_to_explicit(G::AbstractLazyGreen2P{F}, basis) where {F} => Green2P{F}
Convert a lazily-defined Green function to an explicitly-defined Green function on `basis`.
"""
function green_lazy_to_explicit(G::AbstractLazyGreen2P{F}, basis::Basis) where {F}
    nind = get_nind(G)
    vs = get_fitting_points(basis)
    G_data = zeros(eltype(G), nind, nind, length(vs))
    for (iv, v) in enumerate(vs)
        G_data[:, :, iv] .= G(v)
    end
    Green2P{F}(basis, G.norb, fit_basis_coeff(G_data, basis, vs, 3))
end
function green_lazy_to_explicit(G::AbstractLazyGreen2P, basis::NamedTuple{(:freq,)})
    green_lazy_to_explicit(G, basis.freq)
end
function get_G!(G_v, G::AbstractLazyGreen2P, v)
    G_v .= G(v)
end


"""
    Green2P{F}(::Type{T}=ComplexF64, basis, norb=1)
2-point Green function.
"""
struct Green2P{F, T, BT, DT <: AbstractArray{T}} <: AbstractFrequencyVertex{F, T}
    # Basis (can be either fermionic or bosonic)
    basis::BT
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    # Constant offset matrix to be added. Used for the Hartree self-energy. Size (nind, nind).
    offset::Matrix{T}
    function Green2P{F}(basis::BT, norb, data::DT, offset=nothing) where {F, DT, BT}
        nind = norb * nkeldysh(F)
        offset === nothing && (offset = zeros(eltype(data), nind, nind))
        new{F, eltype(data), BT, DT}(basis, norb, data, offset)
    end
end
data_fieldnames(::Type{<:Green2P}) = (:data, :offset)

function Green2P{F}(::Type{T}, basis, norb=1) where {F, T}
    nb = size(basis, 2)
    nk = nkeldysh(F)
    data = zeros(T, norb * nk, norb * nk, nb)
    Green2P{F}(basis, norb, data)
end
Green2P{F}(basis, norb=1) where {F} = Green2P{F}(ComplexF64, basis, norb)

function Base.similar(G::Green2P{F, T}, ::Type{ElType}=T) where {F, T, ElType}
    Green2P{F}(G.basis, G.norb, similar(G.data, ElType), similar(G.offset, ElType))
end
Base.zero(G::Green2P) = (x = similar(G); x.data .= 0; x.offset .= 0; x)

function _check_basis_identity(A::Green2P, B::Green2P)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    A.basis === B.basis || error("Different basis")
end

green_lazy_to_explicit(G::Green2P, basis) = G
get_basis(G::Green2P) = (; freq=G.basis)

"""
    (G::Green2P{F, T})(v) where {F, T}
Evaluate Green function `G` at frequency `v`.
"""
function (G::Green2P)(v)
    nind = get_nind(G)
    get_G!(zeros(eltype(G), nind, nind), G, v)
end

function get_G!(G_v, G::Green2P, v)
    G_v .= G.offset
    @views for ib in 1:nbasis(G.basis)
        coeff = G.basis[v, ib]
        coeff == 0 && continue
        G_v .+= G.data[:, :, ib] .* coeff
    end
    G_v
end

"""
    keldyshview(G::Green2P{:KF})
Return a the 2-point KF vertex as a 5-dimensional array
- `a`: frequency basis index
- `i`: Orbital index
- `k`: Keldysh index
- Input `G.data`: `(i1, k1), (i2, k2), a`
- Output: `k1, k2, i1, i2, a`
"""
function keldyshview(G::Green2P{:KF})
    norb = G.norb
    data_size = (norb, 2, norb, 2, nbasis(G.basis))
    # (i1, k1, i2, k2, a) -> (k1, k2, i1, i2, a)
    perm = [2, 4, 1, 3, 5]
    PermutedDimsArray(Base.ReshapedArray(G.data, data_size, ()), perm)
end

"""
    solve_Dyson(G0, Σ, basis=Σ.basis) => G
Solve Dyson equation to compute the interacting Green function: ``G = (G0⁻¹ - Σ)⁻¹``.
"""
function solve_Dyson(G0, Σ, basis=Σ.basis)
    vs = get_fitting_points(basis)
    nind = get_nind(Σ)
    G_data = zeros(ComplexF64, nind, nind, length(vs))
    for (iv, v) in enumerate(vs)
        G_data[:, :, iv] .= inv(inv(G0(v)) .- Σ(v))
    end
    data = fit_basis_coeff(G_data, basis, vs, 3)
    Green2P{get_formalism(Σ)}(basis, Σ.norb, data)
end
