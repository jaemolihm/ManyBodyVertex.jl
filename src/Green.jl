using OMEinsum

abstract type AbstractFrequencyVertex{F, T} end
Base.eltype(::AbstractFrequencyVertex{F, T}) where {F, T} = T
get_formalism(::AbstractFrequencyVertex{F}) where {F} = F
nkeldysh(F::Symbol) = F === :KF ? 2 : 1
nkeldysh(::AbstractFrequencyVertex{F}) where {F} = nkeldysh(F)

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
    function Green2P{F}(basis::BT, norb, data::DT) where {F, DT, BT}
        new{F, eltype(data), BT, DT}(basis, norb, data)
    end
end

function Green2P{F}(::Type{T}, basis, norb=1) where {F, T}
    nb = size(basis, 2)
    nk = nkeldysh(F)
    data = zeros(T, norb * nk, norb * nk, nb)
    Green2P{F}(basis, norb, data)
end
Green2P{F}(basis, norb=1) where {F} = Green2P{F}(ComplexF64, basis, norb)

function Base.similar(G::Green2P{F, T}, ::Type{ElType}=T) where {F, T, ElType}
    Green2P{F}(G.basis, G.norb, similar(G.data, ElType))
end

"""
    (G::Green2P{F, T})(v) where {F, T}
Evaluate Green function `G` at frequency `v`.
"""
function (G::Green2P)(v)
    coeff = G.basis[v, :]
    @ein G_v[i, j] := G.data[i, j, v1] * coeff[v1]
    G_v::Matrix{eltype(G)}
end
