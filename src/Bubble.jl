abstract type AbstractBubble{F, T} <: AbstractFrequencyVertex{F, T} end

mutable struct Bubble{F, T, BF, BB, DT <: AbstractArray{T}} <: AbstractBubble{F, T}
    # Basis for fermionic frequencies
    basis_f::BF
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    # Cached basis and overlap
    cache_basis_L
    cache_basis_R
    cache_overlap_LR
    function Bubble{F}(basis_f::BF, basis_b::BB, norb, data::DT) where {F, DT <: AbstractArray{T}, BF, BB} where {T}
        new{F, T, BF, BB, DT}(basis_f, basis_b, norb, data, nothing, nothing, nothing)
    end
end

Bubble{F}(basis_f, basis_b, norb=1) where {F} = Bubble{F}(ComplexF64, basis_f, basis_b, norb)

function Bubble{F}(::Type{T}, basis_f, basis_b, norb=1) where {F, T}
    nb_f = size(basis_f, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f, (norb * nk)^2, (norb * nk)^2, nb_b)
    Bubble{F}(basis_f, basis_b, norb, data)
end

function Base.show(io::IO, Π::AbstractBubble)
    print(io, Base.typename(typeof(Π)).wrapper)
    print(io, "(nbasis_f=$(nb_f(Π)), nbasis_b=$(nb_b(Π)), ")
    print(io, "norb=$(Π.norb), data=$(Base.summary(Π.data)))")
end

nb_f(Π::AbstractBubble) = size(Π.basis_f, 2)
nb_b(Π::AbstractBubble) = size(Π.basis_b, 2)

function (Π::AbstractBubble{F, T})(w) where {F, T}
    # Evaluate the bubble at given bosonic frequency w
    # Output: a, (i, j), (i', j')
    coeff_w = Π.basis_b[w, :]
    @ein Π_w[a, ij1, ij2] := Π.data[a, ij1, ij2, b] * coeff_w[b]
    Π_w::Array{T, 3}
end

"""
Load overlap from cache, recompute if basis has changed.
"""
function cache_and_load_overlaps(Π::Bubble, basis_L::Basis, basis_R::Basis)
    if basis_L !== Π.cache_basis_L || basis_R !== Π.cache_basis_R
        Π.cache_overlap_LR = basis_integral(basis_L, basis_R, Π.basis_f)
        Π.cache_basis_L = basis_L
        Π.cache_basis_R = basis_R
    end
    (Π.cache_overlap_LR,)
end

function to_matrix(Π::AbstractBubble, w, basis_L::Basis, basis_R::Basis)
    # Function barrier
    to_matrix(Π, w, cache_and_load_overlaps(Π, basis_L, basis_R)...)
end

"""
    to_matrix(Π, w, overlap)
Evaluate a 4-point bubble at given bosonic frequency `w` and return in the matrix form.
- `a`: fermionic frequency basis index of bubble
- `x`: fermionic frequency basis index of vertex
- `b`: frequency basis index
- `i`, `j`: Orbital/Keldysh index
- Input `Π.data`: `a,  (i, j), (i', j'), b`
- Input `overlap`: `x, x', a`
- Output: `(x, i, j), (x', i', j')`
"""
function to_matrix(Π::Bubble{F, T}, w, overlap) where {F, T}
    @assert ndims(Π.data) == 4
    @assert size(overlap, 3) == nb_f(Π)
    nv_Γ1, nv_Γ2 = size(overlap)[1:2]
    nind2 = get_nind(Π)^2

    Π_w = Π(w)
    @ein Π_vertex_tmp[x1, x2, ij1, ij2] := overlap[x1, x2, a] * Π_w[a, ij1, ij2]
    Π_vertex_tmp = Π_vertex_tmp::Array{T, 4}
    Π_vertex = reshape(PermutedDimsArray(Π_vertex_tmp, (1, 3, 2, 4)), nv_Γ1 * nind2, nv_Γ2 * nind2)
    collect(Π_vertex) .* integral_coeff(Π)
end

integral_coeff(::AbstractBubble{:KF, T}) where {T} = 1 / 2 / real(T)(π)
integral_coeff(::AbstractBubble{:MF}) = error("MF Not yet implemented")
integral_coeff(::AbstractBubble{:ZF}) = error("ZF Not yet implemented")
