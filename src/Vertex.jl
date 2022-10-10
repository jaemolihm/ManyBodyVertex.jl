using TensorOperations


"""
    vertex_to_matrix(Γ, w, channel='a')
Convert a 4-point vertex into a (2-point)×(2-point) matrix format.
# Inputs
- `Γ`: Vertex object
- `channel`: channel parametrization, 'a', 'p', or 't'.
- `w`: bosonic frequency in the given channel

# Vertex parametrization
Currently, we only implement a single channel.
Later, we may implement different

Currently, we write the indices in the order
```
2 3
 X
1 4
```

The frequency parametrization reads
- `a` channel: ``Γ(v, v'; w) = Γ(v-w/2, -v-w/2, v'+w/2, -v'+w/2)``
(Frequency is positive for an outgoing leg.)
Ref: Gievers et al, Eur. Phys. J. B 95, 108 (2022), Fig. 3

# Matrix representation of vertex
A vertex has 4 indices. In the matrix representation, these indices are grouped into two
pairs. We use `Γ(1,2,3,4) → Γ(12, 34)` (which is valid only when we consider a single
channel).
"""

abstract type AbstractFrequencyVertex{F} end
nkeldysh(F::Symbol) = F === :KF ? 2 : 1
nkeldysh(::AbstractFrequencyVertex{F}) where {F} = nkeldysh(F)

struct Vertex4P{F, DT, BF1, BF2, BB} <: AbstractFrequencyVertex{F}
    # Basis for fermionic frequencies
    basis_f1::BF1
    basis_f2::BF2
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    function Vertex4P{F, DT}(basis_f1::BF1, basis_f2::BF2, basis_b::BB, norb, data::DT) where {F, DT, BF1, BF2, BB}
        new{F, DT, BF1, BF2, BB}(basis_f1, basis_f2, basis_b, norb, data)
    end
end

struct Bubble{F, DT, BF, BB} <: AbstractFrequencyVertex{F}
    # Basis for fermionic frequencies
    basis_f::BF
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    function Bubble{F, DT}(basis_f::BF, basis_b::BB, norb, data::DT) where {F, DT, BF, BB}
        new{F, DT, BF, BB}(basis_f, basis_b, norb, data)
    end
end

nb_f1(Γ::Vertex4P) = size(Γ.basis_f1, 2)
nb_f2(Γ::Vertex4P) = size(Γ.basis_f2, 2)
nb_b(Γ::Vertex4P) = size(Γ.basis_b, 2)

nb_f(Π::Bubble) = size(Π.basis_f, 2)
nb_b(Π::Bubble) = size(Π.basis_b, 2)

Vertex4P{F}(basis_f1, basis_f2, basis_b, norb=1) where {F} = Vertex4P{F}(ComplexF64, basis_f1, basis_f2, basis_b, norb)

Bubble{F}(basis_f, basis_b, norb=1) where {F} = Bubble{F}(ComplexF64, basis_f, basis_b, norb)

function Vertex4P{F}(::Type{T}, basis_f1, basis_f2, basis_b, norb=1) where {F, T}
    nb_f1 = size(basis_f1, 2)
    nb_f2 = size(basis_f2, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f1 * (norb * nk)^2, nb_f2 * (norb * nk)^2, nb_b)
    Vertex4P{F, typeof(data)}(basis_f1, basis_f2, basis_b, norb, data)
end

function Bubble{F}(::Type{T}, basis_f, basis_b, norb=1) where {F, T}
    nb_f = size(basis_f, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f, (norb * nk)^2, (norb * nk)^2, nb_b)
    Bubble{F, typeof(data)}(basis_f, basis_b, norb, data)
end

# Customize printing
function Base.show(io::IO, Γ::Vertex4P)
    print(io, Base.typename(typeof(Γ)).wrapper)
    print(io, "(nbasis_f1=$(nb_f2(Γ)), nbasis_f2=$(nb_f2(Γ)), nbasis_b=$(nb_b(Γ)), ")
    print(io, "norb=$(Γ.norb), data=$(Base.summary(Γ.data)))")
end

function Base.show(io::IO, Π::Bubble)
    print(io, Base.typename(typeof(Π)).wrapper)
    print(io, "(nbasis_f=$(nb_f(Π)), nbasis_b=$(nb_b(Π)), ")
    print(io, "norb=$(Π.norb), data=$(Base.summary(Π.data)))")
end

"""
    vertex_to_matrix(Γ::Vertex4P, w, channel='a')
Evaluate a 4-point vertex at given bosonic frequency `w` and return in the matrix form.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`, `j`: Orbital/Keldysh index
- Input `Γ.data`: `(a, i, j), (a', i', j'), b`
- Output: `(a, i, j), (a', i', j')`
"""
function vertex_to_matrix(Γ::Vertex4P, w, channel='a')
    @assert ndims(Γ.data) == 3
    channel ∉ ('a', 'p', 't') && error("Wrong channel $channel")
    channel != 'a' && error("Only channel a is implemented")

    # Contract the bosonic frequency basis
    coeff_w = Γ.basis_b[w, :]
    Γ_w = zeros(eltype(Γ.data), size(Γ.data)[1:2])
    @tensor Γ_w[aij1, aij2] := Γ.data[aij1, aij2, b] * coeff_w[b]
    Γ_w
end

"""
    bubble_to_matrix(Π, w, overlap)
Evaluate a 4-point bubble at given bosonic frequency `w` and return in the matrix form.
- `a`: fermionic frequency basis index of bubble
- `x`: fermionic frequency basis index of vertex
- `b`: frequency basis index
- `i`, `j`: Orbital/Keldysh index
- Input `Π.data`: `a,  (i, j), (i', j'), b`
- Input `overlap`: `x, x', a`
- Output: `(x, i, j), (x', i', j')`
"""
function bubble_to_matrix(Π::Bubble, w, overlap)
    @assert ndims(Π.data) == 4
    @assert size(overlap, 3) == nb_f(Π)
    nv_Γ = size(overlap, 1)
    norb2 = size(Π.data, 2)

    # Contract the bosonic frequency basis
    # Π_w: a, (i, j), (i', j')
    coeff_w = Π.basis_b[w, :]
    Π_w = zeros(eltype(Π.data), size(Π.data)[1:3])
    @tensor Π_w[a, ij1, ij2] := Π.data[a, ij1, ij2, b] * coeff_w[b]

    # Contract with the overlap
    Π_vertex_tmp = zeros(eltype(Π.data), nv_Γ, nv_Γ, norb2, norb2)
    @tensor Π_vertex_tmp[x1, x2, ij1, ij2] := overlap[x1, x2, a] * Π_w[a, ij1, ij2]
    Π_vertex = reshape(PermutedDimsArray(Π_vertex_tmp, (1, 3, 2, 4)), nv_Γ * norb2, nv_Γ * norb2)
    collect(Π_vertex) .* integral_coeff(Π)
end

integral_coeff(Π::Bubble{:KF}) = 1 / 2 / eltype(Π.data)(π)
integral_coeff(Π::Bubble{:MF}) = error("MF Not yet implemented")
integral_coeff(Π::Bubble{:ZF}) = error("ZF Not yet implemented")


"""
    vertex_keldyshview(Γ::Vertex4P{:KF})
Return a the 4-point KF vertex as a 11-dimensional array.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`: Orbital index
- `k`: Keldysh index
- Input `Γ.data`: `(a, i1, k1, i2, k2), (a', i3, k3, i4, k4), b`
- Output: `a, a', i1, i2, i3, i4, k1, k2, k3, k4, b`
"""
function vertex_keldyshview(Γ::Vertex4P{:KF})
    norb = Γ.norb
    data_size = (nb_f1(Γ), norb, 2, norb, 2, nb_f2(Γ), norb, 2, norb, 2, nb_b(Γ))
    PermutedDimsArray(Base.ReshapedArray(Γ.data, data_size, ()), (1, 6, 2, 4, 7, 9, 3, 5, 8, 10, 11))
end
