using OMEinsum

"""
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

abstract type AbstractFrequencyVertex{F, T} end
abstract type AbstractVertex4P{F, C, T} <: AbstractFrequencyVertex{F, T} end
nkeldysh(F::Symbol) = F === :KF ? 2 : 1
nkeldysh(::AbstractFrequencyVertex{F}) where {F} = nkeldysh(F)
get_formalism(::AbstractVertex4P{F}) where {F} = F
channel(::AbstractVertex4P{F, C}) where {F, C} = C

struct Vertex4P{F, C, T, BF1, BF2, BB, DT <: AbstractArray{T}} <: AbstractVertex4P{F, C, T}
    # Basis for fermionic frequencies
    basis_f1::BF1
    basis_f2::BF2
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    function Vertex4P{F, C}(basis_f1::BF1, basis_f2::BF2, basis_b::BB, norb, data::DT) where {F, C, DT <: AbstractArray{T}, BF1, BF2, BB} where {T}
        C ∈ (:A, :P, :T) || throw(ArgumentError("Wrong channel $C"))
        new{F, C, T, BF1, BF2, BB, DT}(basis_f1, basis_f2, basis_b, norb, data)
    end
end

nb_f1(Γ::Vertex4P) = size(Γ.basis_f1, 2)
nb_f2(Γ::Vertex4P) = size(Γ.basis_f2, 2)
nb_b(Γ::Vertex4P) = size(Γ.basis_b, 2)
get_nind(Γ::AbstractFrequencyVertex) = nkeldysh(Γ) * Γ.norb

function Vertex4P{F, C}(basis_f1, basis_f2, basis_b, norb=1) where {F, C}
    Vertex4P{F, C}(ComplexF64, basis_f1, basis_f2, basis_b, norb)
end

function Vertex4P{F, C}(::Type{T}, basis_f1, basis_f2, basis_b, norb=1) where {F, C, T}
    nb_f1 = size(basis_f1, 2)
    nb_f2 = size(basis_f2, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f1 * (norb * nk)^2, nb_f2 * (norb * nk)^2, nb_b)
    Vertex4P{F, C}(basis_f1, basis_f2, basis_b, norb, data)
end

function Base.similar(Γ::Vertex4P{F, C, T}, ::Type{ElType}=T) where {F, C, T, ElType}
    Vertex4P{F, C}(Γ.basis_f1, Γ.basis_f2, Γ.basis_b, Γ.norb, similar(Γ.data, ElType))
end

function Base.:*(x::Number, A::Vertex4P)
    B = similar(A)
    B.data .= A.data .* x
    B
end
function Base.:/(A::Vertex4P, x::Number)
    B = similar(A)
    B.data .= A.data ./ x
    B
end
Base.:*(A::Vertex4P, x::Number) = x * A

function _check_basis_identity(A::Vertex4P, B::Vertex4P)
    A.basis_f1 === B.basis_f1 || error("basis must be identical")
    A.basis_f2 === B.basis_f2 || error("basis must be identical")
    A.basis_b === B.basis_b || error("basis must be identical")
end

function Base.:+(A::Vertex4P, B::Vertex4P)
    _check_basis_identity(A, B)
    C = similar(A)
    C.data .= A.data .+ B.data
    C
end

function Base.:-(A::Vertex4P, B::Vertex4P)
    _check_basis_identity(A, B)
    C = similar(A)
    C.data .= A.data .- B.data
    C
end

# Customize printing
function Base.show(io::IO, Γ::Vertex4P{F, C}) where {F, C}
    print(io, Base.typename(typeof(Γ)).wrapper, "{:$F, :$C}")
    print(io, "(nbasis_f1=$(nb_f1(Γ)), nbasis_f2=$(nb_f2(Γ)), nbasis_b=$(nb_b(Γ)), ")
    print(io, "norb=$(Γ.norb), data=$(Base.summary(Γ.data)))")
end

"""
    (Γ::Vertex4P{F, C_Γ, T})(v1, v2, w, c_out::Val=Val(C_Γ)) where {F, C_Γ, T}
Evaluate vertex at given frequencies in the channel parametrization.
"""
function (Γ::Vertex4P{F, C_Γ, T})(v1, v2, w, c_out::Val=Val(C_Γ)) where {F, C_Γ, T}
    v1234 = frequency_to_standard(Val(F), c_out, v1, v2, w)
    v1_Γ, v2_Γ, w_Γ = frequency_to_channel(Val(F), Val(C_Γ), v1234...)

    nind = get_nind(Γ)
    coeff_f1 = Γ.basis_f1[v1_Γ, :]
    coeff_f2 = Γ.basis_f2[v2_Γ, :]
    coeff_b = Γ.basis_b[w_Γ, :]
    # Compute the following einsum operation with a manually optimized implementation.
    # @ein Γ_vvw[i, j] := Γ_array[v1, i, v2, j, w] * coeff_f1[v1] * coeff_f2[v2] * coeff_b[w]
    Γ_array = zeros(T, (nb_f1(Γ), nind^2, nb_f2(Γ), nind^2))
    Γ_array_reshape = Base.ReshapedArray(Γ_array, (length(Γ_array),), ())
    Γ_data_reshape = Base.ReshapedArray(Γ.data, (prod(size(Γ.data)[1:2]), size(Γ.data, 3)), ())
    mul!(Γ_array_reshape, Γ_data_reshape, coeff_b)
    Γ_vvw = zeros(T, nind^2, nind^2)
    @inbounds for inds in CartesianIndices(Γ_array)
        iv1, i, iv2, j = inds.I
        Γ_vvw[i, j] += Γ_array[inds] * coeff_f1[iv1] * coeff_f2[iv2]
    end
    _permute_orbital_indices_matrix_4p(Val(C_Γ), c_out, Γ_vvw, nind)
end

"""
    to_matrix(Γ::Vertex4P{F, C, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, c::Val=Val(C)) where {F, C, T}
Evaluate a 4-point vertex at given bosonic frequency `w`, fermionic bases `basis1` and
`basis2`, and channel `c`. Return the matrix form.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`, `j`: Orbital/Keldysh index
- Input `Γ.data`: `(a, i, j), (a', i', j'), b`
- Output: `(a, i, j), (a', i', j')`
"""
function to_matrix(Γ::Vertex4P{F, C, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, c::Val=Val(C)) where {F, C, T}
    if c !== Val(C) && (ntails(basis1) > 0 || ntails(basis2) > 0)
        # If we map to a different channel, the tails lead to problems so we disable it.
        # There is a problem with the fitting process that it oversamples regions where
        # v1-v2 or v1+v2 is small. Also, there is a problem for the tail-tail part. For
        # example, the 1/(v1-v2) term cannot be written as a sum of 1/v1 and 1/v2 terms.
        error("to_matrix to different channel with tails do not work.")
    end

    if c === Val(C) && basis1 === Γ.basis_f1 && basis2 === Γ.basis_f2
        # Same channel, same basis. Just need to contract the bosonic frequency basis.
        coeff_w = Γ.basis_b[w, :]
        @ein Γ_w[aij1, aij2] := Γ.data[aij1, aij2, b] * coeff_w[b]
        Γ_w = Γ_w::Matrix{T}
        return Γ_w
    else
        nind = get_nind(Γ)
        vs1 = get_fitting_points(basis1)
        vs2 = get_fitting_points(basis2)
        Γ_w_data = zeros(T, length(vs1), length(vs2), nind^2, nind^2)
        for (i2, v2) in enumerate(vs2), (i1, v1) in enumerate(vs1)
            Γ_w_data[i1, i2, :, :] .= Γ(v1, v2, w, c)
        end

        Γ_tmp1 = fit_basis_coeff(Γ_w_data, basis1, vs1, 1)
        Γ_tmp2 = fit_basis_coeff(Γ_tmp1, basis2, vs2, 2)

        Γ_w = reshape(permutedims(Γ_tmp2, (1, 3, 2, 4)), size(basis1, 2) * nind^2, size(basis2, 2) * nind^2)
        return Γ_w
    end
end

"""
    apply_crossing(Γ::Vertex4P{F, C}) where {F, C}
Apply the crossing operation: swap indices 1 and 3. Works only for channel A and T.
"""
function apply_crossing(Γ::Vertex4P{F, C}) where {F, C}
    if C === :A
        C_out = :T
    elseif C === :T
        C_out = :A
    elseif C === :P
        # For channel P, we need to additionally change v1 -> -v1. This is not implemented.
        error("Not implemented for channel P")
    else
        error("Wrong channel $C")
    end
    # Multiply -1 to data to account the fermionic parity.
    Vertex4P{F, C_out}(Γ.basis_f1, Γ.basis_f2, Γ.basis_b, Γ.norb, .-Γ.data)
end

"""
    vertex_keldyshview(Γ::Vertex4P{:KF})
Return a the 4-point KF vertex as a 11-dimensional array in the standard index order.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`: Orbital index
- `k`: Keldysh index
- Input `Γ.data`: `(a, i1, k1, i2, k2), (a', i3, k3, i4, k4), b`
- Output: `a, a', i1, i2, i3, i4, k1, k2, k3, k4, b`
"""
function vertex_keldyshview(Γ::Vertex4P{:KF, C}) where {C}
    norb = Γ.norb
    data_size = (nb_f1(Γ), norb, 2, norb, 2, nb_f2(Γ), norb, 2, norb, 2, nb_b(Γ))
    # ((v, ik1, ik2), (v', ik3, ik4), w) -> (v, v', i1, i2, i3, i4, k1, k2, k3, k4, w)
    perm = [1, 6, 2, 4, 7, 9, 3, 5, 8, 10, 11]
    # channel orbital/Keldysh order -> standard orbital/Keldysh order
    permute!(perm, [1, 2, indices_to_standard(Val(C), 3:6)..., indices_to_standard(Val(C), 7:10)..., 11])
    PermutedDimsArray(Base.ReshapedArray(Γ.data, data_size, ()), perm)
end

"""
    fit_bosonic_basis_coeff!(Γ, Γ_data, ws)
Compute the basis coefficients using the data computed at a grid of bosonic frequencies ws.
"""
function fit_bosonic_basis_coeff!(Γ, Γ_data, ws)
    Γ.data .= fit_basis_coeff(Γ_data, Γ.basis_b, ws, ndims(Γ.data))
    Γ
end

"""
    fit_basis_coeff(data, basis, grid, dim)
Fit the coefficients of the basis to the data to make ``data ≈ basis_value * coeff`` hold.
Use the `dim`-th index of `data` for fitting and leave other indices.
`size(data, dim) == length(grid)` must hold.
"""
function fit_basis_coeff(data, basis, grid, dim)
    if dim == 1
        # Reshape data to a Matrix by merging indices 2 to end, and call left division.
        basis_value = basis[grid, :]
        ngrid, ncoeff = size(basis_value)
        size_keep = size(data)[2:end]
        reshape(basis_value \ reshape(data, ngrid, prod(size_keep)), ncoeff, size_keep...)
    else
        # Permute the dim-th index to the front and call the dim=1 case
        perm = Vector(1:ndims(data))
        perm[1], perm[dim] = dim, 1
        # perm = pushfirst!(deleteat!(Vector(1:ndims(data)), dim), dim)
        data_permuted = permutedims(data, perm)
        coeff_permuted = fit_basis_coeff(data_permuted, basis, grid, 1)
        permutedims(coeff_permuted, invperm(perm))
    end
end
