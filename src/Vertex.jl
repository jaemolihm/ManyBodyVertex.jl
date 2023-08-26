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

abstract type AbstractVertex4P{F, T} <: AbstractFrequencyVertex{F, T} end

struct Vertex4P{F, T, BF1, BF2, BB, DT <: AbstractArray{T}} <: AbstractVertex4P{F, T}
    # Channel
    channel::Symbol
    # Basis for fermionic frequencies
    basis_f1::BF1
    basis_f2::BF2
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    function Vertex4P{F}(channel, basis_f1::BF1, basis_f2::BF2, basis_b::BB, norb, data::DT) where {F, DT <: AbstractArray{T}, BF1, BF2, BB} where {T}
        channel ∈ (:A, :P, :T) || throw(ArgumentError("Wrong channel $channel"))
        new{F, T, BF1, BF2, BB, DT}(channel, basis_f1, basis_f2, basis_b, norb, data)
    end
end
data_fieldnames(::Type{<:Vertex4P}) = (:data,)

get_channel(Γ::Vertex4P) = Γ.channel

nb_f1(Γ::Vertex4P) = size(Γ.basis_f1, 2)
nb_f2(Γ::Vertex4P) = size(Γ.basis_f2, 2)
nb_b(Γ::Vertex4P) = size(Γ.basis_b, 2)
get_nind(Γ::AbstractFrequencyVertex) = nkeldysh(Γ) * Γ.norb

function Vertex4P{F}(C, basis_f1, basis_f2, basis_b, norb=1) where {F}
    Vertex4P{F}(C, ComplexF64, basis_f1, basis_f2, basis_b, norb)
end

function Vertex4P{F}(C, ::Type{T}, basis_f1, basis_f2, basis_b, norb=1) where {F, T}
    nb_f1 = size(basis_f1, 2)
    nb_f2 = size(basis_f2, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f1 * (norb * nk)^2, nb_f2 * (norb * nk)^2, nb_b)
    Vertex4P{F}(C, basis_f1, basis_f2, basis_b, norb, data)
end

function Base.similar(Γ::Vertex4P{F, T}, ::Type{ElType}=T) where {F, T, ElType}
    Vertex4P{F}(Γ.channel, Γ.basis_f1, Γ.basis_f2, Γ.basis_b, Γ.norb, similar(Γ.data, ElType))
end
Base.zero(Γ::Vertex4P) = (x = similar(Γ); x.data .= 0; x)

function _check_basis_identity(A::Vertex4P, B::Vertex4P)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    get_channel(A) === get_channel(B) || error("Different channel")
    A.basis_f1 === B.basis_f1 || error("Different basis_f1")
    A.basis_f2 === B.basis_f2 || error("Different basis_f2")
    A.basis_b === B.basis_b || error("Different basis_b")
end

# Customize printing
function Base.show(io::IO, Γ::Vertex4P{F}) where {F}
    C = get_channel(Γ)
    print(io, Base.typename(typeof(Γ)).wrapper, "{:$F}")
    print(io, "(channel=$C, nbasis_f1=$(nb_f1(Γ)), nbasis_f2=$(nb_f2(Γ)), ")
    print(io, "nbasis_b=$(nb_b(Γ)), norb=$(Γ.norb), data=$(Base.summary(Γ.data)))")
end

"""
    (Γ::Vertex4P{F})(v1, v2, w, C_out::Symbol=get_channel(Γ)) where {F}
Evaluate vertex at given frequencies in the channel parametrization.
"""
function (Γ::Vertex4P{F})(v1, v2, w, C_out::Symbol=get_channel(Γ)) where {F}
    C_Γ = get_channel(Γ)
    coeff_f1, coeff_f2, coeff_b = _get_vertex_coeff_buffers(Γ)
    nind = get_nind(Γ)
    Γ_vvw = zeros(eltype(Γ), nind^2, nind^2)

    v1234 = frequency_to_standard(Val(F), C_out, v1, v2, w)
    vvw = frequency_to_channel(Val(F), C_Γ, v1234)
    _evaluate_vertex_with_buffer!(Γ_vvw, Γ, vvw..., coeff_f1, coeff_f2, coeff_b)
    _permute_orbital_indices_matrix_4p(C_Γ, C_out, Γ_vvw, nind)
end

"""
Batched version of Vertex evaluation
"""
function (Γ::Vertex4P{F})(v1::AbstractVector, v2::AbstractVector, w::AbstractVector,
        C_out::Symbol=get_channel(Γ)) where {F}
    C_Γ = get_channel(Γ)
    coeff_f1, coeff_f2, coeff_b = _get_vertex_coeff_buffers(Γ)

    nind = get_nind(Γ)
    nfreq = length(v1)
    Γ_vvw = zeros(eltype(Γ), nfreq, nind^2, nind^2)
    @views for ifreq in 1:nfreq
        v1234 = frequency_to_standard(Val(F), C_out, v1[ifreq], v2[ifreq], w[ifreq])
        vvw = frequency_to_channel(Val(F), C_Γ, v1234)
        _evaluate_vertex_with_buffer!(Γ_vvw[ifreq, :, :], Γ, vvw..., coeff_f1, coeff_f2, coeff_b)
    end
    _permute_orbital_indices_matrix_4p_keep_dim1(C_Γ, C_out, Γ_vvw, nind)
end

function _get_vertex_coeff_buffers(Γ)
    coeff_f1 = zeros(eltype(Γ.basis_f1), size(Γ.basis_f1, 2))
    coeff_f2 = zeros(eltype(Γ.basis_f2), size(Γ.basis_f2, 2))
    coeff_b = zeros(eltype(Γ.basis_b), size(Γ.basis_b, 2))
    (; coeff_f1, coeff_f2, coeff_b)
end

function _evaluate_vertex_with_buffer!(Γ_vvw, Γ, v1, v2, w, coeff_f1, coeff_f2, coeff_b)
    # Temporary buffer arrays are given as input
    nind = get_nind(Γ)
    @views coeff_f1 .= Γ.basis_f1[v1, :]
    @views coeff_f2 .= Γ.basis_f2[v2, :]
    @views coeff_b .= Γ.basis_b[w, :]

    # Compute the following einsum operation with a manually optimized implementation.
    # Γ_array = Base.ReshapedArray(Γ.data, (nb_f1(Γ), nind^2, nb_f2(Γ), nind^2, nb_b(Γ)), ())
    # @ein Γ_vvw[ifreq, i, j] := Γ_array[v1, i, v2, j, w]
    #                       * Γ.basis_f1[vvw_Γ[ifreq][1], v1]
    #                       * Γ.basis_f2[vvw_Γ[ifreq][2], v2]
    #                       * Γ.basis_b[vvw_Γ[ifreq][3], w]

    # Use sparsity of coeffs to optimize.
    Γ_vvw .= 0
    @inbounds for ib in 1:nb_b(Γ)
        coeff_b[ib] == 0 && continue
        for j in 1:nind^2, iv2 in 1:nb_f2(Γ)
            coeff_f2[iv2] == 0 && continue
            jj = iv2 + (j - 1) * nb_f2(Γ)
            for i in 1:nind^2, iv1 in 1:nb_f1(Γ)
                coeff_f1[iv1] == 0 && continue
                ii = iv1 + (i - 1) * nb_f1(Γ)
                Γ_vvw[i, j] += Γ.data[ii, jj, ib] * coeff_f1[iv1] * coeff_f2[iv2] * coeff_b[ib]
            end
        end
    end
    Γ_vvw
end


"""
    to_matrix(Γ::Vertex4P{F, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, C::Symbol=get_channel(Γ)) where {F, T}
Evaluate a 4-point vertex at given bosonic frequency `w`, fermionic bases `basis1` and
`basis2`, and channel `c`. Return the matrix form.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`, `j`: Orbital/Keldysh index
- Input `Γ.data`: `(a, i, j), (a', i', j'), b`
- Output: `(a, i, j), (a', i', j')`
"""
function to_matrix(Γ::Vertex4P{F, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, C::Symbol=get_channel(Γ)) where {F, T}
    C_Γ = get_channel(Γ)
    if C !== C_Γ && (ntails(basis1) > 0 || ntails(basis2) > 0)
        # If we map to a different channel, the tails lead to problems so we disable it.
        # There is a problem with the fitting process that it oversamples regions where
        # v1-v2 or v1+v2 is small. Also, there is a problem for the tail-tail part. For
        # example, the 1/(v1-v2) term cannot be written as a sum of 1/v1 and 1/v2 terms.
        error("to_matrix to different channel with tails do not work.")
    end

    if C === C_Γ && basis1 === Γ.basis_f1 && basis2 === Γ.basis_f2
        # Same channel, same basis. Just need to contract the bosonic frequency basis.
        coeff_w = Γ.basis_b[w, :]
        @ein Γ_w[aij1, aij2] := Γ.data[aij1, aij2, b] * coeff_w[b]
        Γ_w = Γ_w::Matrix{T}
        return Γ_w
    else
        nind = get_nind(Γ)
        vs1 = get_fitting_points(basis1)
        vs2 = get_fitting_points(basis2)

        v1_ = vec(ones(length(vs2))' .* vs1)
        v2_ = vec(vs2' .* ones(length(vs1)))
        w_ = fill(w, length(v1_))
        Γ_w_data = reshape(Γ(v1_, v2_, w_, C), (length(vs1), length(vs2), nind^2, nind^2))

        Γ_tmp1 = fit_basis_coeff(Γ_w_data, basis1, vs1, 1)
        Γ_tmp2 = fit_basis_coeff(Γ_tmp1, basis2, vs2, 2)

        Γ_w = reshape(permutedims(Γ_tmp2, (1, 3, 2, 4)), size(basis1, 2) * nind^2, size(basis2, 2) * nind^2)
        return Γ_w
    end
end

_val_to_sym(::Val{C}) where {C} = C
_val_to_sym(C::Symbol) = C

"""
    apply_crossing(Γ::Vertex4P{F, C}) where {F, C}
Apply the crossing operation: swap indices 1 and 3. Works only for channel A and T.
"""
function apply_crossing(Γ::Vertex4P{F}) where {F}
    # For channel P, we need to additionally change v1 -> -v1. This is not implemented.
    C = get_channel(Γ)
    C === :P && error("Not implemented for channel P")
    C_out = channel_apply_crossing(C)
    # Multiply -1 to data to account the fermionic parity.
    Vertex4P{F}(C_out, Γ.basis_f1, Γ.basis_f2, Γ.basis_b, Γ.norb, .-Γ.data)
end

"""
    keldyshview(Γ::Vertex4P{:KF})
Return a the 4-point KF vertex as a 11-dimensional array in the standard index order.
- `a`: fermionic frequency basis index
- `b`: frequency basis index
- `i`: Orbital index
- `k`: Keldysh index
- Input `Γ.data`: `(a, i1, k1, i2, k2), (a', i3, k3, i4, k4), b`
- Output: `a, a', i1, i2, i3, i4, k1, k2, k3, k4, b`
"""
function keldyshview(Γ::Vertex4P{F}) where {F}
    norb = Γ.norb
    nk = nkeldysh(F)
    C = get_channel(Γ)
    data_size = (nb_f1(Γ), norb, nk, norb, nk, nb_f2(Γ), norb, nk, norb, nk, nb_b(Γ))
    # ((v, ik1, ik2), (v', ik3, ik4), w) -> (v, v', i1, i2, i3, i4, k1, k2, k3, k4, w)
    perm = [1, 6, 2, 4, 7, 9, 3, 5, 8, 10, 11]
    # channel orbital/Keldysh order -> standard orbital/Keldysh order
    permute!(perm, [1, 2, indices_to_standard(C, 3:6)..., indices_to_standard(C, 7:10)..., 11])
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

using SparseArrays

"""
    fit_basis_coeff(data, basis, grid, dim)
Fit the coefficients of the basis to the data to make ``data ≈ basis_value * coeff`` hold.
Use the `dim`-th index of `data` for fitting and leave other indices.
`size(data, dim) == length(grid)` must hold.
"""
function fit_basis_coeff(data, basis, grid, dim)
    if dim == 1
        # Reshape data to a Matrix by merging indices 2 to end, and call left division.
        basis_value = sparse(basis[grid, :])
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
