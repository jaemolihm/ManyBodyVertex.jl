"""
# Orbital and Keldysh indices
The "standard" ordering of indices is ``G[1,2,3,4] = G[d₁, d₂†, d₃, d₄†]``.
Specific channels use different ordering of indices:
- A channel: ``G{:A}[1,2,3,4] = G[1,2,3,4]``
- P channel: ``G{:P}[1,2,3,4] = G[1,3,2,4]``
- T channel: ``G{:T}[1,2,3,4] = G[3,2,1,4]``
To convert between the standard and channel orderings,u se the `indices_to_standard` and
`indices_to_channel` functions.

Diagrammatically, the channel orderings are represented as follows.
```
Γ_a(1234): 2 3   Γ_p(1234): 3 2   Γ_t(1234): 2 1
            X                X                X
           1 4              1 4              3 4
```

The reason to use different ordering for each channel is to make the matrix representation
for BSE can be obtained by just flattening without additional permutation.

# Frequency parametrization
The frequency parametrization reads
- A channel: ``(v1, v2; w) = (v1-w/2, -v1-w/2,  v2+w/2, -v2+w/2)``
- P channel: ``(v1, v2; w) = (v1+w/2,  v2-w/2, -v1+w/2, -v2-w/2)``
- T channel: ``(v1, v2; w) = (v2+w/2, -v1-w/2,  v1-w/2, -v2+w/2)``
(Frequency is positive for an outgoing leg.)
Ref: Gievers et al, Eur. Phys. J. B 95, 108 (2022), Fig. 3

# Matsubara frequencies
Matsubara frequencies at finite temperature are discrete. We index the frequencies as
- Bosonic   frequencies: w_n = 2π/β * n
- Fermionic frequencies: v_n = 2π/β * (n + 1/2)

For Matsubara frequencies with odd bosonic frequency index, special care needs to be taken.
To get the standard parametrization, we floor for the outgoing legs and ceil for the
incoming legs, respectively. To get the channel parametrization, we ceil both frequencies
so that `frequency_to_channel` becomes the inverse of `frequency_to_standard`.

To convert between the standard and channel orderings, use the `frequency_to_standard` and
`frequency_to_channel` functions.
"""

const _RealFreq = Union{Val{:KF}, Val{:ZF}}
const _ImagFreq = Union{Val{:MF}}

indices_to_standard(::Val{:A}, i) = (i[1], i[2], i[3], i[4])
indices_to_standard(::Val{:P}, i) = (i[1], i[3], i[2], i[4])
indices_to_standard(::Val{:T}, i) = (i[3], i[2], i[1], i[4])

indices_to_channel(::Val{:A}, i) = (i[1], i[2], i[3], i[4])
indices_to_channel(::Val{:P}, i) = (i[1], i[3], i[2], i[4])
indices_to_channel(::Val{:T}, i) = (i[3], i[2], i[1], i[4])

# Real frequencies
frequency_to_standard(::_RealFreq, ::Val{:A}, v1, v2, w) = (v1-w/2, -v1-w/2,  v2+w/2, -v2+w/2)
frequency_to_standard(::_RealFreq, ::Val{:P}, v1, v2, w) = (v1+w/2,  v2-w/2, -v1+w/2, -v2-w/2)
frequency_to_standard(::_RealFreq, ::Val{:T}, v1, v2, w) = (v2+w/2, -v1-w/2,  v1-w/2, -v2+w/2)

# v1 + v2 + v3 + v4 = 0 is assumed to hold.
frequency_to_channel(::_RealFreq, ::Val{:A}, v1, v2, v3, v4) = ((v1-v2)/2, (v3-v4)/2, v3+v4)
frequency_to_channel(::_RealFreq, ::Val{:P}, v1, v2, v3, v4) = ((v1-v3)/2, (v2-v4)/2, v1+v3)
frequency_to_channel(::_RealFreq, ::Val{:T}, v1, v2, v3, v4) = ((v3-v2)/2, (v1-v4)/2, v1+v4)
frequency_to_channel(F, C, v1234::NTuple{4}) = frequency_to_channel(F, C, v1234...)

# Imaginary frequencies
@inline function frequency_to_standard(::_ImagFreq, ::Val{:A}, v1, v2, w)
    (floor(Int, v1-w/2), ceil(Int, -v1-1-w/2), floor(Int, v2+w/2), ceil(Int, -v2-1+w/2))
end
@inline function frequency_to_standard(::_ImagFreq, ::Val{:P}, v1, v2, w)
    (floor(Int, v1+w/2), floor(Int, v2-w/2), ceil(Int, -v1-1+w/2), ceil(Int, -v2-1-w/2))
end
@inline function frequency_to_standard(::_ImagFreq, ::Val{:T}, v1, v2, w)
    (floor(Int, v2+w/2), ceil(Int, -v1-1-w/2), floor(Int, v1-w/2), ceil(Int, -v2-1+w/2))
end

# v1 + v2 + v3 + v4 = -2 is assumed to hold.
@inline function frequency_to_channel(::_ImagFreq, ::Val{:A}, v1, v2, v3, v4)
    (ceil(Int, (v1-v2-1)/2), ceil(Int, (v3-v4-1)/2), v3+v4+1)
end
@inline function frequency_to_channel(::_ImagFreq, ::Val{:P}, v1, v2, v3, v4)
    (ceil(Int, (v1-v3-1)/2), ceil(Int, (v2-v4-1)/2), v1+v3+1)
end
@inline function frequency_to_channel(::_ImagFreq, ::Val{:T}, v1, v2, v3, v4)
    (ceil(Int, (v3-v2-1)/2), ceil(Int, (v1-v4-1)/2), v1+v4+1)
end

"""
    _permute_orbital_indices_matrix_4p(c_in, c_out, Γ_mat_in, nind)
Permute orbital indices for a 2-by-2 matrix representation of a 4-point vertex `Γ_mat_in`.
"""
function _permute_orbital_indices_matrix_4p(c_in, c_out, Γ_mat_in, nind)
    c_in === c_out && return Γ_mat_in
    x1 = Base.ReshapedArray(Γ_mat_in, (nind, nind, nind, nind), ())
    # channel C -> standard -> channel c_out
    inds = indices_to_channel(c_out, indices_to_standard(c_in, (1, 2, 3, 4)))
    x2 = permutedims(x1, inds)
    x3 = Base.ReshapedArray(x2, (nind^2, nind^2), ())
    collect(x3)
end

"""
Same as `_permute_orbital_indices_matrix_4p`, but keep the 1st dimension.
"""
function _permute_orbital_indices_matrix_4p_keep_dim1(c_in, c_out, Γ_mat_in, nind)
    c_in === c_out && return Γ_mat_in
    nkeep = size(Γ_mat_in, 1)
    x1 = Base.ReshapedArray(Γ_mat_in, (nkeep, nind, nind, nind, nind), ())
    # channel C -> standard -> channel c_out
    inds = indices_to_channel(c_out, indices_to_standard(c_in, (2, 3, 4, 5)))
    x2 = permutedims(x1, (1, inds...))
    x3 = Base.ReshapedArray(x2, (nkeep, nind^2, nind^2), ())
    collect(x3)
end

"""
# Bubbles
- ``Πᴬ_{12,34}(v; w) =       G_41( v + w/2) * G_23(v - w/2)``
- ``Πᴾ_{12,34}(v; w) = 1/2 * G_14(-v + w/2) * G_23(v - w/2)``
- ``Πᵀ_{12,34}(v; w) =  -1 * G_41( v + w/2) * G_23(v - w/2) = -Πᴬ_{12,34}(v; w)``

These equations are implemented in `_bubble_prefactor`, `_bubble_frequencies`, and
`_bubble_indices`. Use these functions when generating bubble objects.

For imaginary frequencies, we floor (ceil) to get the integer index for fermionic frequencies
if the sign for the fermionic frequency is positive (negative), as done in the vertex.
"""

_bubble_prefactor(::Val{:A}) = 1
_bubble_prefactor(::Val{:P}) = 1/2
_bubble_prefactor(::Val{:T}) = -1

_bubble_frequencies(::_RealFreq, ::Val{:A}, v, w) = ( v+w/2, v-w/2)
_bubble_frequencies(::_RealFreq, ::Val{:P}, v, w) = (-v+w/2, v+w/2)
_bubble_frequencies(::_RealFreq, ::Val{:T}, v, w) = ( v+w/2, v-w/2)
_bubble_frequencies(::_ImagFreq, ::Val{:A}, v, w) = (floor(Int,   v+w/2), floor(Int, v-w/2))
_bubble_frequencies(::_ImagFreq, ::Val{:P}, v, w) = (ceil(Int, -v-1+w/2), floor(Int, v+w/2))
_bubble_frequencies(::_ImagFreq, ::Val{:T}, v, w) = (floor(Int,   v+w/2), floor(Int, v-w/2))

_bubble_frequencies_inv(::_RealFreq, ::Val{:A}, v1, v2) = ((v1 + v2) / 2, v1 - v2)
_bubble_frequencies_inv(::_RealFreq, ::Val{:P}, v1, v2) = ((v2 - v1) / 2, v1 + v2)
_bubble_frequencies_inv(::_RealFreq, ::Val{:T}, v1, v2) = ((v1 + v2) / 2, v1 - v2)
_bubble_frequencies_inv(::_ImagFreq, ::Val{:A}, v1, v2) = (fld(v1 + v2 + 1, 2), v1 - v2)
_bubble_frequencies_inv(::_ImagFreq, ::Val{:P}, v1, v2) = (fld(v2 - v1, 2), v1 + v2 + 1)
_bubble_frequencies_inv(::_ImagFreq, ::Val{:T}, v1, v2) = (fld(v1 + v2 + 1, 2), v1 - v2)

_bubble_indices(::Val{:A}, i) = (i[4], i[1], i[2], i[3])
_bubble_indices(::Val{:P}, i) = (i[1], i[4], i[2], i[3])
_bubble_indices(::Val{:T}, i) = (i[4], i[1], i[2], i[3])
