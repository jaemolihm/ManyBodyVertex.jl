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

# Matsubara frequencies at finite temperature are discrete. We index the frequencies as
Bosonic   frequencies: w_n = 2π/β * n
Fermionic frequencies: v_n = 2π/β * (n + 1/2)

For Matsubara frequencies with odd bosonic frequency index, special care needs to be taken.
To get the standard parametrization, we floor for the outgoing legs and ceil for the
incoming legs, respectively. To get the channel parametrization, we ceil both frequencies
so that `frequency_to_channel` becomes the inverse of `frequency_to_standard`.

To convert between the standard and channel orderings, use the `frequency_to_standard` and
`frequency_to_channel` functions.
"""


indices_to_standard(::Val{:A}, i) = (i[1], i[2], i[3], i[4])
indices_to_standard(::Val{:P}, i) = (i[1], i[3], i[2], i[4])
indices_to_standard(::Val{:T}, i) = (i[3], i[2], i[1], i[4])

indices_to_channel(::Val{:A}, i) = (i[1], i[2], i[3], i[4])
indices_to_channel(::Val{:P}, i) = (i[1], i[3], i[2], i[4])
indices_to_channel(::Val{:T}, i) = (i[3], i[2], i[1], i[4])

# Real frequencies
frequency_to_standard(::Union{Val{:KF},Val{:ZF}}, ::Val{:A}, v1, v2, w) = (v1-w/2, -v1-w/2,  v2+w/2, -v2+w/2)
frequency_to_standard(::Union{Val{:KF},Val{:ZF}}, ::Val{:P}, v1, v2, w) = (v1+w/2,  v2-w/2, -v1+w/2, -v2-w/2)
frequency_to_standard(::Union{Val{:KF},Val{:ZF}}, ::Val{:T}, v1, v2, w) = (v2+w/2, -v1-w/2,  v1-w/2, -v2+w/2)

# v1 + v2 + v3 + v4 = 0 is assumed to hold.
frequency_to_channel(::Union{Val{:KF},Val{:ZF}}, ::Val{:A}, v1, v2, v3, v4) = ((v1-v2)/2, (v3-v4)/2, v3+v4)
frequency_to_channel(::Union{Val{:KF},Val{:ZF}}, ::Val{:P}, v1, v2, v3, v4) = ((v1-v3)/2, (v2-v4)/2, v1+v3)
frequency_to_channel(::Union{Val{:KF},Val{:ZF}}, ::Val{:T}, v1, v2, v3, v4) = ((v3-v2)/2, (v1-v4)/2, v1+v4)

# Imaginary frequencies
@inline function frequency_to_standard(::Val{:MF}, ::Val{:A}, v1, v2, w)
    (floor(Int, v1-w/2), ceil(Int, -v1-1-w/2), floor(Int, v2+w/2), ceil(Int, -v2-1+w/2))
end
@inline function frequency_to_standard(::Val{:MF}, ::Val{:P}, v1, v2, w)
    (floor(Int, v1+w/2), floor(Int, v2-w/2), ceil(Int, -v1-1+w/2), ceil(Int, -v2-1-w/2))
end
@inline function frequency_to_standard(::Val{:MF}, ::Val{:T}, v1, v2, w)
    (floor(Int, v2+w/2), ceil(Int, -v1-1-w/2), floor(Int, v1-w/2), ceil(Int, -v2-1+w/2))
end

# v1 + v2 + v3 + v4 = -2 is assumed to hold.
@inline function frequency_to_channel(::Val{:MF}, ::Val{:A}, v1, v2, v3, v4)
    (ceil(Int, (v1-v2-1)/2), ceil(Int, (v3-v4-1)/2), v3+v4+1)
end
@inline function frequency_to_channel(::Val{:MF}, ::Val{:P}, v1, v2, v3, v4)
    (ceil(Int, (v1-v3-1)/2), ceil(Int, (v2-v4-1)/2), v1+v3+1)
end
@inline function frequency_to_channel(::Val{:MF}, ::Val{:T}, v1, v2, v3, v4)
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
