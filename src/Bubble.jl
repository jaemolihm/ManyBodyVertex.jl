abstract type AbstractBubble{F, C, T} <: AbstractFrequencyVertex{F, T} end

"""
    Bubble{F}(basis_f, basis_b, norb, data)
### Diagram for `to_matrix(Π)`:
                Π
                |
                bΠ
                |
    -- bL2 -- ov_LR -- bR1 --
"""
mutable struct Bubble{F, C, T, BF, BB, DT <: AbstractArray{T}} <: AbstractBubble{F, C, T}
    # Basis for fermionic frequencies
    basis_f::BF
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data array
    data::DT
    # Temperature (needed for integral_coeff in MF)
    temperature
    # Cached basis and overlap
    cache_basis_L
    cache_basis_R
    cache_overlap_LR
    function Bubble{F, C}(basis_f::BF, basis_b::BB, norb, data::DT; temperature=nothing) where {F, C, DT <: AbstractArray{T}, BF, BB} where {T}
        F === :MF && temperature === nothing && error("For MF, temperature must be set")
        new{F, C, T, BF, BB, DT}(basis_f, basis_b, norb, data, temperature, nothing, nothing, nothing)
    end
end

Bubble{F, C}(basis_f, basis_b, norb=1; temperature=nothing) where {F, C} = Bubble{F, C}(ComplexF64, basis_f, basis_b, norb; temperature)

function Bubble{F, C}(::Type{T}, basis_f, basis_b, norb=1; temperature=nothing) where {F, C, T}
    nb_f = size(basis_f, 2)
    nb_b = size(basis_b, 2)
    nk = nkeldysh(F)
    data = zeros(T, nb_f, (norb * nk)^2, (norb * nk)^2, nb_b)
    Bubble{F, C}(basis_f, basis_b, norb, data; temperature)
end

function Base.similar(Π::Bubble{F, C, T}, ::Type{ElType}=T) where {F, C, T, ElType}
    Bubble{F, C}(Π.basis_f, Π.basis_b, Π.norb, similar(Π.data, ElType); Π.temperature)
end

function Base.:*(x::Number, A::Bubble)
    B = similar(A)
    B.data .= A.data .* x
    B
end
Base.:*(A::Bubble, x::Number) = x * A

function Base.show(io::IO, Π::AbstractBubble{F, C}) where {F, C}
    print(io, Base.typename(typeof(Π)).wrapper, "{:$F, :$C}")
    print(io, "(nbasis_f=$(nb_f(Π)), nbasis_b=$(nb_b(Π)), ")
    print(io, "norb=$(Π.norb), data=$(Base.summary(Π.data)))")
end

channel(::AbstractBubble{F, C}) where {F, C} = C
nb_f(Π::AbstractBubble) = size(Π.basis_f, 2)
nb_b(Π::AbstractBubble) = size(Π.basis_b, 2)

function (Π::AbstractBubble)(w)
    # Evaluate the bubble at given bosonic frequency w
    # Output: a, (i, j), (i', j')
    coeff_w = Π.basis_b[w, :]
    @ein Π_w[a, ij1, ij2] := Π.data[a, ij1, ij2, b] * coeff_w[b]
    Π_w::Array{eltype(Π), 3}
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
    to_matrix(Π, w, cache_and_load_overlaps(Π, basis_L, basis_R)...)::Array{eltype(Π), 2}
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
function to_matrix(Π::Bubble{F, C, T}, w, overlap) where {F, C, T}
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

integral_coeff(Π::AbstractBubble{:KF}) = -im / 2 / eltype(Π)(π)
integral_coeff(Π::AbstractBubble{:MF}) = eltype(Π)(Π.temperature)::eltype(Π)
integral_coeff(Π::AbstractBubble{:ZF}) = -im / 2 / eltype(Π)(π)


"""
    compute_bubble(G, basis_f, basis_b, ::Val{C}; temperature) where {C}
Compute the Bubble in channel `C` for the 2-point Green function `G`.
"""
function compute_bubble(G, basis_f, basis_b, ::Val{C}; temperature) where {C}
    F = get_formalism(G)
    nind = get_nind(G)
    Π = Bubble{F, C}(basis_f, basis_b, G.norb; temperature)
    vs = get_fitting_points(basis_f)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(eltype(Π.data), length(vs), nind^4, length(ws))

    for (iw, w) in enumerate(ws)
        for (iv, v) in enumerate(vs)
            v1, v2 = mfRG._bubble_frequencies(Val(F), Val(C), v, w)
            G1 = G(v1)
            G2 = G(v2)
            for (i, inds) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                i11, i12, i21, i22 = mfRG._bubble_indices(Val(C), inds)
                Π_data[iv, i, iw] = G1[i11, i12] * G2[i21, i22]
            end
        end
    end
    Π_data .*= mfRG._bubble_prefactor(Val(C))
    Π_data_tmp1 = mfRG.fit_basis_coeff(Π_data, basis_f, vs, 1)
    Π_data_tmp2 = mfRG.fit_basis_coeff(Π_data_tmp1, basis_b, ws, 3)
    Π.data .= reshape(Π_data_tmp2, size(Π.data))
    Π
end
