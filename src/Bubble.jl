abstract type AbstractBubble{F, C, T} <: AbstractFrequencyVertex{F, T} end
channel(::T) where {T <:AbstractBubble} = channel(T)
channel(::Type{T}) where {T <: AbstractBubble{F, C}} where {F, C} = C

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
    function Bubble{F, C}(basis_f::BF, basis_b::BB, norb, data::DT; temperature=nothing,
        cache_basis_L=nothing, cache_basis_R=nothing, cache_overlap_LR=nothing) where {F, C, DT <: AbstractArray{T}, BF, BB} where {T}
        F === :MF && temperature === nothing && error("For MF, temperature must be set")
        new{F, C, T, BF, BB, DT}(basis_f, basis_b, norb, data, temperature, cache_basis_L, cache_basis_R, cache_overlap_LR)
    end
end
data_fieldnames(::Type{<:Bubble}) = (:data,)

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

function _check_basis_identity(A::Bubble, B::Bubble)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    channel(A) === channel(B) || error("Different channel")
    A.basis_f === B.basis_f || error("Different basis_f")
    A.basis_b === B.basis_b || error("Different basis_b")
end

function Base.show(io::IO, Π::AbstractBubble{F, C}) where {F, C}
    print(io, Base.typename(typeof(Π)).wrapper, "{:$F, :$C}")
    print(io, "(nbasis_f=$(nb_f(Π)), nbasis_b=$(nb_b(Π)), ")
    print(io, "norb=$(Π.norb), data=$(Base.summary(Π.data)))")
end

nb_f(Π::AbstractBubble) = size(Π.basis_f, 2)
nb_b(Π::AbstractBubble) = size(Π.basis_b, 2)

function (Π::AbstractBubble)(w)
    # Evaluate the bubble at given bosonic frequency w
    # Output: a, (i, j), (i', j')
    Π_w = zeros(eltype(Π), size(Π.data)[1:3])
    @inbounds @views for ib in 1:nb_b(Π)
        coeff_w = Π.basis_b[w, ib]
        coeff_w === 0 && continue
        Π_w .+= Π.data[:, :, :, ib] .* coeff_w
    end
    Π_w
end

"""
Load overlap from cache, recompute if basis has changed.
"""
function cache_and_load_overlaps(Π::AbstractBubble, basis_L::Basis, basis_R::Basis)
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
    # Implement optimized version
    # @ein Π_vertex_tmp[x1, x2, ij1, ij2] := overlap[x1, x2, a] * Π_w[a, ij1, ij2]
    A = Base.ReshapedArray(overlap, (nv_Γ1*nv_Γ2, nb_f(Π)), ())
    B = Base.ReshapedArray(Π_w, (nb_f(Π), nind2^2), ())
    Π_vertex_tmp = Base.ReshapedArray(A * B, (nv_Γ1, nv_Γ2, nind2, nind2), ())
    Π_vertex = reshape(PermutedDimsArray(Π_vertex_tmp, (1, 3, 2, 4)), nv_Γ1 * nind2, nv_Γ2 * nind2)
    collect(Π_vertex) .* integral_coeff(Π)
end


integral_coeff(::Union{Val{:KF}, Val{:ZF}}, _) = -im / 2π
integral_coeff(::Val{:MF}, temperature) = temperature
integral_coeff(::Val{:MF}, ::Nothing) = error("For MF, temperature must be provided")
integral_coeff(Π::AbstractBubble{F}) where {F} = eltype(Π)(integral_coeff(Val(F), Π.temperature))::eltype(Π)
