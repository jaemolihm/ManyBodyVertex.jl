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

function _check_basis_identity(A::Bubble, B::Bubble)
    channel(A) === channel(B) || error("channel must be identical")
    A.basis_f === B.basis_f || error("basis must be identical")
    A.basis_b === B.basis_b || error("basis must be identical")
end

function Base.:+(A::Bubble, B::Bubble)
    _check_basis_identity(A, B)
    C = similar(A)
    C.data .= A.data .+ B.data
    C
end

function Base.:-(A::Bubble, B::Bubble)
    _check_basis_identity(A, B)
    C = similar(A)
    C.data .= A.data .- B.data
    C
end

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


integral_coeff(::Union{Val{:KF}, Val{:ZF}}, _) = -im / 2π
integral_coeff(::Val{:MF}, temperature) = temperature
integral_coeff(Π::AbstractBubble{F}) where {F} = eltype(Π)(integral_coeff(Val(F), Π.temperature))::eltype(Π)
