"""
Precompute and cache the result of `to_matrix`. The values can be accessed via the
`to_matrix` function.
"""
struct CachedVertex4P{F, T, BF1, BF2, FT} <: AbstractVertex4P{F, T}
    channel::Symbol
    data::Array{T, 3}
    ws::Vector{FT}
    basis_f1::BF1
    basis_f2::BF2
    norb::Int
    function CachedVertex4P{F}(channel, data, ws, basis_f1::BF1, basis_f2::BF2, norb) where {F, BF1, BF2}
        channel ∈ (:A, :P, :T) || throw(ArgumentError("Wrong channel $channel"))
        T = eltype(eltype(data))
        new{F, T, BF1, BF2, eltype(ws)}(channel, data, ws, basis_f1, basis_f2, norb)
    end
end

get_channel(Γ::CachedVertex4P) = Γ.channel

data_fieldnames(::Type{<:CachedVertex4P}) = (:data,)
function _check_basis_identity(A::CachedVertex4P, B::CachedVertex4P)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    get_channel(A) === get_channel(B) || error("Different channel")
    A.ws ≈ B.ws || error("Different ws")
    A.basis_f1 === B.basis_f1 || error("Different basis_f1")
    A.basis_f2 === B.basis_f2 || error("Different basis_f2")
end

nb_f1(Γ::CachedVertex4P) = size(Γ.basis_f1, 2)
nb_f2(Γ::CachedVertex4P) = size(Γ.basis_f2, 2)

function Base.show(io::IO, Γ::CachedVertex4P{F}) where {F}
    C = get_channel(Γ)
    print(io, Base.typename(typeof(Γ)).wrapper, "{:$F}")
    print(io, "(channel=$C, nbasis_f1=$(nb_f1(Γ)), nbasis_f2=$(nb_f2(Γ)), ")
    print(io, "nw=$(length(Γ.ws)), norb=$(Γ.norb), data=$(Base.summary(Γ.data)))")
end

function Base.similar(Γ::CachedVertex4P{F, T}, ::Type{ElType}=T) where {F, T, ElType}
    C = get_channel(Γ)
    CachedVertex4P{F}(C, similar(Γ.data, ElType), Γ.ws, Γ.basis_f1, Γ.basis_f2, Γ.norb)
end
Base.zero(Γ::CachedVertex4P) = (Γ_new = similar(Γ); Γ_new.data .= 0; Γ_new)

function to_matrix(Γ::CachedVertex4P{F, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, C::Symbol=get_channel(Γ)) where {F, T}
    CΓ = get_channel(Γ)
    C === CΓ || error("Invalid channel for CachedVertex4P")
    basis1 === Γ.basis_f1 || error("Invalid basis1 for CachedVertex4P")
    basis2 === Γ.basis_f2 || error("Invalid basis2 for CachedVertex4P")
    iw = findfirst(x -> x ≈ w, Γ.ws)
    iw === nothing && error("w not found in Γ.ws")
    view(Γ.data, :, :, iw)
end

"""
    cache_vertex_matrix(Γ, C, ws, basis_aux1, basis_aux2) => CachedVertex4P
Compute the matrix representation of `Γ` in channel `C` for bosonic frequencies `ws` and
store them in a `CachedVertex4P`. If `Γ` is in a different channel than `C`, use
`basis_aux1` and `basis_aux2`.
If in the same channel, use the bases of `Γ`.

If input `Γ` is a list of vertices, add up all the matri1ces.
"""
function cache_vertex_matrix(Γ::AbstractFrequencyVertex, C, ws, basis_aux1=nothing, basis_aux2=basis_aux1)
    cache_vertex_matrix([Γ], C, ws, basis_aux1, basis_aux2)
end

function cache_vertex_matrix(Γs::AbstractVector, C, ws, basis_aux1=nothing, basis_aux2=basis_aux1)
    isempty(Γs) && return nothing
    if any(get_channel.(Γs) .!= C) && basis_aux1 === nothing
        error("For vertex with different channel than C=$C, basis_aux must be provided")
    end
    if basis_aux1 isa NamedTuple{(:freq,), Tuple{T}} where {T<:Basis}
        basis_aux1_ = basis_aux1.freq
        basis_aux2_ = basis_aux2.freq
    else
        basis_aux1_ = basis_aux1
        basis_aux2_ = basis_aux2
    end

    Γ = first(Γs)
    basis_f1, basis_f2 = get_channel(Γ) === C ? (Γ.basis_f1, Γ.basis_f2) : (basis_aux1_, basis_aux2_)
    nind = get_nind(Γ)
    data = zeros(eltype(first(Γs)), nind^2 * nbasis(basis_f1), nind^2 * nbasis(basis_f2), length(ws))
    Base.Threads.@threads for iw in eachindex(ws)
        @views for Γ in Γs
            data[:, :, iw] .+= to_matrix(Γ, ws[iw], basis_f1, basis_f2, C)
        end
    end
    F = get_formalism(Γ)
    CachedVertex4P{F}(C, data, ws, basis_f1, basis_f2, Γ.norb)
end
