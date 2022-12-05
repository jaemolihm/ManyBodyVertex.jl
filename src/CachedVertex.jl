"""
Precompute and cache the result of `to_matrix`. The values can be accessed via the
`to_matrix` function.
"""
struct CachedVertex4P{F, C, T, BF1, BF2, FT} <: AbstractVertex4P{F, C, T}
    data::Vector{Array{T, 2}}
    ws::Vector{FT}
    basis_f1::BF1
    basis_f2::BF2
    norb::Int
    function CachedVertex4P{F, C}(data, ws, basis_f1::BF1, basis_f2::BF2, norb) where {F, C, BF1, BF2}
        T = eltype(eltype(data))
        inds = sortperm(ws)
        new{F, C, T, BF1, BF2, eltype(ws)}(data[inds], ws[inds], basis_f1, basis_f2, norb)
    end
end

nb_f1(Γ::CachedVertex4P) = size(Γ.basis_f1, 2)
nb_f2(Γ::CachedVertex4P) = size(Γ.basis_f2, 2)

function Base.show(io::IO, Γ::CachedVertex4P{F, C}) where {F, C}
    print(io, Base.typename(typeof(Γ)).wrapper, "{:$F, :$C}")
    print(io, "(nbasis_f1=$(nb_f1(Γ)), nbasis_f2=$(nb_f2(Γ)), nw=$(length(Γ.ws)), ")
    print(io, "norb=$(Γ.norb), data=$(Base.summary(first(Γ.data))))")
end

function Base.similar(Γ::CachedVertex4P{F, C, T}, ::Type{ElType}=T) where {F, C, T, ElType}
    data = [zeros(ElType, size(x)) for x in Γ.data]
    CachedVertex4P{F, C}(data, Γ.ws, Γ.basis_f1, Γ.basis_f2, Γ.norb)
end
Base.zero(Γ::CachedVertex4P) = (Γ_new = similar(Γ); for x in Γ_new.data; x .= 0; end; Γ_new)

function to_matrix(Γ::CachedVertex4P{F, CΓ, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, c::Val=Val(CΓ)) where {F, CΓ, T}
    c === Val(CΓ) || error("Invalid channel for CachedVertex4P")
    basis1 === Γ.basis_f1 || error("Invalid basis1 for CachedVertex4P")
    basis2 === Γ.basis_f2 || error("Invalid basis2 for CachedVertex4P")
    iw = findfirst(x -> x ≈ w, Γ.ws)
    iw === nothing && error("w not found in Γ.ws")
    Γ.data[iw]
end

"""
    cache_vertex_matrix(Γ, C, ws, basis_aux1, basis_aux2) => CachedVertex4P
Compute the matrix representation of `Γ` in channel `C` for bosonic frequencies `ws` and
store them in a `CachedVertex4P`. If `Γ` is in a different channel than `C`, use
`basis_aux1` and `basis_aux2`.
If in the same channel, use the bases of `Γ`.

If input `Γ` is a list of vertices, add up all the matri1ces.
"""
function cache_vertex_matrix(Γ::AbstractVertex4P, C, ws, basis_aux1=nothing, basis_aux2=basis_aux1)
    cache_vertex_matrix([Γ], C, ws, basis_aux1, basis_aux2)
end

function cache_vertex_matrix(Γs::AbstractVector, C, ws, basis_aux1=nothing, basis_aux2=basis_aux1)
    isempty(Γs) && return nothing
    if any(channel.(Γs) .!= C) && basis_aux1 === nothing
        error("For vertex with different channel than C=$C, basis_aux must be provided")
    end

    Γ = first(Γs)
    basis_f1, basis_f2 = channel(Γ) === C ? (Γ.basis_f1, Γ.basis_f2) : (basis_aux1, basis_aux2)
    nind = get_nind(Γ)
    data = [zeros(eltype(first(Γs)), nind^2 * nbasis(basis_f1), nind^2 * nbasis(basis_f2)) for _ in eachindex(ws)]
    Base.Threads.@threads for iw in eachindex(ws)
        for Γ in Γs
            data[iw] .+= to_matrix(Γ, ws[iw], basis_f1, basis_f2, Val(C))
        end
    end
    F = get_formalism(Γ)
    CachedVertex4P{F, C}(data, ws, basis_f1, basis_f2, Γ.norb)
end
