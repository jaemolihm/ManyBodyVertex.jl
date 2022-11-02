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

function to_matrix(Γ::CachedVertex4P{F, CΓ, T}, w, basis1=Γ.basis_f1, basis2=Γ.basis_f2, c::Val=Val(CΓ)) where {F, CΓ, T}
    c === Val(CΓ) || error("Invalid channel for CachedVertex4P")
    basis1 === Γ.basis_f1 || error("Invalid basis1 for CachedVertex4P")
    basis2 === Γ.basis_f2 || error("Invalid basis2 for CachedVertex4P")
    iw = findfirst(x -> x ≈ w, Γ.ws)
    iw === nothing && error("w not found in Γ.ws")
    Γ.data[iw]
end

"""
    cache_vertex_matrix(Γ, C, ws, basis_aux) => CachedVertex4P
Compute the matrix representation of `Γ` in channel `C` for bosonic frequencies `ws` and
store them in a `CachedVertex4P`. If `Γ` is in a different channel than `C`, use `basis_aux`.
If in the same channel, use the bases of `Γ`.

If input `Γ` is a list of vertices, add up all the matrices.
"""
cache_vertex_matrix(Γ::AbstractVertex4P, C, ws, basis_aux) = cache_vertex_matrix([Γ], C, ws, basis_aux)

function cache_vertex_matrix(Γs::AbstractVector, C, ws, basis_aux)
    Γ = first(Γs)
    basis_f1, basis_f2 = channel(Γ) === C ? (Γ.basis_f1, Γ.basis_f2) : (basis_aux, basis_aux)
    data = map(ws) do w
        mapreduce(Γ -> to_matrix(Γ, w, basis_f1, basis_f2, Val(C)), .+, Γs)
    end
    F = get_formalism(Γ)
    CachedVertex4P{F, C}(data, ws, basis_f1, basis_f2, Γ.norb)
end
