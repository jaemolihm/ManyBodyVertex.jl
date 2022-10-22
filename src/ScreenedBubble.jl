"""
    ScreenedBubble(Π, K1)
Bubble screened by a K1 vertex: ``Πscr = Π + Π * K1 * Π``.
### Diagram for `to_matrix(Π)`:
                Π                             Π -- K1 -- Π
                |                             |          |
                bΠ             +              bΠ         bΠ
                |                             |          |
    -- bL2 -- ov_LR -- bR1 --     -- bL2 -- ov_L        ov_R -- bR1 --
"""
mutable struct ScreenedBubble{F, T, BT <: AbstractBubble{F, T}, VT} <: AbstractBubble{F, T}
    # Basis for fermionic frequencies
    Π::BT
    # K1 vertex for screening
    K1::VT
    # Cached basis and overlap
    cache_basis_L
    cache_basis_R
    cache_overlap_LR
    cache_overlap_L
    cache_overlap_R
    function ScreenedBubble(Π::AbstractBubble{F, T}, K1::AbstractVertex4P{F, C, T}) where {F, C, T}
        K1.basis_f1 isa ConstantBasis || error("K1.basis_f1 is not a ConstantBasis")
        K1.basis_f2 isa ConstantBasis || error("K1.basis_f2 is not a ConstantBasis")
        new{F, T, typeof(Π), typeof(K1)}(Π, K1, nothing, nothing, nothing, nothing, nothing)
    end
end

function Base.getproperty(a::ScreenedBubble, s::Symbol)
    if s === :basis_f || s === :basis_b || s === :norb || s === :data
        getfield(getfield(a, :Π), s)
    else
        getfield(a, s)
    end
end

function cache_and_load_overlaps(Π::ScreenedBubble, basis_L::Basis, basis_R::Basis)
    if basis_L !== Π.cache_basis_L || basis_R !== Π.cache_basis_R
        Π.cache_overlap_LR = basis_integral(basis_L, basis_R, Π.basis_f)
        Π.cache_overlap_L = basis_integral(basis_L, Π.basis_f)
        Π.cache_overlap_R = basis_integral(basis_R, Π.basis_f)
        Π.cache_basis_L = basis_L
        Π.cache_basis_R = basis_R
    end
    Π.cache_overlap_LR, Π.cache_overlap_L, Π.cache_overlap_R
end

function to_matrix(Π::ScreenedBubble{F, T}, w, ov_LR, ov_L, ov_R) where {F, T}
    nb_L, nb_R = size(ov_LR)[1:2]
    nind2 = get_nind(Π)^2

    # Π part
    Π_mat = mfRG.to_matrix(Π.Π, w, ov_LR)

    # Π * K1 * Π part
    Π_w = Π(w)
    K1_w = Π.K1(0, 0, w)
    @ein Π_vertex_tmp1[xL, ij1, ij2] := ov_L[xL, a] * Π_w[a, ij1, ij2]
    @ein Π_vertex_tmp2[ij1, xR, ij2] := ov_R[xR, a] * Π_w[a, ij1, ij2]
    Π_vertex_tmp1 = Π_vertex_tmp1::Array{T, 3}
    Π_vertex_tmp2 = Π_vertex_tmp2::Array{T, 3}
    Π_vertex1 = reshape(Π_vertex_tmp1, nb_L * nind2, nind2)
    Π_vertex2 = reshape(Π_vertex_tmp2, nind2, nb_R * nind2)
    Π_mat .+= (Π_vertex1 * K1_w * Π_vertex2) .* integral_coeff(Π)

    Π_mat
end
