"""
    ScreenedBubble(Π, K1)
Bubble screened by a K1 vertex: ``Πscr = Π + Π * (Γ0 + K1) * Π``.
### Diagram for `to_matrix(Π)`:
                Π                             Π -- (Γ0 + K1) -- Π
                |                             |                 |
                bΠ             +              bΠ                bΠ
                |                             |                 |
    -- bL2 -- ov_LR -- bR1 --     -- bL2 -- ov_L               ov_R -- bR1 --
"""
mutable struct ScreenedBubble{F, T, BT <: AbstractBubble{F, T}, VT0, VT1} <: AbstractBubble{F, T}
    # Channel
    channel::Symbol
    # Basis for fermionic frequencies
    Π::BT
    # Γ0 vertex for screening
    Γ0::VT0
    # K1 vertex for screening
    K1::VT1
    # Cached basis and overlap
    cache_basis_L
    cache_basis_R
    cache_overlap_LR
    cache_overlap_L
    cache_overlap_R
    function ScreenedBubble(Π::AbstractBubble{F, T}, Γ0::AbstractVertex4P{F}, K1::AbstractVertex4P{F, T}) where {F, T}
        C = get_channel(Π)
        get_channel(Γ0) == C || throw(ArgumentError("channel does not match between Γ0 and Π"))
        get_channel(K1) == C || throw(ArgumentError("channel does not match between K1 and Π"))
        Γ0.basis_f1 isa Union{ConstantBasis, ImagConstantBasis} || error("Γ0.basis_f1 is not a ConstantBasis")
        Γ0.basis_f2 isa Union{ConstantBasis, ImagConstantBasis} || error("Γ0.basis_f2 is not a ConstantBasis")
        K1.basis_f1 isa Union{ConstantBasis, ImagConstantBasis} || error("K1.basis_f1 is not a ConstantBasis")
        K1.basis_f2 isa Union{ConstantBasis, ImagConstantBasis} || error("K1.basis_f2 is not a ConstantBasis")
        new{F, T, typeof(Π), typeof(Γ0), typeof(K1)}(C, Π, Γ0, K1, nothing, nothing, nothing, nothing, nothing)
    end
end

get_channel(Π::ScreenedBubble) = Π.channel

function Base.getproperty(a::ScreenedBubble, s::Symbol)
    if s === :basis_f || s === :basis_b || s === :norb || s === :data || s === :temperature
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

function to_matrix(Π::ScreenedBubble, w, ov_LR, ov_L, ov_R)
    nb_L, nb_R = size(ov_LR)[1:2]
    nind2 = get_nind(Π)^2

    # Π part
    Π_mat = to_matrix(Π.Π, w, ov_LR)

    # Π * (Γ0 + K1) * Π part
    Π_w = Π(w)
    K1_w = Π.Γ0(0, 0, w) + Π.K1(0, 0, w)
    @ein Π_vertex_tmp1[xL, ij1, ij2] := ov_L[xL, a] * Π_w[a, ij1, ij2]
    @ein Π_vertex_tmp2[ij1, xR, ij2] := ov_R[xR, a] * Π_w[a, ij1, ij2]
    Π_vertex_tmp1 = Π_vertex_tmp1::Array{eltype(Π), 3}
    Π_vertex_tmp2 = Π_vertex_tmp2::Array{eltype(Π), 3}
    Π_vertex1 = reshape(Π_vertex_tmp1, nb_L * nind2, nind2)
    Π_vertex2 = reshape(Π_vertex_tmp2, nind2, nb_R * nind2)
    Π_mat .+= (Π_vertex1 * K1_w * Π_vertex2) .* integral_coeff(Π)^2

    Π_mat
end
