using StaticArrays

_val_to_value(::Val{T}) where {T} = T

"""
    get_bare_vertex(U::Number, ::Val{F}, ::Val{C})
Bare vertex of an 1-orbital model with interaction `U`, formalism `F`, and channel `C`.
"""
function get_bare_vertex(U::Number, ::Val{F}, ::Val{C}) where {F, C}
    basis = F === :MF ? ImagConstantBasis() : ConstantBasis()
    Γ0 = Vertex4P{F, C}(basis, basis, basis, 1)
    if F === :KF
        Γ0_kv = vertex_keldyshview(Γ0)
        for ks in CartesianIndices((2, 2, 2, 2))
            k1, k2, k3, k4 = ks.I
            if mod(k1 + k2 + k3 + k4, 2) == 1
                Γ0_kv[:, :, 1, 1, 1, 1, ks, :] .= U / 2
            end
        end
    else
        Γ0.data .= U
    end
    Γ0
end


# SIAM: single-impurity Anderson model

"""
    siam_get_green_function(v, ::Val{F}; e, Δ, t, D=Inf) where {F}
Green function of the single-impurity anderson model in the wide-band limit.
"""
function siam_get_green_function(v, ::Val{F}; e, Δ, t, D=Inf) where {F}
    if F === :KF
        isinf(D) || error("SIAM with finite bandwith not implemented for KF")
        GR = 1 / (v - e + im * Δ)
        GA = 1 / (v - e - im * Δ)
        GK = -2im * Δ * tanh(v / t / 2) / ((v - e)^2 + Δ^2)
        SMatrix{2, 2}(0, GR, GA, GK)
    elseif F === :MF
        vv = 2π * t * (v + 1/2)
        if isinf(D)
            1 / (im * vv - e + im * sign(vv) * Δ)
        else
            1 / (im * vv - e + im * 2 * Δ / π * atan(D / vv))
        end
    elseif F === :ZF
        error("SIAM with ZF not implemented")
    else
        error("Wrong formalism $formalism")
    end
end

"""
    SIAMLazyGreen2P{F}(::Type{T}=ComplexF64; e, Δ, t, D=Inf)
Lazy Green2P object for getting SIAM bare Green function.
"""
struct SIAMLazyGreen2P{F, T} <: AbstractLazyGreen2P{F, T}
    norb::Int
    e::Float64
    Δ::Float64
    t::Float64
    D::Float64
    function SIAMLazyGreen2P{F, T}(; e, Δ, t, D=Inf) where {F, T}
        F === :KF && !isinf(D) && error("SIAM with finite bandwith not implemented for KF")
        F === :ZF && error("SIAM with ZF not implemented")
        F ∉ (:KF, :MF, :ZF) && error("Wrong formalism $formalism")
        norb = 1
        new{F, T}(norb, e, Δ, t, D)
    end
end
SIAMLazyGreen2P{F}(::Type{T}=ComplexF64; e, Δ, t, D=Inf) where {F, T} = SIAMLazyGreen2P{F, T}(; e, Δ, t, D)
function (G0::SIAMLazyGreen2P{F, T})(v) where {F, T}
    g = siam_get_green_function(v, Val(F); G0.e, G0.Δ, G0.t, G0.D)
    F === :KF ? g : SMatrix{1, 1}(g)
end
(G0::SIAMLazyGreen2P)(k, v) = G0(v)

"""
    siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t)
# Bubble for the SIAM in the wide-band limit in formalism `F` and channel `C`.
"""
function siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t, D=Inf) where {F, C}
    Base.depwarn("Use SIAMLazyGreen2P and compute_bubble", :siam_get_bubble, force=true)
    G0 = SIAMLazyGreen2P{F}(; e, Δ, t, D)
    compute_bubble(G0, basis_f, basis_b, Val(C); temperature=t)
end


function siam_get_bubble_improved(basis_f, basis_b, basis_1p, ::Val{F}, ::Val{C}; e, Δ, t, D=Inf) where {F, C}
    Base.depwarn("Use SIAMLazyGreen2P and compute_bubble_smoothed", :siam_get_bubble_improved, force=true)
    G0 = SIAMLazyGreen2P{F}(; e, Δ, t, D)
    compute_bubble_smoothed(G0, basis_f, basis_b, Val(C); temperature=t)
end
