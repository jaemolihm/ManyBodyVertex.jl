# Hubbard model
# ``H = -∑_{(i,j)∈n.n.} t cᵢ†cⱼ - ∑_{(i,j)∈next n.n.} t2 cᵢ†cⱼ + U ∑ᵢ nᵢ↑ nᵢ↓ - μ ∑ᵢ nᵢ``
# ``ϵk = -2t(cos kx + cos ky) - 4t' cos kx * cos ky``
# ``G0(k, iv) = (iv + μ - ϵk)⁻¹``
# `μ = 0` corresponds to the half filling.

"""
    hubbard_get_green_function(k, v, ::Val{F}; T, μ, t, t2=0) where {F}
Green function of the single-impurity anderson model in the wide-band limit.
# Inputs
-`k`: Fermionic momentum in reduced coordinates (with Brillouin zone [0, 1]).
-`v`: Fermionic frequency. (For the Matsubara formalism, use integer values so that the
    actual frequency is `2π * T * (v + 1/2)`.)
-`F`: Formalism. `Val(:MF)` or `Val(:KF)` or `Val(:ZF)`.
-`T`: temperature
-`μ`: chemical potential
-`t`: nearest-neighbor hopping
-`t2`: next-nearest-neighbor hopping
"""
function hubbard_get_green_function(k, v, ::Val{F}; T, μ, t, t2=0) where {F}
    length(k) != 2 && error("Hubbard model implemented only for the 2D case")
    ϵk = -2 * t * (cospi(2k[1]) + cospi(2k[2])) - 4 * t2 * cospi(2k[1]) * cospi(2k[2])
    if F === :KF
        error("Hubbard model not yet implemented for KF")
    elseif F === :MF
        vv = 2π * T * (v + 1/2)
        1 / (im * vv + μ - ϵk)
    elseif F === :ZF
        error("Hubbard model with ZF not implemented")
    else
        error("Wrong formalism $formalism")
    end
end

"""
    HubbardLazyGreen2P{F}(::Type{T}=ComplexF64; e, Δ, t, D=Inf)
Lazy Green2P object for getting SIAM bare Green function.
"""
struct HubbardLazyGreen2P{F, T} <: AbstractLazyGreen2P{F, T}
    norb::Int
    temperature::Float64
    t::Float64
    μ::Float64
    t2::Float64
    function HubbardLazyGreen2P{F, T}(; temperature, t, μ, t2=0) where {F, T}
        F === :KF && error("Hubbard model with KF not yet implemented")
        F === :ZF && error("Hubbard model with ZF not implemented")
        F ∉ (:KF, :MF, :ZF) && error("Wrong formalism $formalism")
        norb = 1
        new{F, T}(norb, temperature, t, μ, t2)
    end
end
HubbardLazyGreen2P{F}(::Type{T}=ComplexF64; temperature, t, μ, t2=0) where {F, T} = HubbardLazyGreen2P{F, T}(; temperature, t, μ, t2)
function (G0::HubbardLazyGreen2P{F, T})(k, v) where {F, T}
    g = hubbard_get_green_function(k, v, Val(F); T=G0.temperature, G0.μ, G0.t, G0.t2)
    F === :KF ? g : SMatrix{1, 1}(g)
end