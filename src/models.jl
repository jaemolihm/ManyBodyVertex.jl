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
struct SIAMLazyGreen2P{F, T} <: mfRG.AbstractFrequencyVertex{F, T}
    nind::Int
    e::Float64
    Δ::Float64
    t::Float64
    D::Float64
    function SIAMLazyGreen2P{F, T}(; e, Δ, t, D=Inf) where {F, T}
        F === :KF && !isinf(D) && error("SIAM with finite bandwith not implemented for KF")
        F === :ZF && error("SIAM with ZF not implemented")
        F ∉ (:KF, :MF, :ZF) && error("Wrong formalism $formalism")
        nind = nkeldysh(F)
        new{F, ComplexF64}(nind, e, Δ, t, D)
    end
end
SIAMLazyGreen2P{F}(::Type{T}=ComplexF64; e, Δ, t, D=Inf) where {F, T} = SIAMLazyGreen2P{F, T}(; e, Δ, t, D)
(G0::SIAMLazyGreen2P{F, T})(v) where {F, T} = siam_get_green_function(v, Val(F); G0.e, G0.Δ, G0.t, G0.D)

"""
    siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t)
# Bubble for the SIAM in the wide-band limit in formalism `F` and channel `C`.
"""
function siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t, D=Inf) where {F, C}
    Π = Bubble{F, C}(basis_f, basis_b; temperature=t)
    ws = get_fitting_points(basis_b)
    if F === :KF
        Π_data = zeros(eltype(Π.data), nbasis(basis_f), 16, length(ws))
    else
        Π_data = zeros(eltype(Π.data), nbasis(basis_f), 1, length(ws))
    end

    for (iw, w) in enumerate(ws)
        set_basis_shift!(basis_f, w)
        vs = get_fitting_points(basis_f)
        if F === :KF
            Π_data_iw = zeros(eltype(Π.data), length(vs), 16)
        else
            Π_data_iw = zeros(eltype(Π.data), length(vs), 1)
        end

        for (iv, v) in enumerate(vs)
            v1, v2 = _bubble_frequencies(Val(F), Val(C), v, w)
            G1 = siam_get_green_function(v1, Val(F); e, Δ, t, D)
            G2 = siam_get_green_function(v2, Val(F); e, Δ, t, D)
            if F === :KF
                for (i, ks) in enumerate(CartesianIndices((2, 2, 2, 2)))
                    k11, k12, k21, k22 = _bubble_indices(Val(C), ks)
                    Π_data_iw[iv, i] = G1[k11, k12] * G2[k21, k22]
                end
            else
                Π_data_iw[iv, 1] = G1 * G2
            end
        end
        Π_data[:, :, iw] .= fit_basis_coeff(Π_data_iw, basis_f, vs, 1)
    end
    Π_data .*= _bubble_prefactor(Val(C))
    Π_data_tmp = fit_basis_coeff(Π_data, basis_b, ws, 3)
    Π.data .= reshape(Π_data_tmp, size(Π.data))
    Π
end


"""
    siam_get_bubble_improved(basis_f, basis_b, basis_1p, F::Val, C::Val; e, Δ, t, D=Inf)
Improved version of `siam_get_bubble`.

The information Π should store: data on v1 & v2 grid. (for simple case: separable as G1 and
G2, for k-integrated case: not separable.)

What we need in an actual calculation: `∫dv Π(v, w) f(v)` for given `w`.

We approximate this integral by first computing `cᵢ(w) = ∫dv Π(v, w) bᵢ(v)` so that the
bubble is approximated as `Π(v, w) ≈ ∑ᵢ bᵢ(v) * cᵢ(w)`. Then, the integral is approximated
as `∫dv Π(v, w) f(v) ≈ ∑ᵢ cᵢ(w) * ∫dv bᵢ(v) f(v)`.
This is equivalent to using the truncated-unity approximation `δ(v, v') ≈ ∑ᵢ bᵢ(v) bᵢ(v')`.
This approximation is accurate when `f(v)` is spanned by `bᵢ(v)`.

(FIXME: In the above we assumed `bᵢ(v')` are orthonormal. But, we actually do deal with
non-orthogonality of `bᵢ(v')`.)

The basis for `w` of `Π` should be chosen such that it is as dense as the `v`-grid for
`|w| < 2 * max(v)`, and coarse but spanning up to largest bosonic frequency of interest
(e.g. largest fitting grid point for `w` used in BSE) for larger `|w|`.
"""
function siam_get_bubble_improved(basis_f, basis_b, basis_1p, F::Val{:KF}, C::Val; e, Δ, t, D=Inf)
    ntails(basis_b) > 0 && error("tails cannot be used for bosonic frequencies of a bubble")

    # Compute the SIAM 1-particle green function with basis_1p
    vs = get_fitting_points(basis_1p)
    green_ = siam_get_green_function.(vs, F; e, Δ, t, D)
    green = reshape(reduce(hcat, green_), 2, 2, :)
    green_coeff = fit_basis_coeff(green, basis_1p, vs, 3)

    overlap_f = basis_integral(basis_f, basis_f)

    # Compute ∫dv b_f(v) * b_1p(v1) * b_1p(v2) where v1 and v2 are bubble frequencies for (v, w)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(ComplexF64, size(basis_f, 2), 16, length(ws))
    Base.Threads.@threads for iw in axes(ws, 1)
        w = ws[iw]
        coeff_g1 = zeros(eltype(basis_1p), size(basis_1p, 2))
        coeff_g2 = zeros(eltype(basis_1p), size(basis_1p, 2))
        bubble_value_integral = zeros(ComplexF64, size(basis_f, 2), 16)
        for i_f in axes(basis_f, 2)
            intervals_v = integration_intervals((basis_f,), (i_f,))
            l_v, r_v = endpoints(intervals_v)
            l_v >= r_v && continue
            l_v ≈ r_v && continue
            function f(v)
                v1, v2 = _bubble_frequencies(F, C, v, w)
                coeff_f = basis_f[v, i_f]
                @views coeff_g1 .= basis_1p[v1, :]
                @views coeff_g2 .= basis_1p[v2, :]
                g1 = SVector(zero(ComplexF64),
                        coeff_g1' * view(green_coeff, 2, 1, :),
                        coeff_g1' * view(green_coeff, 1, 2, :),
                        coeff_g1' * view(green_coeff, 2, 2, :))
                g2 = SVector(zero(ComplexF64),
                        coeff_g2' * view(green_coeff, 2, 1, :),
                        coeff_g2' * view(green_coeff, 1, 2, :),
                        coeff_g2' * view(green_coeff, 2, 2, :))
                SArray{Tuple{2,2,2,2}}(g1 * transpose(g2)) * coeff_f
            end
            res, err = quadgk(f, l_v, r_v)
            for (ik, ks) in enumerate(Iterators.product(1:2, 1:2, 1:2, 1:2))
                k11, k12, k21, k22 = _bubble_indices(C, ks)
                bubble_value_integral[i_f, ik] += res[k11, k12, k21, k22]
            end
        end
        Π_data[:, :, iw] .= overlap_f \ bubble_value_integral
    end

    Π = Bubble{_val_to_value(F), _val_to_value(C)}(basis_f, basis_b; temperature=t)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, 4, 4, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(C)

    Π
end


function siam_get_bubble_improved(basis_f, basis_b, basis_1p, F::Val{:MF}, C::Val; e, Δ, t, D=Inf)
    ntails(basis_b) > 0 && error("tails cannot be used for bosonic frequencies of a bubble")

    # Compute the SIAM 1-particle green function with basis_1p
    vs = get_fitting_points(basis_1p)
    green_ = siam_get_green_function.(vs, F; e, Δ, t, D)
    green = reshape(reduce(hcat, green_), 1, 1, :)
    green_coeff = fit_basis_coeff(green, basis_1p, vs, 3)

    overlap_f = basis_integral(basis_f, basis_f)

    # Compute ∫dv b_f(v) * b_1p(v1) * b_1p(v2) where v1 and v2 are bubble frequencies for (v, w)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(ComplexF64, size(basis_f, 2), 1, length(ws))
    Base.Threads.@threads for iw in axes(ws, 1)
        w = ws[iw]
        coeff_g1 = zeros(eltype(basis_1p), size(basis_1p, 2))
        coeff_g2 = zeros(eltype(basis_1p), size(basis_1p, 2))
        bubble_value_integral = zeros(ComplexF64, size(basis_f, 2), 1)
        for i_f in axes(basis_f, 2)
            function f(v)
                v1, v2 = mfRG._bubble_frequencies(F, C, v, w)
                coeff_f = basis_f[v, i_f]
                @views coeff_g1 .= basis_1p[v1, :]
                @views coeff_g2 .= basis_1p[v2, :]
                g1 = coeff_g1' * view(green_coeff, 1, 1, :)
                g2 = coeff_g2' * view(green_coeff, 1, 1, :)
                g1 * g2 * coeff_f
            end
            intervals_v = integration_intervals((basis_f,), (i_f,))
            for (l_v, r_v) in intervals_v
                res = integrate_imag(f, l_v, r_v)
                res !== nothing && (bubble_value_integral[i_f, 1] += res.val)
            end
        end
        Π_data[:, :, iw] .= overlap_f \ bubble_value_integral
    end

    Π = Bubble{_val_to_value(F), _val_to_value(C)}(basis_f, basis_b; temperature=t)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, 1, 1, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(C)

    Π
end
