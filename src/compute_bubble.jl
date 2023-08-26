# TODO: Optimize by reducing allocation inside function call

"""
    compute_bubble(G1, G2, basis_f, basis_b, C::Symbol; temperature)
Compute the Bubble in channel `C` for the 2-point Green function `G`.
"""
function compute_bubble(G1, G2, basis_f, basis_b, C::Symbol; temperature)
    F = get_formalism(G1)
    nind = get_nind(G1)
    Π = Bubble{F}(C, basis_f, basis_b, G1.norb; temperature)
    vs = get_fitting_points(basis_f)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(eltype(Π.data), length(vs), nind^4, length(ws))

    G1_v = zeros(eltype(G1), nind, nind)
    G2_v = zeros(eltype(G2), nind, nind)
    for (iw, w) in enumerate(ws)
        for (iv, v) in enumerate(vs)
            v1, v2 = _bubble_frequencies(Val(F), C, v, w)
            get_G!(G1_v, G1, v1)
            get_G!(G2_v, G2, v2)
            for (i, inds) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                i11, i12, i21, i22 = _bubble_indices(C, inds)
                Π_data[iv, i, iw] = G1_v[i11, i12] * G2_v[i21, i22]
            end
        end
    end
    Π_data .*= _bubble_prefactor(C)
    Π_data_tmp1 = fit_basis_coeff(Π_data, basis_f, vs, 1)
    Π_data_tmp2 = fit_basis_coeff(Π_data_tmp1, basis_b, ws, 3)
    Π.data .= reshape(Π_data_tmp2, size(Π.data))
    Π
end


"""
    compute_bubble_smoothed(G1, G2, basis_f, basis_b, C::Symbol; temperature=nothing)
Improved version of `compute_bubble`.

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
function compute_bubble_smoothed(G1::AbstractLazyGreen2P{F}, G2::AbstractLazyGreen2P{F},
            basis_f, basis_b, C::Symbol; temperature=nothing) where {F}
    ntails(basis_b) > 0 && error("tails cannot be used for `basis_b` in compute_bubble_smoothed")
    nind = get_nind(G1)
    valF = Val(F)

    # Compute ∫dv b_f(v) * b_1p(v1) * b_1p(v2) where v1 and v2 are bubble frequencies for (v, w)
    ws = get_fitting_points(basis_b)
    Π_data = zeros(eltype(G1), nbasis(basis_f), nind^4, length(ws))
    # Base.Threads.@threads
    for iw in axes(ws, 1)
        w = ws[iw]
        set_basis_shift!(basis_f, w)
        overlap_f = basis_integral(basis_f, basis_f)

        bubble_value_integral = zeros(eltype(G1), size(basis_f, 2), nind^4)
        for i_f in axes(basis_f, 2)
            interval_v = integration_interval((basis_f,), (i_f,))
            isempty(interval_v) && continue

            function f(v)
                v1, v2 = _bubble_frequencies(valF, C, v, w)
                coeff_f = basis_f[v, i_f]
                G1_v = vec(G1(v1))
                G2_v = vec(G2(v2))
                (G1_v * transpose(G2_v)) .* coeff_f
            end

            l_v, r_v = endpoints(interval_v)
            if F === :MF
                res = integrate_imag(f, l_v, r_v)
            else
                l_v >= r_v && continue
                l_v ≈ r_v && continue
                res = quadgk(f, l_v, r_v)
            end

            value = reshape(res[1], nind, nind, nind, nind)
            for (ik, ks) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                k11, k12, k21, k22 = _bubble_indices(C, ks)
                bubble_value_integral[i_f, ik] += value[k11, k12, k21, k22]
            end
        end
        Π_data[:, :, iw] .= qr(overlap_f, ColumnNorm()) \ bubble_value_integral
    end

    Π = Bubble{F}(C, basis_f, basis_b; temperature)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, nind^2, nind^2, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(C)

    Π
end


function compute_bubble_smoothed(G1::Green2P{F}, G2::Green2P{F}, basis_f, basis_b,
            C::Symbol, overlap_bubble_cached=nothing; temperature=nothing) where {F}
    # Same as above, but G is a Green2P, not a AbstractLazyGreen2P.

    ntails(basis_b) > 0 && error("tails cannot be used for `basis_b` in compute_bubble_smoothed")
    nind = get_nind(G1)

    ws = get_fitting_points(basis_b)
    Π_data = zeros(eltype(G1), nbasis(basis_f), nind^4, length(ws))
    overlap_f = basis_integral(basis_f, basis_f)

    Base.Threads.@threads for iw in axes(ws, 1)
        w = ws[iw]

        # Instead of integrating G1 and G2 directly, integrate the basis functions and
        # do matrix multiplication with G1.data and G2.data.
        if overlap_bubble_cached !== nothing
            overlap_bubble = overlap_bubble_cached[iw]
        else
            overlap_bubble = basis_integral_bubble(basis_f, G1.basis, G2.basis, w, C)
        end

        bubble_value_integral = zeros(eltype(G1), nbasis(basis_f), nind^4)
        # for i_f in axes(basis_f, 2)
        #     @views for (ik, ks) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
        #         k11, k12, k21, k22 = _bubble_indices(C, ks)
        #         bubble_value_integral[i_f, ik] = dot(G1.data[k11, k12, :]',
        #                                              overlap_bubble[i_f, :, :],
        #                                              G2.data[k21, k22, :])
        #     end
        # end
        # Optimize the above using the sparsity of overlap_bubble
        for i2 in axes(G2.basis, 2), i1 in axes(G1.basis, 2), i_f in axes(basis_f, 2)
            overlap_bubble[i_f, i1, i2] == 0 && continue
            for (ik, ks) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                k11, k12, k21, k22 = _bubble_indices(C, ks)
                bubble_value_integral[i_f, ik] += (G1.data[k11, k12, i1]
                    * overlap_bubble[i_f, i1, i2] * G2.data[k21, k22, i2])
            end
        end

        Π_data[:, :, iw] .= overlap_f \ bubble_value_integral
    end

    Π = Bubble{F}(C, basis_f, basis_b; temperature)
    Π.data .= fit_basis_coeff(reshape(Π_data, :, nind^2, nind^2, length(ws)), basis_b, ws, 4)
    Π.data .*= _bubble_prefactor(C)

    Π
end
