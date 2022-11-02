using StaticArrays

"""
    get_bare_vertex(U::Number, ::Val{F}, ::Val{C})
Bare vertex of an 1-orbital model with interaction `U`, formalism `F`, and channel `C`.
"""
function get_bare_vertex(U::Number, ::Val{F}, ::Val{C}) where {F, C}
    basis = ConstantBasis()
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
    siam_get_green_function(v, e, Δ, t, ::Val{F}) where {F}
Green function of the single-impurity anderson model in the wide-band limit.
"""
function siam_get_green_function(v, e, Δ, t, ::Val{F}) where {F}
    if F === :KF
        GR = 1 / (v - e + im * Δ)
        GA = 1 / (v - e - im * Δ)
        GK = -2im * Δ * tanh(v / t / 2) / ((v - e)^2 + Δ^2)
        SMatrix{2, 2}(0, GR, GA, GK)
    elseif F === :MF
        vv = 2π * t * (v + 1/2)
        1 / (im * vv - e + im * sign(vv) * Δ)
    elseif F === :ZF
        error("SIAM with ZF not implemented")
    else
        error("Wrong formalism $formalism")
    end
end

"""
    siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t)
# Bubble for the SIAM in the wide-band limit in formalism `F` and channel `C`.
"""
function siam_get_bubble(basis_f, basis_b, ::Val{F}, ::Val{C}; e, Δ, t) where {F, C}
    Π = Bubble{F, C}(basis_f, basis_b)
    vs = get_fitting_points(basis_f)
    ws = get_fitting_points(basis_b)
    if F === :KF
        Π_data = zeros(eltype(Π.data), length(vs), 16, length(ws))
    else
        Π_data = zeros(eltype(Π.data), length(vs), 1, length(ws))
    end

    for (iw, w) in enumerate(ws)
        for (iv, v) in enumerate(vs)
            v1, v2 = mfRG._bubble_frequencies(Val(F), Val(C), v, w)
            G1 = siam_get_green_function(v1, e, Δ, t, Val(F))
            G2 = siam_get_green_function(v2, e, Δ, t, Val(F))
            if F === :KF
                for (i, ks) in enumerate(CartesianIndices((2, 2, 2, 2)))
                    k11, k12, k21, k22 = mfRG._bubble_indices(Val(C), ks)
                    Π_data[iv, i, iw] = G1[k11, k12] * G2[k21, k22]
                end
            else
                Π_data[iv, 1, iw] = G1 * G2
            end
        end
    end
    Π_data .*= mfRG._bubble_prefactor(Val(C))
    Π_data_tmp1 = mfRG.fit_basis_coeff(Π_data, basis_f, vs, 1)
    Π_data_tmp2 = mfRG.fit_basis_coeff(Π_data_tmp1, basis_b, ws, 3)
    Π.data .= reshape(Π_data_tmp2, size(Π.data))
    Π
end
