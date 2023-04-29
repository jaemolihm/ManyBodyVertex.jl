export HubbardAtomLazyGreen2P

"""
    hubbard_atom_get_green_function(v, ::Val{F}; U, temperature) where {F}
Green function of the single-impurity anderson model in the wide-band limit.
"""
function hubbard_atom_get_green_function(v, ::Val{F}; U, temperature) where {F}
    if F === :KF
        error("Hubbard atom with KF not implemented")
    elseif F === :MF
        vv = 2π * temperature * (v + 1/2)
        1 / (im * vv - U^2 / (4 * im * vv))
    elseif F === :ZF
        error("Hubbard atom with ZF not implemented")
    else
        error("Wrong formalism $formalism")
    end
end

"""
    HubbardAtomLazyGreen2P{F}(::Type{T}=ComplexF64; U, temperature)
Lazy Green2P object for the bare Green function of the Hubbard atom.

TODO: non-half-filling
"""
struct HubbardAtomLazyGreen2P{F, T} <: AbstractLazyGreen2P{F, T}
    norb::Int
    U::Float64
    temperature::Float64
    function HubbardAtomLazyGreen2P{F, T}(; U, temperature) where {F, T}
        F === :KF && error("Hubbard atom not yet implemented for KF")
        F === :ZF && error("Hubbard atom not yet implemented for ZF")
        F ∉ (:KF, :MF, :ZF) && error("Wrong formalism $formalism")
        norb = 1
        new{F, T}(norb, U, temperature)
    end
end
function HubbardAtomLazyGreen2P{F}(::Type{T}=ComplexF64; U, temperature) where {F, T}
    HubbardAtomLazyGreen2P{F, T}(; U, temperature)
end

function (G0::HubbardAtomLazyGreen2P{F, T})(v) where {F, T}
    g = hubbard_atom_get_green_function(v, Val(F); G0.U, G0.temperature)
    F === :KF ? g : SMatrix{1, 1}(g)
end
(G0::HubbardAtomLazyGreen2P)(k, v) = G0(v)

function Base.show(io::IO, A::HubbardAtomLazyGreen2P{F, T}) where {F, T}
    print(io, Base.typename(typeof(A)).wrapper, "{:$F, :$T}")
    print(io, "(U=$(A.U), temperature=$(A.temperature))")
end


"""
    hubbard_atom_analytic_vertex(v, vp, w, U, temperature, ::Val{C}=Val(:A)) where {C}
Compute the 4p vertex of the Hubbard atom at half filling in MF.

`Γ.Λ` contains the core part of the asymptotic decomposition, instead of the fully
irreducible vertex.

Use Eq. (85) of Kugler, Lee, von Delft, PRX 11, 041006 (2021)
"""
function hubbard_atom_analytic_vertex(v, vp, w, U, temperature, ::Val{C}=Val(:A)) where {C}
    v1234 = im * 2π .* temperature .* (mfRG.frequency_to_standard(Val(:MF), Val(C), v, vp, w) .+ 1/2)
    u = U / 2
    β = 1 / temperature
    g_uudd = 2u + u^3 * sum(v1234.^2) / prod(v1234) - 6 * u^5 / prod(v1234)
    g_uuuu = 0.0im
    if abs(v1234[1] + v1234[2]) < temperature * 1e-2
        g_uudd += β * u^2 * tanh(β * u / 2) * prod(v1234 .+ u) / prod(v1234)
        g_uuuu -= β * u^2 * prod(v1234 .+ u) / prod(v1234)
    end
    if abs(v1234[1] + v1234[3]) < temperature * 1e-2
        g_uudd += β * u^2 * (tanh(β * u / 2) - 1) * prod(v1234 .+ u) / prod(v1234)
    end
    if abs(v1234[1] + v1234[4]) < temperature * 1e-2
        g_uudd += β * u^2 * (tanh(β * u / 2) + 1) * prod(v1234 .+ u) / prod(v1234)
        g_uuuu += β * u^2 * prod(v1234 .+ u) / prod(v1234)
    end
    # (uuuu, uudd) -> (d, m)
    g_dm = (g_uuuu + g_uudd, g_uuuu - g_uudd)  # (d, m)
    mfRG.su2_convert_spin_channel(C, g_dm, :A)
end


function hubbard_atom_asymptotic_vertex(U, temperature, basis_k1_b, basis_k2_b, basis_k2_f)
    # Large frequency values used to evalute the asymptotic limit
    v_large1 = 2000
    v_large2 = 3000

    Γ0A = mfRG.su2_bare_vertex(Val(:MF), Val(:A), U)
    Γ0P = mfRG.su2_bare_vertex(Val(:MF), Val(:P), U)
    Γ0T = mfRG.su2_bare_vertex(Val(:MF), Val(:T), U)
    Γ0 = (; A = Γ0A, P = Γ0P, T = Γ0T)


    Γs = Dict()

    for C in (:A, :P, :T)
        # Compute K1 vertex
        ws = get_fitting_points(basis_k1_b)
        K1_data = [zeros(ComplexF64, length(ws)) for _ in 1:2]
        for (iw, w) in enumerate(ws)
            val = hubbard_atom_analytic_vertex(v_large1, v_large2, w, U, temperature, Val(C))
            K1_data[1][iw] = val[1] - getproperty(Γ0, C)[1](0, 0, 0)[1, 1]
            K1_data[2][iw] = val[2] - getproperty(Γ0, C)[2](0, 0, 0)[1, 1]
        end
        K1 = Tuple(Vertex4P{:MF, C}(ImagConstantBasis(), ImagConstantBasis(), basis_k1_b, 1) for _ in 1:2)
        for ispin in 1:2
            K1[ispin].data .= reshape(mfRG.fit_basis_coeff(K1_data[ispin], basis_k1_b, ws, 1), size(K1[ispin].data))
        end
        Γs[(C, :K1)] = K1

        # Compute K2 vertex
        vs = get_fitting_points(basis_k2_f)
        ws = get_fitting_points(basis_k2_b)
        K2_data = [zeros(ComplexF64, length(vs), length(ws)) for _ in 1:2]
        for (iw, w) in enumerate(ws), (iv, v) in enumerate(vs)
            val = hubbard_atom_analytic_vertex(v, v_large2, w, U, temperature, Val(C))
            val_K1 = hubbard_atom_analytic_vertex(v_large1, v_large2, w, U, temperature, Val(C))
            K2_data[1][iv, iw] = val[1] - val_K1[1]
            K2_data[2][iv, iw] = val[2] - val_K1[2]
        end
        K2 = Tuple(Vertex4P{:MF, C}(basis_k2_f, ImagConstantBasis(), basis_k2_b, 1) for _ in 1:2)
        for ispin in 1:2
            tmp1 = mfRG.fit_basis_coeff(K2_data[ispin], basis_k2_f, vs, 1)
            tmp2 = mfRG.fit_basis_coeff(tmp1, basis_k2_b, ws, 2)
            K2[ispin].data .= reshape(tmp2, size(K2[ispin].data))
        end
        Γs[(C, :K2)] = K2

        # Compute K2p vertex
        vs = get_fitting_points(basis_k2_f)
        ws = get_fitting_points(basis_k2_b)
        K2p_data = [zeros(ComplexF64, length(vs), length(ws)) for _ in 1:2]
        for (iw, w) in enumerate(ws), (iv, v) in enumerate(vs)
            val = hubbard_atom_analytic_vertex(v_large1, v, w, U, temperature, Val(C))
            val_K1 = hubbard_atom_analytic_vertex(v_large1, v_large2, w, U, temperature, Val(C))
            K2p_data[1][iv, iw] = val[1] - val_K1[1]
            K2p_data[2][iv, iw] = val[2] - val_K1[2]
        end
        K2p = Tuple(Vertex4P{:MF, C}(ImagConstantBasis(), basis_k2_f, basis_k2_b, 1) for _ in 1:2)
        for ispin in 1:2
            tmp1 = mfRG.fit_basis_coeff(K2p_data[ispin], basis_k2_f, vs, 1)
            tmp2 = mfRG.fit_basis_coeff(tmp1, basis_k2_b, ws, 2)
            K2p[ispin].data .= reshape(tmp2, size(K2p[ispin].data))
        end
        Γs[(C, :K2p)] = K2p

        # Compute rest function
        vs = get_fitting_points(basis_k2_f)
        ws = get_fitting_points(basis_k2_b)
        R_data = [zeros(ComplexF64, length(vs), length(vs), length(ws)) for _ in 1:2]
        for (iw, w) in enumerate(ws), (iv, v) in enumerate(vs), (ivp, vp) in enumerate(vs)
            val = hubbard_atom_analytic_vertex(v, vp, w, U, temperature, Val(C))
            for C2 in (:A, :P, :T)
                v1234 = mfRG.frequency_to_standard(Val(:MF), Val(C), v, vp, w)
                v_C2, vp_C2, w_C2 = mfRG.frequency_to_channel(Val(:MF), Val(C2), v1234)
                val_K2 = hubbard_atom_analytic_vertex(v_C2, v_large2, w_C2, U, temperature, Val(C2))
                val_K2p = hubbard_atom_analytic_vertex(v_large1, vp_C2, w_C2, U, temperature, Val(C2))
                val_K1 = hubbard_atom_analytic_vertex(v_large1, v_large2, w_C2, U, temperature, Val(C2))
                val_U = Tuple(x(0, 0, 0)[1, 1] for x in getproperty(Γ0, C2))
                val_C2 = @. (val_K2 - val_K1) + (val_K2p - val_K1) + (val_K1 - val_U)
                val = val .- mfRG.su2_convert_spin_channel(C, val_C2, C2)
            end
            R_data[1][iv, ivp, iw] = val[1] - getproperty(Γ0, C)[1](0, 0, 0)[1, 1]
            R_data[2][iv, ivp, iw] = val[2] - getproperty(Γ0, C)[2](0, 0, 0)[1, 1]
        end
        R = Tuple(Vertex4P{:MF, C}(basis_k2_f, basis_k2_f, basis_k2_b, 1) for _ in 1:2)
        for ispin in 1:2
            tmp1 = mfRG.fit_basis_coeff(R_data[ispin], basis_k2_f, vs, 1)
            tmp2 = mfRG.fit_basis_coeff(tmp1, basis_k2_f, vs, 2)
            R[ispin].data .= mfRG.fit_basis_coeff(tmp2, basis_k2_b, ws, 3)
        end
        Γs[(C, :R)] = R
    end

    mfRG.AsymptoticVertex{:MF, ComplexF64}(; max_class=3,
        Γ0_A = Γ0.A, Γ0_P = Γ0.P, Γ0_T = Γ0.T,
        K1_A = Γs[(:A, :K1)], K1_P = Γs[(:P, :K1)], K1_T = Γs[(:T, :K1)],
        K2_A = Γs[(:A, :K2)], K2_P = Γs[(:P, :K2)], K2_T = Γs[(:T, :K2)],
        K2p_A = Γs[(:A, :K2p)], K2p_P = Γs[(:P, :K2p)], K2p_T = Γs[(:T, :K2p)],
        # K3_A = Γs[(:A, :K3)], K3_P = Γs[(:P, :K3)], K3_T = Γs[(:T, :K3)],
        # Λ_A = Γs[(:A, :Λ)], Λ_P = Γs[(:P, :Λ)], Λ_T = Γs[(:T, :Λ)],
        Λ_A = Γs[(:A, :R)], Λ_P = Γs[(:P, :R)], Λ_T = Γs[(:T, :R)],
    )
end
