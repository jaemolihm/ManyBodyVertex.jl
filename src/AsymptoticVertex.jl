# FIXME: Better treatment of Γ0_A, Γ0_P (why no Γ0_T...?)

"""
    AsymptoticVertex
4-point vertex in the asymptotic decomposition.
- `max_class`: maximum asymptotic class (1 or 2 or 3)
"""
Base.@kwdef struct AsymptoticVertex{F, T} <: AbstractFrequencyVertex{F, T}
    max_class::Int
    # Γ0_A, Γ0_P, Γ0_T are the same bare vertex, just represented in different channels.
    # Only one of them should be counted when computing the total vertex.
    Γ0_A
    Γ0_P
    Γ0_T
    K1_A
    K1_P
    K1_T
    K2_A = nothing
    K2_P = nothing
    K2_T = nothing
    K2p_A = nothing
    K2p_P = nothing
    K2p_T = nothing
    K3_A = nothing
    K3_P = nothing
    K3_T = nothing
    basis_k1_b = K1_A[1].basis_b
    basis_k2_f = K2_A === nothing ? nothing : K2_A[1].basis_f1
    basis_k2_b = K2_A === nothing ? nothing : K2_A[1].basis_b
    basis_k3_f = K3_A === nothing ? nothing : K3_A[1].basis_f1
    basis_k3_b = K3_A === nothing ? nothing : K3_A[1].basis_b
end

_vertex_names(::AsymptoticVertex) = (:K1_A, :K1_P, :K1_T, :K2_A, :K2_P, :K2_T, :K2p_A,
    :K2p_P, :K2p_T, :K3_A, :K3_P, :K3_T)

get_vertices(Γ::AsymptoticVertex) = filter!(!isnothing, [getproperty(Γ, n) for n in _vertex_names(Γ)])

"""
    get_irreducible_vertices(C, Γ::AsymptoticVertex)
Return the list of vertices in `Γ` that are irreducible in channel `C`, except for the fully
irreducible vertex. For example, if `C == :A`, vertices reducible in the P and T channel are
returned.
"""
function get_irreducible_vertices(C, Γ::AsymptoticVertex)
    # FIXME: Currently SU2 is assumed.
    su2_convert_spin_channel.(C, filter!(x -> channel(x[1]) != C, get_vertices(Γ)))
end

"""
    get_difference_norm(Γ1::AsymptoticVertex, Γ2::AsymptoticVertex) => (; abserr, relerr)
Absolute and relative difference between `Γ1` and `Γ2`.
"""
function get_difference_norm(Γ1::AsymptoticVertex, Γ2::AsymptoticVertex)
    abserr = zero(real(eltype(Γ1)))
    relerr = zero(real(eltype(Γ1)))
    for n in mfRG._vertex_names(Γ1)
        x1 = getproperty(Γ1, n)
        x2 = getproperty(Γ2, n)
        for i in 1:2
            if x1 !== nothing && x2 !== nothing
                err = norm(x1[i].data .- x2[i].data)
                val = (norm(x1[i].data) + norm(x2[i].data)) / 2
            elseif x1 === nothing && x2 !== nothing
                err = norm(x2[i].data)
                val = norm(x2[i].data)
            elseif x1 !== nothing && x2 === nothing
                err = norm(x1[i].data)
                val = norm(x1[i].data)
            elseif x1 === nothing && x2 === nothing
                continue
            end
            abserr += err
            if val > 1e-10
                # skip relative error if value is too small
                relerr = max(relerr, err / val)
            end
        end
    end
    (; abserr, relerr)
end
