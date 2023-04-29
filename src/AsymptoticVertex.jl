# TODO: Add test for vector_to_vertex and vector_to_vector

"""
    AsymptoticVertex
4-point vertex in the asymptotic decomposition.
- `max_class`: maximum asymptotic class (1 or 2 or 3)
"""
Base.@kwdef struct AsymptoticVertex{F, T} <: AbstractFrequencyVertex{F, T}
    max_class::Int
    # Γ0_A, Γ0_P, Γ0_T are the same bare vertex, just represented in different channels.
    # Only one of them should be counted when computing the total vertex.
    # The same holds for Λ_A, Λ_P, and Λ_T.
    Γ0_A
    Γ0_P
    Γ0_T
    K1_A = nothing
    K1_P = nothing
    K1_T = nothing
    K2_A = nothing
    K2_P = nothing
    K2_T = nothing
    K2p_A = nothing
    K2p_P = nothing
    K2p_T = nothing
    K3_A = nothing
    K3_P = nothing
    K3_T = nothing
    Λ_A = nothing  # 2-particle irreducible vertex
    Λ_P = nothing  # 2-particle irreducible vertex
    Λ_T = nothing  # 2-particle irreducible vertex
    basis_k1_b = K1_A === nothing ? nothing : get_bosonic_basis(K1_A[1])
    basis_k2_f = K2_A === nothing ? nothing : get_fermionic_basis_1(K2_A[1])
    basis_k2_b = K2_A === nothing ? nothing : get_bosonic_basis(K2_A[1])
    basis_k3_f = K3_A === nothing ? nothing : get_fermionic_basis_1(K3_A[1])
    basis_k3_b = K3_A === nothing ? nothing : get_bosonic_basis(K3_A[1])
end

get_fermionic_basis_1(Γ::AbstractVertex4P) = (; freq=Γ.basis_f1)
get_bosonic_basis(Γ::AbstractVertex4P) = (; freq=Γ.basis_b)

_vertex_names(::AsymptoticVertex) = (:K1_A, :K1_P, :K1_T, :K2_A, :K2_P, :K2_T, :K2p_A,
    :K2p_P, :K2p_T, :K3_A, :K3_P, :K3_T, :Λ_A, :Λ_P, :Λ_T)

get_vertices(Γ::AsymptoticVertex) = filter!(!isnothing, [getproperty(Γ, n) for n in _vertex_names(Γ)])

function (Γ::AsymptoticVertex)(C::Symbol, class::Symbol)
    (C, class) == (:A, :Γ0)  && return Γ.Γ0_A
    (C, class) == (:P, :Γ0)  && return Γ.Γ0_P
    (C, class) == (:T, :Γ0)  && return Γ.Γ0_T
    (C, class) == (:A, :K1)  && return Γ.K1_A
    (C, class) == (:P, :K1)  && return Γ.K1_P
    (C, class) == (:T, :K1)  && return Γ.K1_T
    (C, class) == (:A, :K2)  && return Γ.K2_A
    (C, class) == (:P, :K2)  && return Γ.K2_P
    (C, class) == (:T, :K2)  && return Γ.K2_T
    (C, class) == (:A, :K2p) && return Γ.K2p_A
    (C, class) == (:P, :K2p) && return Γ.K2p_P
    (C, class) == (:T, :K2p) && return Γ.K2p_T
    (C, class) == (:A, :K3)  && return Γ.K3_A
    (C, class) == (:P, :K3)  && return Γ.K3_P
    (C, class) == (:T, :K3)  && return Γ.K3_T
    (C, class) == (:A, :Λ)   && return Γ.Λ_A
    (C, class) == (:P, :Λ)   && return Γ.Λ_P
    (C, class) == (:T, :Λ)   && return Γ.Λ_T
    error("Wrong channel $C or class $class")
end

function get_bare_vertex(C, Γ::AsymptoticVertex)
    Base.depwarn("Use Γ(C, Γ0) instead", :get_bare_vertex, force=true)
    Γ(C, Γ0)
end

function get_reducible_vertices(C, Γ::AsymptoticVertex)
    if C === :A
        (; K1=Γ.K1_A, K2=Γ.K2_A, K2p=Γ.K2p_A, K3=Γ.K3_A)
    elseif C === :P
        (; K1=Γ.K1_P, K2=Γ.K2_P, K2p=Γ.K2p_P, K3=Γ.K3_P)
    elseif C === :T
        (; K1=Γ.K1_T, K2=Γ.K2_T, K2p=Γ.K2p_T, K3=Γ.K3_T)
    else
        error("Wrong channel $C")
    end
end

"""
    get_irreducible_vertices(C, Γ::AsymptoticVertex)
Return the list of vertices in `Γ` that are irreducible in channel `C`, except for the fully
irreducible vertex. For example, if `C == :A`, vertices reducible in the P and T channel are
returned.
"""
function get_irreducible_vertices(C, Γ::AsymptoticVertex)
    # FIXME: Currently SU2 is assumed.
    Γs = []
    for C2 in (:A, :P, :T)
        C2 === C && continue
        for class in (:K1, :K2, :K2p, :K3, :Λ)
            x = Γ(C2, class)
            x !== nothing && push!(Γs, x)
        end
    end
    su2_convert_spin_channel.(C, Γs)
end
