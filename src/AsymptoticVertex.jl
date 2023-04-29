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
    su2_convert_spin_channel.(C, filter!(x -> channel(x[1]) != C, get_vertices(Γ)))
end

vertex_norm(x) = (; err=norm.(getproperty.(x, :data)), val=norm.(getproperty.(x, :data)))

vertex_diff_norm(x1, x2) = (; err=norm.(getproperty.(x1, :data) .- getproperty.(x2, :data)), val=(norm(getproperty.(x1, :data)) .+ norm.(getproperty.(x2, :data))) ./ 2)
vertex_diff_norm(x1, x2::Nothing) = vertex_norm(x1)
vertex_diff_norm(x1::Nothing, x2) = vertex_norm(x2)
vertex_diff_norm(x1::Nothing, x2::Nothing) = (; err=nothing, val=nothing)

"""
    get_difference_norm(Γ1::AsymptoticVertex, Γ2::AsymptoticVertex) => (; abserr, relerr)
Absolute and relative difference between `Γ1` and `Γ2`.
"""
function get_difference_norm(Γ1::AsymptoticVertex, Γ2::AsymptoticVertex)
    abserr = zero(real(eltype(Γ1)))
    relerr = zero(real(eltype(Γ1)))
    for n in _vertex_names(Γ1)
        x1 = getproperty(Γ1, n)
        x2 = getproperty(Γ2, n)
        err, val = vertex_diff_norm(x1, x2)
        err === nothing && continue
        for i in 1:2
            abserr += err[i]
            if val[i] > 1e-10  # skip relative error if value is too small
                relerr = max(relerr, err[i] / val[i])
            end
        end
    end
    (; abserr, relerr)
end

"""
    vertex_to_vector(Γ::AsymptoticVertex)
Reshape and concatenate the vertices of `Γ` to a one-dimensional vector.
"""
function vertex_to_vector(Γ::AsymptoticVertex)
    v = eltype(Γ)[]
    for Γ_field in get_vertices(Γ)
        for x in Γ_field
            vertex_to_vector!(v, x)
        end
    end
    v
end

"""
    vector_to_vertex(v, Γ_template::AsymptoticVertex)
Reshape a one-dimensional vector `v` to the vertices with the same type and size as
`Γ_template`.
"""
function vector_to_vertex(v, Γ_template::AsymptoticVertex)
    offset = 0
    vertices = Dict()
    for name in _vertex_names(Γ_template)
        getproperty(Γ_template, name) === nothing && continue
        vertices[name] = similar.(getproperty(Γ_template, name))
        for x in vertices[name]
            offset = vector_to_vertex!(x, v, offset)
        end
    end
    typeof(Γ_template)(; Γ_template.max_class, Γ_template.Γ0_A, Γ_template.Γ0_P,
                         Γ_template.Γ0_T, vertices...)
end

"""
    vertex_to_vector!(v, Γ::AbstractVertex4P)
Flatten the data of the vertex `Γ` and append it to `v`.
"""
vertex_to_vector!(v, Γ::AbstractVertex4P) = append!(v, vec(Γ.data))

"""
    vector_to_vertex(Γ::AbstractVertex4P, v, offset=0)
Use `v[offset+1 : offset+n]` to set the vertex `Γ`. Return the new offset.
"""
function vector_to_vertex!(Γ::AbstractVertex4P, v, offset=0)
    n = length(Γ.data)
    vec(Γ.data) .= v[offset+1:offset+n]
    return offset + n
end
