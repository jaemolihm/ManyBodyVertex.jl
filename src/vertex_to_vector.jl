
vertex_norm(x) = (; err=norm.(getproperty.(x, :data)), val=norm.(getproperty.(x, :data)))

vertex_diff_norm(x1, x2) = (; err=norm.(getproperty.(x1, :data) .- getproperty.(x2, :data)), val=(norm(getproperty.(x1, :data)) .+ norm.(getproperty.(x2, :data))) ./ 2)
vertex_diff_norm(x1, x2::Nothing) = vertex_norm(x1)
vertex_diff_norm(x1::Nothing, x2) = vertex_norm(x2)
vertex_diff_norm(x1::Nothing, x2::Nothing) = (; err=nothing, val=nothing)

"""
    get_difference_norm(Γ1::AsymptoticVertex, Γ2::AsymptoticVertex) => (; abserr, relerr)
Absolute and relative difference between `Γ1` and `Γ2`.
"""
function get_difference_norm(Γ1::T, Γ2::T) where {T <: Union{AsymptoticVertex, SBEVertexX5X, SBEReducibleVertex}}
    abserr = zero(real(eltype(Γ1)))
    relerr = zero(real(eltype(Γ1)))
    for n in _vertex_names(Γ1)
        x1 = getproperty(Γ1, n)
        x2 = getproperty(Γ2, n)
        # FIXME
        if x1 isa NTuple{2, SBEReducibleVertex}
            x1 = getproperty.(x1, :W)
            x2 = getproperty.(x2, :W)
        end
        err, val = vertex_diff_norm(x1, x2)
        err === nothing && continue
        for i in 1:2
            abserr += err[i]
            if val[i] > 1e-10  # skip relative error if value is too small
                relerr = max(relerr, err[i] / val[i])
            end
        end
        @info n, maximum(filter!(!isnan, collect(err ./ val)))
    end
    (; abserr, relerr)
end

"""
    vertex_to_vector(Γ)
Reshape and concatenate the vertices of `Γ` to a one-dimensional vector.
"""
function vertex_to_vector(Γ)
    v = eltype(Γ)[]
    for Γ_field in get_vertices(Γ)
        for x in Γ_field
            vertex_to_vector!(v, x)
        end
    end
    v
end

function vertex_to_vector(Γ::NTuple{N, <: Vertex4P}) where {N}
    v = eltype(first(Γ))[]
    for x in Γ
        vertex_to_vector!(v, x)
    end
    v
end

"""
    vector_to_vertex(v, Γ_template)
Reshape a one-dimensional vector `v` to the vertices with the same type and size as
`Γ_template`.
"""
function vector_to_vertex(v, Γ_template)
    offset = 0
    vertices = Dict()
    for name in _vertex_names(Γ_template)
        getproperty(Γ_template, name) === nothing && continue
        vertices[name] = similar.(getproperty(Γ_template, name))
        for x in vertices[name]
            offset = vector_to_vertex!(x, v, offset)
        end
    end
    if Γ_template isa AsymptoticVertex
        typeof(Γ_template)(; Γ_template.max_class, Γ_template.Γ0_A, Γ_template.Γ0_P,
                            Γ_template.Γ0_T, vertices...)
    elseif Γ_template isa SBEVertexX5X
        typeof(Γ_template)(; vertices...)
    end
end


function vector_to_vertex(v, Γ_template::NTuple{N, <: Vertex4P}) where {N}
    Γ = similar.(Γ_template)
    offset = 0
    for x in Γ
        offset = vector_to_vertex!(x, v, offset)
    end
    Γ
end

"""
    vertex_to_vector!(v, Γ::AbstractVertex4P)
Flatten the data of the vertex `Γ` and append it to `v`.
"""
vertex_to_vector!(v, Γ::AbstractVertex4P) = append!(v, vec(Γ.data))

vertex_to_vector!(v, ∇::SBEReducibleVertex) = append!(v, vec(∇.U.data), vec(∇.W.data), vec(∇.Λb.data), vec(∇.Λb.data))

"""
    vector_to_vertex(Γ::AbstractVertex4P, v, offset=0)
Use `v[offset+1 : offset+n]` to set the vertex `Γ`. Return the new offset.
"""
function vector_to_vertex!(Γ::AbstractVertex4P, v, offset=0)
    n = length(Γ.data)
    vec(Γ.data) .= v[offset+1:offset+n]
    return offset + n
end

function vector_to_vertex!(Γ::SBEReducibleVertex, v, offset=0)
    offset = vector_to_vertex!(Γ.U, v, offset)
    offset = vector_to_vertex!(Γ.W, v, offset)
    offset = vector_to_vertex!(Γ.Λb, v, offset)
    offset = vector_to_vertex!(Γ.Λ, v, offset)
    return offset
end
