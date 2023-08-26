using Dictionaries

"""
    RealSpaceVertex{RC}(rbasis, vertices_R)

`RC` is the channel used for real-space (and momentum) parametrization. Note that it may not
be equal to the channel used for the frequency parametrization of the vertices. The former
can be accessed via `real_space_channel`, and the latter via `channel`.

# Fields
- `rbasis`: Real-space basis for the vertex.
- `vertices_R`: Dictionary of vertices in the real space basis with keys `[ibL, ibR, iR_B]`.
"""
struct RealSpaceVertex{F, RC, T, VT <: AbstractFrequencyVertex{F, T}, RBT} <: AbstractFrequencyVertex{F, T}
    rbasis::RBT
    vertices_R::Dictionary{NTuple{3, Int}, VT}
    function RealSpaceVertex{RC}(rbasis::RBT, vertices_R::Dictionary) where {RC, RBT}
        VT = eltype(vertices_R)
        F = get_formalism(VT)
        T = eltype(VT)
        new{F, RC, T, VT, RBT}(rbasis, vertices_R)
    end
end
channel(::RealSpaceVertex{F, RC, T, VT}) where {F, RC, T, VT} = channel(VT)
real_space_channel(::RealSpaceVertex{F, RC}) where {F, RC} = RC

get_fermionic_basis_1(Γ::RealSpaceVertex) = (; freq=Γ.basis_f1, r=Γ.rbasis)
get_bosonic_basis(Γ::RealSpaceVertex) = (; freq=Γ.basis_b, r=Γ.rbasis)

function RealSpaceVertex{RC}(rbasis, ::Type{VT}) where {RC, VT}
    RealSpaceVertex{RC}(rbasis, Dictionary{NTuple{3, Int}, VT}())
end

function Base.show(io::IO, A::RealSpaceVertex{F, RC}) where {F, RC}
    print(io, Base.typename(typeof(A)).wrapper, "{:$F, :$RC}, ")
    println(io, "contains $(length(A.vertices_R)) vertices.")
    println(io, "  rbasis: $(A.rbasis)")
    length(A.vertices_R) > 0 && println(io, "  Vertex: $(first(A.vertices_R))")
end

function Base.getindex(A::RealSpaceVertex, bL::Bond, bR::Bond, R_B)
    # Find ibL, ibR, iR_B from A.rbasis, return nothing if not stored
    ibL = findfirst(x -> x == bL, A.rbasis.bonds_L)
    ibL === nothing && return nothing
    ibR = findfirst(x -> x == bR, A.rbasis.bonds_R)
    ibR === nothing && return nothing
    iR_B = findfirst(x -> x == R_B, A.rbasis.R_B_replicas[ibL, ibR])
    iR_B === nothing && return nothing
    haskey(A.vertices_R, (ibL, ibR, iR_B)) ? A.vertices_R[(ibL, ibR, iR_B)] : nothing
end

function Base.getproperty(A::RealSpaceVertex, s::Symbol)
    if s === :basis_f1 || s === :basis_f2 || s === :basis_b || s === :norb
        getfield(first(getfield(A, :vertices_R)), s)
    else
        getfield(A, s)
    end
end

function Base.similar(A::RealSpaceVertex{F, RC, T}) where {F, RC, T}
    RealSpaceVertex{RC}(A.rbasis, zero.(A.vertices_R))
end
Base.zero(A::RealSpaceVertex) = similar(A)

function Base.:+(A::T, B::T) where {T <: RealSpaceVertex{F, RC}} where {F, RC}
    # A.rbasis == B.rbasis || error("Cannot add vertices with different rbasis")
    RealSpaceVertex{RC}(A.rbasis, A.vertices_R .+ B.vertices_R)
end

function Base.:-(A::T, B::T) where {T <: RealSpaceVertex{F, RC}} where {F, RC}
    # A.rbasis == B.rbasis || error("Cannot add vertices with different rbasis")
    RealSpaceVertex{RC}(A.rbasis, A.vertices_R .- B.vertices_R)
end

function Base.:*(x::Number, A::RealSpaceVertex)
    RealSpaceVertex{real_space_channel(A)}(A.rbasis, .*(A.vertices_R, x))
end

function Base.:/(A::RealSpaceVertex, x::Number)
    RealSpaceVertex{real_space_channel(A)}(A.rbasis, ./(A.vertices_R, x))
end

"""
- `i1234`: atom indices in the standard representation
- `R1234`: lattice vectors in the standard representation
"""
function to_real_space(A::RealSpaceVertex, i1234, R1234)
    C = real_space_channel(A)
    i1, i2, i3, i4 = indices_to_channel(Val(C), i1234)
    R, Rp, R_B = lattice_vectors_to_channel(Val(C), R1234)
    bL = (i1, i2, R)
    bR = (i4, i3, Rp)
    A[bL, bR, R_B]
end

function real_space_convert_channel(A::RealSpaceVertex, rbasis_out, ::Val{RC_out}) where {RC_out}
    A_out = RealSpaceVertex{RC_out}(rbasis_out, eltype(A.vertices_R))

    for (ibR, bR) in enumerate(rbasis_out.bonds_R), (ibL, bL) in enumerate(rbasis_out.bonds_L)
        i1, i2, R = bL
        i4, i3, Rp = bR
        i1234 = indices_to_standard(Val(RC_out), (i1, i2, i3, i4))

        R_B_replicas = rbasis_out.R_B_replicas[ibL, ibR]
        for (iR_B, R_B) in enumerate(R_B_replicas)
            R1234 = lattice_vectors_to_standard(Val(RC_out), R, Rp, R_B)
            A_R = to_real_space(A, i1234, R1234)
            A_R !== nothing && insert!(A_out.vertices_R, (ibL, ibR, iR_B), A_R)
        end
    end
    A_out
end

function vertex_to_vector!(v, Γ::RealSpaceVertex)
    for key in keys(Γ.vertices_R)
        vertex_to_vector!(v, Γ.vertices_R[key])
    end
end

function vector_to_vertex!(Γ::RealSpaceVertex, v, offset=0)
    for key in keys(Γ.vertices_R)
        offset = vector_to_vertex!(Γ.vertices_R[key], v, offset)
    end
    offset
end

function vertex_norm(x::RealSpaceVertex)
    res = vertex_norm.(x.vertices_R)
    err = norm(getproperty.(res, :err))
    val = norm(getproperty.(res, :val))
    (; err, val)
end

function vertex_diff_norm(x1::NTuple{2, <:RealSpaceVertex}, x2::NTuple{2, <:RealSpaceVertex})
    norms1 = vertex_diff_norm(x1[1].vertices_R, x2[1].vertices_R)
    norms2 = vertex_diff_norm(x1[2].vertices_R, x2[2].vertices_R)
    (; err=norm.((norms1.err, norms2.err)), val=norm.((norms1.val, norms2.val)))
end

"""
    interpolate_to_q(A::RealSpaceVertex, xq, bL::Bond, bR::Bond)
Fourier interpolate the real-space vertex `A` to an arbitrary q-vector `xq` at bonds `bL`
and `bR`.
"""
@timeit timer "itp_Γ" function interpolate_to_q(A::RealSpaceVertex, xq, bL::Bond, bR::Bond)
    ibL = findfirst(x -> x == bL, A.rbasis.bonds_L)
    ibL === nothing && return nothing
    ibR = findfirst(x -> x == bR, A.rbasis.bonds_R)
    ibR === nothing && return nothing
    R, Rp = bL[3], bR[3]

    A_q = zero(first(A.vertices_R))
    for (iR_B, R_B) in enumerate(A.rbasis.R_B_replicas[ibL, ibR])
        if haskey(A.vertices_R, (ibL, ibR, iR_B))
            # Fouirer transform R_B -> xq
            coeff = cispi(2 * xq' * (R_B + (R - Rp) / 2))
            A_q.data .+= A.vertices_R[(ibL, ibR, iR_B)].data .* coeff
        end
    end
    A_q
end

function apply_crossing(Γ::RealSpaceVertex{F, RC}) where {F, RC}
    RC_out = channel_apply_crossing(RC)
    RealSpaceVertex{RC_out}(Γ.rbasis, apply_crossing.(Γ.vertices_R))
end

@timeit timer "cache_Γ" function cache_vertex_matrix(Γs::AbstractVector{<:RealSpaceVertex},
        C, ws, basis_aux1=nothing, basis_aux2=basis_aux1)

    isempty(Γs) && return nothing
    if any(channel.(Γs) .!= C) && basis_aux1 === nothing
        error("For vertex with different channel than C=$C, basis_aux must be provided")
    end
    rbasis = channel(first(Γs)) === C ? first(Γs).rbasis : basis_aux1.r

    # 1. Convert the real-space channel of Γs to C
    Γs_in_real_space_C = real_space_convert_channel.(Γs, Ref(rbasis), Val(C))

    # 2. For each real-space index, apply cache_vertex_matrix
    vertices_R_pairs = map(get_indices(rbasis)) do (ibL, ibR, iR_B)
        bL = rbasis.bonds_L[ibL]
        bR = rbasis.bonds_R[ibR]
        R_B = rbasis.R_B_replicas[ibL, ibR][iR_B]

        Γs_R = filter!(!isnothing, [Γ[bL, bR, R_B] for Γ in Γs_in_real_space_C])
        Γ_R_cached = cache_vertex_matrix(Γs_R, C, ws, basis_aux1.freq, basis_aux2.freq)
        (ibL, ibR, iR_B) => Γ_R_cached
    end
    vertices_R = dictionary(filter!(x -> !isnothing(x[2]), vertices_R_pairs))

    RealSpaceVertex{C}(rbasis, vertices_R)
end


function get_bare_vertex(::Val{F}, ::Val{C}, U::Number, rbasis::RealSpaceBasis{Dim}) where {F, C, Dim}
    R0 = zero(SVector{Dim, Int})
    ibL = findfirst(x -> x == (1, 1, R0), rbasis.bonds_L)
    ibL === nothing && error("onsite bond not found in bonds_L")
    ibR = findfirst(x -> x == (1, 1, R0), rbasis.bonds_R)
    ibR === nothing && error("onsite bond not found in bonds_R")
    iR_B = findfirst(x -> x == R0, rbasis.R_B_replicas[ibL, ibR])
    iR_B === nothing && error("onsite R_B not found")
    Γ0 = get_bare_vertex(Val(F), Val(C), U)
    @assert rbasis.R_B_ndegen[ibL, ibR][iR_B] == 1
    RealSpaceVertex{C}(rbasis, dictionary(((ibL, ibR, iR_B) => Γ0,)))
end
