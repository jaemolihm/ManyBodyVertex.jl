"""
    RealSpaceBubble{RC}(rbasis::RBT, bubbles_q)
"""
Base.@kwdef mutable struct RealSpaceBubble{F, RC, C, T, BF, BB, DT <: AbstractArray{T}, RBT} <: mfRG.AbstractBubble{F, C, T}
    rbasis::RBT
    # Basis for fermionic frequencies
    basis_f::BF
    # Basis for bosonic frequency
    basis_b::BB
    # Number of orbitals
    norb::Int
    # Data for bubbles, Π[ibL, ibR, iq].data = data_q[ibL, ibR][iq]
    data_q::Matrix{Vector{DT}}
    # Temperature (needed for integral_coeff in MF)
    temperature = nothing
    # Cached basis and overlap
    cache_basis_L = nothing
    cache_basis_R = nothing
    cache_overlap_LR = nothing
end

function RealSpaceBubble{RC}(rbasis::RBT, bubbles_q) where {RC, RBT}
    Π_ = first(first(bubbles_q))
    (; basis_f, basis_b, norb, temperature) = Π_
    data_q = [[y.data for y in x] for x in bubbles_q]
    DT = eltype(eltype(data_q))
    F = get_formalism(Π_)
    C = mfRG.channel(Π_)
    T = eltype(Π_)
    RealSpaceBubble{F, RC, C, T, typeof(basis_f), typeof(basis_b), DT, RBT}(;
        rbasis, basis_f, basis_b, norb, data_q, temperature)
end

real_space_channel(::RealSpaceBubble{F, RC}) where {F, RC} = RC

function Base.similar(Π::RealSpaceBubble{F, RC}) where {F, RC}
    typeof(Π)(; Π.rbasis, Π.basis_f, Π.basis_b, Π.norb, data_q=[similar.(x) for x in Π.data_q], Π.temperature)
end

function Base.:*(x::Number, Π::RealSpaceBubble)
    typeof(Π)(; Π.rbasis, Π.basis_f, Π.basis_b, Π.norb, data_q=[v .* x for v in Π.data_q], Π.temperature)
end

function Base.show(io::IO, A::RealSpaceBubble{F, RC}) where {F, RC}
    print(io, Base.typename(typeof(A)).wrapper, "{:$F, :$RC}, ")
    print(io, "$(Base.dims2string(size(A.data_q))) array of Vectors of bubbles.\n")
    print(io, "- Bubble data size: $(size(first(first(A.data_q))))\n")
    print(io, "- Total number of bubbles: $(sum(length, A.data_q))")
end

"""
    interpolate_to_q(A::RealSpaceBubble, xq, bL::mfRG.Bond, bR::mfRG.Bond)
TODO: Interpolation (linear / Fourier)
"""
@timeit timer "itp_Π" function interpolate_to_q(A::RealSpaceBubble, q, bL::mfRG.Bond, bR::mfRG.Bond)
    ibL = findfirst(x -> x == bL, A.rbasis.bonds_L)
    ibL === nothing && return nothing
    ibR = findfirst(x -> x == bR, A.rbasis.bonds_R)
    ibR === nothing && return nothing
    iq = findfirst(x -> x ≈ q, A.rbasis.qpts)
    iq === nothing && error("q point interpolation not implemented for RealSpaceBubble")

    F = mfRG.get_formalism(A)
    C = mfRG.channel(A)
    Bubble{F, C}(A.basis_f, A.basis_b, A.norb, A.data_q[ibL, ibR][iq]; A.temperature,
                 A.cache_basis_L, A.cache_basis_R, A.cache_overlap_LR)
end

function compute_bubble_nonlocal(G, basis_f, basis_b, ::Val{C}, q, nk; temperature=nothing) where {C}
    F = get_formalism(G)
    nind = get_nind(G)

    vs = get_fitting_points(basis_f)
    ws = get_fitting_points(basis_b)

    Π_data = zeros(eltype(G), length(vs), nind^4, length(ws))

    for ky in range(0, 1; length=nk+1)[1:end-1]
        for kx in range(0, 1; length=nk+1)[1:end-1]
            k = SVector(kx, ky)
            k1, k2 = _bubble_frequencies(Val(:ZF), Val(C), k, q)
            if !(G isa AbstractLazyGreen2P)
                # TODO: Cleanup
                G1_ = mfRG.interpolate_to_q(G, k1, 1, 1)
                G2_ = mfRG.interpolate_to_q(G, k2, 1, 1)
            end
            for (iw, w) in enumerate(ws)
                for (iv, v) in enumerate(vs)
                    v1, v2 = _bubble_frequencies(Val(F), Val(C), v, w)
                    if G isa AbstractLazyGreen2P
                        G1 = G(k1, v1)
                        G2 = G(k2, v2)
                    else
                        G1 = G1_(v1)
                        G2 = G2_(v2)
                    end
                    for (i, inds) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                        i11, i12, i21, i22 = _bubble_indices(Val(C), inds)
                        Π_data[iv, i, iw] += G1[i11, i12] * G2[i21, i22]
                    end
                end
            end
        end
    end
    Π_data .*= _bubble_prefactor(Val(C)) / nk^2
    Π_data_tmp1 = fit_basis_coeff(Π_data, basis_f, vs, 1)
    Π_data_tmp2 = fit_basis_coeff(Π_data_tmp1, basis_b, ws, 3)

    Π = Bubble{F, C}(basis_f, basis_b, G.norb; temperature)
    Π.data .= reshape(Π_data_tmp2, size(Π.data))
    Π
end

function _mapreduce_bubble_integrals(Γ1s, Π::RealSpaceBubble, Γ2s, basis_b)
    Γ1s_filtered = filter!(!isnothing, Γ1s)
    Γ2s_filtered = filter!(!isnothing, Γ2s)
    isempty(Γ1s_filtered) && return nothing
    isempty(Γ2s_filtered) && return nothing
    mapreduce(.+, Iterators.product(Γ1s_filtered, Γ2s_filtered)) do (Γ1, Γ2)
        vertex_bubble_integral.(Γ1, Π, Γ2, Ref(Π[1].rbasis.qgrid), Ref(basis_b))
    end
end
