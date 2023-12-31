"""
    RealSpaceBubble(RC, rbasis::RBT, bubbles_q)
"""
Base.@kwdef mutable struct RealSpaceBubble{F, T, BF, BB, DT <: AbstractArray{T}, RBT} <: AbstractBubble{F, T}
    # Frequency and real-space channels
    channel::Symbol
    real_space_channel::Symbol
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

function RealSpaceBubble(real_space_channel::Symbol, rbasis::RBT, bubbles_q) where {RBT}
    Π_ = first(first(bubbles_q))
    (; basis_f, basis_b, norb, temperature) = Π_
    data_q = [[y.data for y in x] for x in bubbles_q]
    DT = eltype(eltype(data_q))
    F = get_formalism(Π_)
    T = eltype(Π_)
    channel = get_channel(Π_)
    RealSpaceBubble{F, T, typeof(basis_f), typeof(basis_b), DT, RBT}(;
        channel, real_space_channel, rbasis, basis_f, basis_b, norb, data_q, temperature)
end

get_channel(Π::RealSpaceBubble) = Π.channel
real_space_channel(Π::RealSpaceBubble) = Π.real_space_channel

function Base.similar(Π::RealSpaceBubble{F}) where {F}
    typeof(Π)(; Π.channel, Π.real_space_channel, Π.rbasis, Π.basis_f, Π.basis_b, Π.norb,
        data_q=[zero.(x) for x in Π.data_q], Π.temperature)
end
Base.zero(Π::RealSpaceBubble) = similar(Π)

function _check_basis_identity(A::RealSpaceBubble, B::RealSpaceBubble)
    get_formalism(A) === get_formalism(B) || error("Different formalism")
    get_channel(A) === get_channel(B) || error("Different frequency channel")
    real_space_channel(A) === real_space_channel(B) || error("Different real-space channel")
    for n in (:rbasis, :basis_f, :basis_b, :norb)
        getproperty(A, n) === getproperty(B, n) || error("Different $n")
    end
end
data_fieldnames(::Type{<:RealSpaceBubble}) = (:data_q,)

function Base.:*(x::Number, Π::RealSpaceBubble)
    typeof(Π)(; Π.channel, Π.real_space_channel, Π.rbasis, Π.basis_f, Π.basis_b, Π.norb,
        data_q=[v .* x for v in Π.data_q], Π.temperature)
end

function Base.show(io::IO, A::RealSpaceBubble{F, RC}) where {F, RC}
    C = get_channel(A)
    print(io, Base.typename(typeof(A)).wrapper, "{:$F, :$RC}, ")
    print(io, "$(Base.dims2string(size(A.data_q))) array of Vectors of bubbles in channel $C.\n")
    print(io, "- Bubble data size: $(size(first(first(A.data_q))))\n")
    print(io, "- Total number of bubbles: $(sum(length, A.data_q))")
end

"""
    interpolate_to_q(A::RealSpaceBubble, xq, bL::Bond, bR::Bond)
TODO: Interpolation (linear / Fourier)
"""
@timeit timer "itp_Π" function interpolate_to_q(A::RealSpaceBubble, q, bL::Bond, bR::Bond)
    ibL = findfirst(x -> x == bL, A.rbasis.bonds_L)
    ibL === nothing && return nothing
    ibR = findfirst(x -> x == bR, A.rbasis.bonds_R)
    ibR === nothing && return nothing
    iq = findfirst(x -> x ≈ q, A.rbasis.qpts)
    iq === nothing && error("q point interpolation not implemented for RealSpaceBubble")

    F = get_formalism(A)
    C = get_channel(A)
    Bubble{F}(C, A.basis_f, A.basis_b, A.norb, A.data_q[ibL, ibR][iq]; A.temperature,
                 A.cache_basis_L, A.cache_basis_R, A.cache_overlap_LR)
end

function compute_bubble_nonlocal(G1, G2, basis_f, basis_b, C::Symbol, q, nk; temperature=nothing)
    F = get_formalism(G1)
    nind = get_nind(G1)

    vs = get_fitting_points(basis_f)
    ws = get_fitting_points(basis_b)

    Π_data = zeros(eltype(G1), length(vs), nind^4, length(ws))

    for ky in range(0, 1; length=nk+1)[1:end-1]
        for kx in range(0, 1; length=nk+1)[1:end-1]
            k = SVector(kx, ky)
            k1, k2 = _bubble_frequencies(Val(:ZF), C, k, q)
            # TODO: Cleanup
            G1_ = G1 isa AbstractLazyGreen2P ? G1 : interpolate_to_q(G1, k1, 1, 1)
            G2_ = G2 isa AbstractLazyGreen2P ? G2 : interpolate_to_q(G2, k2, 1, 1)
            for (iw, w) in enumerate(ws)
                for (iv, v) in enumerate(vs)
                    v1, v2 = _bubble_frequencies(Val(F), C, v, w)
                    G1_v = G1 isa AbstractLazyGreen2P ? G1(k1, v1) : G1_(v1)
                    G2_v = G2 isa AbstractLazyGreen2P ? G2(k2, v2) : G2_(v2)
                    for (i, inds) in enumerate(Iterators.product(1:nind, 1:nind, 1:nind, 1:nind))
                        i11, i12, i21, i22 = _bubble_indices(C, inds)
                        Π_data[iv, i, iw] += G1_v[i11, i12] * G2_v[i21, i22]
                    end
                end
            end
        end
    end
    Π_data .*= _bubble_prefactor(C) / nk^2
    Π_data_tmp1 = fit_basis_coeff(Π_data, basis_f, vs, 1)
    Π_data_tmp2 = fit_basis_coeff(Π_data_tmp1, basis_b, ws, 3)

    Π = Bubble{F}(C, basis_f, basis_b, G1.norb; temperature)
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


"""
    compute_bubble_nonlocal_real_space(G1::RealSpaceGreen2P{F}, G2::RealSpaceGreen2P{F},
            basis_f, basis_b, C::Symbol, rbasis; temperature=nothing, smooth_bubble=false)
(k1, k2) = ( k + q/2, k - q/2) : channel A, T
         = (-k - q/2, k - q/2) : channel P

Π(q) = ∑_k G(k1) * G(k2)
     = ∑_{k, R1, R2} exp(i*k1*R1) * exp(i*k2*R2) * G(R1) * G(R2)
     = ∑_{R} exp(i*q*R) * G(R) * G(-R)  : channel A, T
     = ∑_{R} exp(i*q*R) * G(-R) * G(-R) : channel P
"""
function compute_bubble(G1::RealSpaceGreen2P{F}, G2::RealSpaceGreen2P{F}, basis_f, basis_b,
            C::Symbol, rbasis; temperature=nothing, smooth_bubble=false) where {F}
    RC = C
    Πs = [[Bubble{F}(C, eltype(G1), basis_f, basis_b, G1.norb; temperature) for _ in rbasis.qpts]
        for _ in eachindex(rbasis.bonds_L), _ in eachindex(rbasis.bonds_R)]

    # Precompute overlap_bubble
    if smooth_bubble
        ws = get_fitting_points(basis_b)
        overlap_bubble_cached = [basis_integral_bubble(basis_f, G1.basis, G2.basis, w, C)
                                 for w in ws]
    end

    for ibR in eachindex(rbasis.bonds_R), ibL in eachindex(rbasis.bonds_L)
        bL = rbasis.bonds_L[ibL]
        bR = rbasis.bonds_R[ibR]
        iatm11, iatm12, iatm21, iatm22 = _bubble_indices(C, (bL[2], bL[1], bR[1], bR[2]))

        Rs = G1.rbasis.R_replicas[iatm11, iatm12]
        Π_R_datas = [zeros(eltype(G1), size(first(Πs[ibL, ibR]).data)) for _ in eachindex(Rs)]

        Threads.@threads for iR in eachindex(Rs)
            R1 = Rs[iR]
            R2 = (C === :A || C === :T) ? bL[3] - bR[3] - R1 : bL[3] - bR[3] + R1

            G1_R = G1[iatm11, iatm12, R1]
            G2_R = G2[iatm21, iatm22, R2]
            G1_R === nothing && continue
            G2_R === nothing && continue

            Π_R_datas[iR] .= if smooth_bubble
                compute_bubble_smoothed(G1_R, G2_R, basis_f, basis_b, C,
                    overlap_bubble_cached; temperature).data
            else
                compute_bubble(G1_R, G2_R, basis_f, basis_b, C; temperature).data
            end
        end

        for iR in eachindex(Rs)
            R1 = Rs[iR]
            R2 = (C === :A || C === :T) ? bL[3] - bR[3] - R1 : bL[3] - bR[3] + R1
            for (iq, xq) in enumerate(rbasis.qpts)
                R = (C === :A || C === :T) ? (R1 - R2) / 2 : (-R1 - R2) / 2
                coeff = cispi(2 * xq' * R)
                Πs[ibL, ibR][iq].data .+= Π_R_datas[iR] .* coeff
            end
        end
    end
    RealSpaceBubble(RC, rbasis, Πs)
end
