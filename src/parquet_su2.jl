# FIXME: non-SU2

function iterate_parquet_asymptotic_single_channel_BSE(Π, U, γ, Irr;
                                        max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = channel(Π[1])
    K1_only_U = solve_BSE.(U, Π, U, Ref(basis_k1_b))
    Πscr = ScreenedBubble.(Π, U, K1_only_U)

    K1_new = K1_only_U
    K2_new = nothing
    K2p_new = nothing
    K3_new = nothing

    if !isempty(Irr)
        ws = get_fitting_points(basis_k2_b)
        Irr_mat = Tuple(cache_vertex_matrix(getindex.(Irr, i), C, ws, basis_k2_f) for i in 1:2);
        K2_new = solve_BSE.(Irr_mat, Πscr, U, Ref(basis_k2_b))
        K1_new = K1_new .+ vertex_bubble_integral.(U, Πscr, K2_new, Ref(basis_k1_b))
        if max_class >= 2
            K2p_new = solve_BSE_left.(Irr_mat, Πscr, U, Ref(basis_k2_b))
        end
        if max_class >= 3
            K3_new = solve_BSE.(Irr_mat, Πscr, Irr_mat, Ref(basis_k2_b))
        end
    end

    (; K1=K1_new, K2=K2_new, K2p=K2p_new, K3=K3_new)
end

function _mapreduce_bubble_integrals(Γ1s, Π, Γ2s, basis_b)
    Γ1s_filtered = filter!(!isnothing, Γ1s)
    Γ2s_filtered = filter!(!isnothing, Γ2s)
    isempty(Γ1s_filtered) && return nothing
    isempty(Γ2s_filtered) && return nothing
    mapreduce(.+, Iterators.product(Γ1s_filtered, Γ2s_filtered)) do (Γ1, Γ2)
        vertex_bubble_integral.(Γ1, Π, Γ2, Ref(basis_b))
    end
end

function iterate_parquet_asymptotic_single_channel(Π, U, γ, Irr;
                                        max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = channel(Π[1])
    K1_new = _mapreduce_bubble_integrals([U], Π, [U, γ.K1, γ.K2], basis_k1_b)

    if max_class >= 2 && !isempty(Irr)
        ws = get_fitting_points(basis_k2_b)
        Irr_mat = Tuple(cache_vertex_matrix(getindex.(Irr, i), C, ws, basis_k2_f) for i in 1:2);
        K2_new = _mapreduce_bubble_integrals([Irr_mat], Π, [U, γ.K1, γ.K2], basis_k2_b)
        K2p_new = _mapreduce_bubble_integrals([U, γ.K1, γ.K2p], Π, [Irr_mat], basis_k2_b)
    else
        K2_new = nothing
        K2p_new = nothing
    end

    if max_class >= 3 && !isempty(Irr)
        K3_new = _mapreduce_bubble_integrals([Irr_mat], Π, [Irr_mat, γ.K2p, γ.K3], basis_k2_b)
    else
        K3_new = nothing
    end

    (; K1=K1_new, K2=K2_new, K2p=K2p_new, K3=K3_new)
end

function iterate_parquet(Γ::AsymptoticVertex, ΠA, ΠP; iterate_by_bse=false)
    single_channel_iterate_function = if iterate_by_bse
        iterate_parquet_asymptotic_single_channel_BSE
    else
        iterate_parquet_asymptotic_single_channel
    end

    # BSE for channel A
    @info "Solving BSE for channel A"
    K1_A, K2_A, K2p_A, K3_A = single_channel_iterate_function(
        ΠA, Γ.Γ0_A, get_reducible_vertices(:A, Γ), get_irreducible_vertices(:A, Γ);
        Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_f, Γ.basis_k2_b
    )

    # BSE for channel P
    @info "Solving BSE for channel P"
    K1_P, K2_P, K2p_P, K3_P = single_channel_iterate_function(
        ΠP, Γ.Γ0_P, get_reducible_vertices(:P, Γ), get_irreducible_vertices(:P, Γ);
        Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_f, Γ.basis_k2_b
    )

    # Channel T: use crossing symmetry
    @info "Applying crossing symmetry for channel T"
    K1_T = su2_apply_crossing(K1_A)
    K2_T = su2_apply_crossing(K2_A)
    K2p_T = su2_apply_crossing(K2p_A)
    K3_T = su2_apply_crossing(K3_A)

    Γs = (; Γ.Γ0_A, Γ.Γ0_P, Γ.Γ0_T, K1_A, K1_P, K1_T)
    if Γ.max_class >= 2
        Γs = (; Γs..., K2_A, K2_P, K2_T, K2p_A, K2p_P, K2p_T)
    end
    if Γ.max_class >= 3
        Γs = (; Γs..., K3_A, K3_P, K3_T)
    end
    typeof(Γ)(; Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_b, Γ.basis_k2_f, Γs...)
end

function _compute_self_energy(Γ, G, v, overlap=nothing; temperature=nothing)
    F = get_formalism(Γ)
    C = channel(Γ)
    nind = get_nind(G)
    nb_f1(Γ) == 1 || error("Γ.basis_f1 must be a constant basis")
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")
    if overlap === nothing
        overlap = basis_integral_self_energy(Γ.basis_f2, Γ.basis_b, G.basis, v, Val(C))
    end
    # (ab, icd, j) -> (a, b, c, d, ij)
    Γ_data = reshape(permutedims(reshape(Γ.data, (nind^2, nb_f2(Γ), nind^2, nb_b(Γ))), (1, 3, 2, 4)), nind^4, :)
    # (a, b, c, d, ij) * (ij, k) -> (a, b, c, d, k)
    Γ_data_2 = reshape(Γ_data * reshape(overlap, (:, nbasis(G.basis))), nind, nind, nind, nind, nbasis(G.basis))

    Σ_v = zeros(eltype(G), nind, nind)
    @views for b in 1:nind, d in 1:nind, c in 1:nind, a in 1:nind
        if C === :P
            Σ_v[a, d] += transpose(Γ_data_2[a, b, c, d, :]) * G.data[c, b, :]
        else
            Σ_v[a, d] += transpose(Γ_data_2[a, b, c, d, :]) * G.data[b, c, :]
        end
    end
    Σ_v .*= (integral_coeff(Val(F), temperature) * -1 / 2)
    Σ_v
end

function _compute_self_energy_SU2(Γ, G, v; temperature=nothing)
    C = channel(Γ[1])
    overlap = basis_integral_self_energy(Γ[1].basis_f2, Γ[1].basis_b, G.basis, v, Val(C))
    coeff = C === :P ? (-1/2, 3/2) : (1/2, 3/2)
    (  _compute_self_energy(Γ[1], G, v, overlap; temperature) .* coeff[1]
    .+ _compute_self_energy(Γ[2], G, v, overlap; temperature) .* coeff[2])
end

"""
    compute_self_energy_SU2(Γ, G, ΠA, ΠP, basis=G.basis; temperature=nothing)
Compute the self-energy by solving the Schwinger-Dyson equation.
``Σ[a, d](v) = ∫ dw K1[ab, cd](v+w/2, w) * G[b, c](v+w) * integral_coeff * -1/2``
"""
function compute_self_energy_SU2(Γ, G, ΠA, ΠP, basis=G.basis; temperature=nothing)
    F = get_formalism(Γ)
    nind = get_nind(G)
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")

    vs = get_fitting_points(basis)
    Σ_data_iv = zeros(ComplexF64, nind, nind, length(vs))

    # If G is lazily defined, compute G on the basis explicitly.
    if G isa AbstractLazyGreen2P
        vs = get_fitting_points(basis)
        G_data = zeros(eltype(G), nind, nind, length(vs))
        for (iv, v) in enumerate(vs)
            G_data[:, :, iv] .= G(v)
        end
        G_ = Green2P{F}(basis, G.norb, fit_basis_coeff(G_data, basis, vs, 3))
    else
        G_ = G
    end

    U_ΠA_U = vertex_bubble_integral.(Γ.Γ0_A, ΠA, Γ.Γ0_A, Ref(Γ.basis_k1_b))
    U_ΠA_K2pK3 = _mapreduce_bubble_integrals([Γ.Γ0_A], ΠA, [Γ.K2p_A, Γ.K3_A], Γ.basis_k2_b)

    U_ΠP_U = vertex_bubble_integral.(Γ.Γ0_P, ΠP, Γ.Γ0_P, Ref(Γ.basis_k1_b))
    U_ΠP_K2pK3 = _mapreduce_bubble_integrals([Γ.Γ0_P], ΠP, [Γ.K2p_P, Γ.K3_P], Γ.basis_k2_b)

    Base.Threads.@threads for iv in eachindex(vs)
        # U_Π_U term (the O(U²) term): to avoid double counting, 1/3 for each channel.
        # Need to consider that K1 already contains this term.
        # Also, for P channel, multiply 2 because Π has factor 1/2.
        # A, T channel: (1/3 + 1/3) - 2 (from K1_A) = -4/3
        # P channel: (1/3 * 2) - 2 (from K1_P) = -4/3
        # Factor 2 is multiplied outside the loop, so we use multiply -2/3.
        v = vs[iv]
        Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(Γ.K1_A, G_, v; temperature)
        Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(U_ΠA_U, G_, v; temperature) .* (-2/3)
        if U_ΠA_K2pK3 !== nothing
            Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(U_ΠA_K2pK3, G_, v; temperature)
        end

        Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(Γ.K1_P, G_, v; temperature)
        Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(U_ΠP_U, G_, v; temperature) .* (-2/3)
        if U_ΠP_K2pK3 !== nothing
            Σ_data_iv[:, :, iv] .+= _compute_self_energy_SU2(U_ΠP_K2pK3, G_, v; temperature)
        end
    end

    # Multiply by factor 2
    # For the A channel, this accounts for the contribution of the T channel.
    # For the P channel, this account for the factor of 1/2 in the bubble.
    Σ_data_iv .*= 2

    Σ_data = fit_basis_coeff(Σ_data_iv, basis, vs, 3)
    Green2P{F}(basis, 1, Σ_data)
end

function setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
    bubble_function = smooth_bubble ? compute_bubble_smoothed : compute_bubble
    @time ΠA_ = bubble_function(G, basis_v_bubble, basis_w_bubble, Val(:A); temperature)
    @time ΠP_ = bubble_function(G, basis_v_bubble, basis_w_bubble, Val(:P); temperature)
    ΠA = (ΠA_, ΠA_)
    ΠP = (ΠP_ * -1, ΠP_)
    (; ΠA, ΠP)
end

function run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_k1_b, basis_k2_b, basis_k2_f, basis_1p=G0.basis;
        max_class=3, max_iter=5, reltol=1e-2, temperature=nothing, smooth_bubble=false,
        mixing_history=10, mixing_coeff=0.5, iterate_by_bse=false)
    F = get_formalism(G0)
    T = eltype(G0)

    Γ0_A = su2_bare_vertex(U, Val(F), Val(:A))
    Γ0_P = su2_bare_vertex(U, Val(F), Val(:P))
    Γ0_T = su2_apply_crossing(Γ0_A)

    # 1st iteration
    ΠA, ΠP = setup_bubble_SU2(G0, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
    Γ = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T, basis_k1_b, basis_k2_b, basis_k2_f)

    # Initialize self-energy and Green function. Here Σ = 0, G = G0.
    Σ = Green2P{F}(basis_1p, G0.norb)
    G = solve_Dyson(G0, Σ)

    acceleration = AndersonAcceleration(; m=mixing_history)

    for i in 1:max_iter
        @info "== Iteration $i =="
        @time Γ_new = iterate_parquet(Γ, ΠA, ΠP; iterate_by_bse)

        err = get_difference_norm(Γ_new, Γ)

        # Mixing
        if i <= 2
            # For the first two iterations, new class of vertices may appear, so we do not
            # apply mixing.
            Γ = Γ_new
        else
            Γ_vec = vertex_to_vector(Γ)
            Γ_diff_vec = vertex_to_vector(Γ_new) .- Γ_vec
            Γ = vector_to_vertex(acceleration(Γ_vec, mixing_coeff, Γ_diff_vec), Γ)
        end

        @info "Updating self-energy and the bubble"
        @time Σ = compute_self_energy_SU2(Γ, G, ΠA, ΠP; temperature)
        G = solve_Dyson(G0, Σ)
        ΠA, ΠP = setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)

        @info err
        if err.relerr < reltol
            @info "Convergence reached"
            break
        end
    end
    (; Γ, Σ)
end
