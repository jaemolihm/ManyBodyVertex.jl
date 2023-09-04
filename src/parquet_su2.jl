# FIXME: non-SU2
# FIXME: Exclude Λ from Anderson

function _mapreduce_bubble_integrals(Γ1s, Π, Γ2s, basis_b)
    Γ1s_filtered = filter!(!isnothing, Γ1s)
    Γ2s_filtered = filter!(!isnothing, Γ2s)
    isempty(Γ1s_filtered) && return nothing
    isempty(Γ2s_filtered) && return nothing
    mapreduce(.+, Iterators.product(Γ1s_filtered, Γ2s_filtered)) do (Γ1, Γ2)
        vertex_bubble_integral.(Γ1, Π, Γ2, Ref(basis_b))
    end
end

function iterate_parquet_single_channel_asymptotic(Π, U, γ, Irr;
                                        max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = get_channel(Π[1])
    # K1_new = _mapreduce_bubble_integrals([U], Π, [U, γ.K1, γ.K2], basis_k1_b)
    K1_new = (
        _mapreduce_bubble_integrals([U], Π, [U, γ.K1, γ.K2], basis_k1_b)
        .+ _mapreduce_bubble_integrals([U, γ.K1, γ.K2p], Π, [U], basis_k1_b)
    ) ./ 2

    if max_class >= 2 && !isempty(Irr)
        ws = get_fitting_points(basis_k2_b.freq)
        Irr_mat = Tuple(cache_vertex_matrix(getindex.(Irr, i), C, ws, basis_k2_f) for i in 1:2);
        K2_new = _mapreduce_bubble_integrals([γ.K2, γ.K3, Irr_mat], Π, [U], basis_k2_b)
        K2p_new = _mapreduce_bubble_integrals([U], Π, [γ.K2p, γ.K3, Irr_mat], basis_k2_b)
    else
        K2_new = nothing
        K2p_new = nothing
    end

    if max_class >= 3 && !isempty(Irr)
        # K3_new = _mapreduce_bubble_integrals([Irr_mat], Π, [Irr_mat, γ.K2p, γ.K3], basis_k2_b)
        K3_new = (
            _mapreduce_bubble_integrals([Irr_mat], Π, [Irr_mat, γ.K2p, γ.K3], basis_k2_b)
            .+ _mapreduce_bubble_integrals([Irr_mat, γ.K2, γ.K3], Π, [Irr_mat], basis_k2_b)
        ) ./ 2
    else
        K3_new = nothing
    end

    (; K1=K1_new, K2=K2_new, K2p=K2p_new, K3=K3_new)
end

function iterate_parquet_single_channel(C, Γ::AsymptoticVertex, Π)
    U = Γ(C, :Γ0)
    γ = get_reducible_vertices(C, Γ)
    Irr = get_irreducible_vertices(C, Γ)
    # Function barrier
    iterate_parquet_single_channel_asymptotic(
        Π, U, γ, Irr; Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_f, Γ.basis_k2_b
    )
end

function iterate_parquet(Γ::AsymptoticVertex, ΠA, ΠP)
    # BSE for channel A
    @info "Solving BSE for channel A"
    K1_A, K2_A, K2p_A, K3_A = iterate_parquet_single_channel(:A, Γ, ΠA)

    # BSE for channel P
    @info "Solving BSE for channel P"
    K1_P, K2_P, K2p_P, K3_P = iterate_parquet_single_channel(:P, Γ, ΠP)

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
    Λ_A = isnothing(Γ.Λ_A) ? nothing : copy.(Γ.Λ_A)
    Λ_P = isnothing(Γ.Λ_P) ? nothing : copy.(Γ.Λ_P)
    Λ_T = isnothing(Γ.Λ_T) ? nothing : copy.(Γ.Λ_T)
    typeof(Γ)(; Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_b, Γ.basis_k2_f, Λ_A, Λ_P, Λ_T, Γs...)
end

function setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature,
        smooth_bubble=get_formalism(G) === :MF ? false : true)
    bubble_function = smooth_bubble ? compute_bubble_smoothed : compute_bubble
    @time ΠA_ = bubble_function(G, G, basis_v_bubble, basis_w_bubble, :A; temperature)
    @time ΠP_ = bubble_function(G, G, basis_v_bubble, basis_w_bubble, :P; temperature)
    ΠA = (ΠA_, ΠA_)
    ΠP = (ΠP_ * -1, ΠP_)
    (; ΠA, ΠP)
end

function run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_k1_b, basis_k2_b, basis_k2_f, basis_1p=G0.basis;
        max_class=3, max_iter=5, reltol=1e-2, temperature=nothing,
        smooth_bubble=get_formalism(G0) === :MF ? false : true,
        mixing_history=10, mixing_coeff=0.5,
        Γ_init=nothing, Σ_init=nothing)
    F = get_formalism(G0)
    T = eltype(G0)

    Γ0_A = su2_bare_vertex(Val(F), :A, U)
    Γ0_P = su2_bare_vertex(Val(F), :P, U)
    Γ0_T = su2_apply_crossing(Γ0_A)

    # 1st iteration
    if Γ_init === nothing
        Γ = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T, basis_k1_b=(; freq=basis_k1_b), basis_k2_b=(; freq=basis_k2_b), basis_k2_f=(; freq=basis_k2_f))
    else
        Γ = Γ_init
    end

    # Initialize self-energy and Green function.
    if Σ_init === nothing
        Σ = Green2P{F}(basis_1p, G0.norb)  # Σ = 0
        G = green_to_basis(G0, basis_1p)  # G = G0
        ΠA, ΠP = setup_bubble_SU2(G0, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
    else
        Σ = green_to_basis(Σ_init, basis_1p)
        G = solve_Dyson(G0, Σ)
        ΠA, ΠP = setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
    end

    acceleration = AndersonAcceleration(; m=mixing_history)

    for i in 1:max_iter
        @info "== Iteration $i =="
        GC.gc()
        @time Γ_new = iterate_parquet(Γ, ΠA, ΠP)

        err = get_difference_norm(Γ_new, Γ)

        # Mixing
        if i <= 2
            # For the first two iterations, new class of vertices may appear, so we do not
            # apply mixing.
            Γ = Γ_new
        else
            Γ_vec = vertex_to_vector(Γ)
            Γ_diff_vec = vertex_to_vector(Γ_new) .- Γ_vec
            GC.gc()
            Γ = vector_to_vertex(acceleration(Γ_vec, mixing_coeff, Γ_diff_vec), Γ)
            GC.gc()
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
    (; Γ, Σ, Π=(; A=ΠA, P=ΠP))
end
