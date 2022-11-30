# FIXME: non-SU2

function iterate_parquet_asymptotic_single_channel_BSE(Π, Γ0, K1, K2, K2p, K3, Γirr;
                                        max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = channel(Π[1])
    K1_only_Γ0 = solve_BSE.(Γ0, Π, Γ0, Ref(basis_k1_b))
    Πscr = ScreenedBubble.(Π, Γ0, K1_only_Γ0)

    K1_new = K1_only_Γ0
    K2_new = nothing
    K2p_new = nothing
    K3_new = nothing

    if !isempty(Γirr)
        ws = get_fitting_points(basis_k2_b)
        Γirr_mat = Tuple(cache_vertex_matrix(getindex.(Γirr, i), C, ws, basis_k2_f) for i in 1:2);
        K2_new = solve_BSE.(Γirr_mat, Πscr, Γ0, Ref(basis_k2_b))
        K1_new = K1_new .+ vertex_bubble_integral.(Γ0, Πscr, K2_new, Ref(basis_k1_b))
        if max_class >= 2
            K2p_new = solve_BSE_left.(Γirr_mat, Πscr, Γ0, Ref(basis_k2_b))
        end
        if max_class >= 3
            K3_new = solve_BSE.(Γirr_mat, Πscr, Γirr_mat, Ref(basis_k2_b))
        end
    end

    (; K1_new, K2_new, K2p_new, K3_new)
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

function iterate_parquet_asymptotic_single_channel(Π, Γ0, K1, K2, K2p, K3, Γirr;
                                        max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = channel(Π[1])
    K1_new = _mapreduce_bubble_integrals([Γ0], Π, [Γ0, K1, K2], basis_k1_b)

    if max_class >= 2 && !isempty(Γirr)
        ws = get_fitting_points(basis_k2_b)
        Γirr_mat = Tuple(cache_vertex_matrix(getindex.(Γirr, i), C, ws, basis_k2_f) for i in 1:2);
        K2_new = _mapreduce_bubble_integrals([Γirr_mat], Π, [Γ0, K1, K2], basis_k2_b)
        K2p_new = _mapreduce_bubble_integrals([Γ0, K1, K2p], Π, [Γirr_mat], basis_k2_b)
    else
        K2_new = nothing
        K2p_new = nothing
    end

    if max_class >= 3 && !isempty(Γirr)
        K3_new = _mapreduce_bubble_integrals([Γirr_mat], Π, [Γirr_mat, K2p, K3], basis_k2_b)
    else
        K3_new = nothing
    end

    (; K1_new, K2_new, K2p_new, K3_new)
end

function iterate_parquet(Γ::AsymptoticVertex, ΠA, ΠP; iterate_by_bse=false)
    single_channel_iterate_function = if iterate_by_bse
        iterate_parquet_asymptotic_single_channel_BSE
    else
        iterate_parquet_asymptotic_single_channel
    end

    # BSE for channel A
    @info "Solving BSE for channel A"
    Γ_A_irr = get_irreducible_vertices(:A, Γ)
    K1_A, K2_A, K2p_A, K3_A = single_channel_iterate_function(
        ΠA, Γ.Γ0_A, Γ.K1_A, Γ.K2_A, Γ.K2p_A, Γ.K3_A, Γ_A_irr;
        Γ.max_class, Γ.basis_k1_b, Γ.basis_k2_f, Γ.basis_k2_b
    )

    # BSE for channel P
    @info "Solving BSE for channel P"
    Γ_P_irr = get_irreducible_vertices(:P, Γ)
    K1_P, K2_P, K2p_P, K3_P = single_channel_iterate_function(
        ΠP, Γ.Γ0_P, Γ.K1_P, Γ.K2_P, Γ.K2p_P, Γ.K3_P, Γ_P_irr;
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

function compute_self_energy_SU2(Γ, G, basis=G.basis; temperature=nothing)
    F = mfRG.get_formalism(Γ)
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")
    nind = get_nind(G)
    if nind > 1 || F === :KF
        @warn "Multiorbital or Keldysh not implemented yet, returning zero self-energy"
        return Green2P{F}(basis, G.norb)
    end

    vs = get_fitting_points(basis)
    Σ_data_iv = zeros(ComplexF64, nind, nind, length(vs))

    # FIXME
    wmax = 500
    Base.Threads.@threads for iv in eachindex(vs)
        v2 = vs[iv]
        Γ_vw = zeros(eltype(Γ), nind, nind)
        for v1 in -wmax:wmax
            G_v1 = G(v1)[1, 1]
            v, w = mfRG._bubble_frequencies_inv(Val(F), Val(:A), v1, v2)
            Γ_vw .= 0
            if Γ.K1_A !== nothing
                Γ_vw .= Γ.K1_A[1](0, v, w) ./ 2 .+ Γ.K1_A[2](0, v, w) .* 3 ./ 2
            end
            if Γ.K2p_A !== nothing
                Γ_vw .+= Γ.K2p_A[1](0, v, w) ./ 2 .+ Γ.K2p_A[2](0, v, w) .* 3 ./ 2
            end
            Σ_data_iv[:, :, iv] .+= Γ_vw .* G_v1
        end
    end

    if F === :MF
        Σ_data_iv .*= -temperature / 2
    else
        error("KF and ZF not implemented yet")
    end

    Σ_data = mfRG.fit_basis_coeff(Σ_data_iv, basis, vs, 3)
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
        max_class, max_iter=5, reltol=1e-2, temperature=nothing, smooth_bubble=false)
    F = get_formalism(G0)
    T = eltype(G0)

    Γ0_A = su2_bare_vertex(U, Val(F), Val(:A))
    Γ0_P = su2_bare_vertex(U, Val(F), Val(:P))
    Γ0_T = su2_apply_crossing(Γ0_A)

    # 1st iteration
    ΠA, ΠP = setup_bubble_SU2(G0, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
    vertex = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T, basis_k1_b, basis_k2_b, basis_k2_f)

    # Initialize self-energy and Green function. Here Σ = 0, G = G0.
    Σ = Green2P{F}(basis_1p, G0.norb)
    G = solve_Dyson(G0, Σ)

    for i in 1:max_iter
        @info "== Iteration $i =="
        @time vertex_new = iterate_parquet(vertex, ΠA, ΠP)

        Γ_diff = get_difference_norm(vertex_new, vertex)

        vertex = vertex_new
        # Mixing
        if i == 1
            vertex = vertex_new
        else i >= 2
            mixing_coeff = 0.2
            # mixing_coeff = i <= 20 ? 0.1 : 0.02
            new_vertices = map(_vertex_names(vertex_new)) do name
                xold = getproperty(vertex, name)
                xnew = getproperty(vertex_new, name)
                if xold !== nothing && xnew !== nothing
                    (name, xnew .* mixing_coeff .+ xold .* (1 - mixing_coeff))
                elseif xnew !== nothing
                    (name, xnew .* mixing_coeff)
                elseif xold !== nothing
                    (name, xold .* (1 - mixing_coeff))
                else
                    (name, nothing)
                end
            end
            vertex = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T, new_vertices...)
        end

        @info "Updating self-energy and the bubble"
        if mod1(i, 10) == 1
            @time Σ = compute_self_energy_SU2(vertex, G; temperature)
            G = solve_Dyson(G0, Σ)
            ΠA, ΠP = setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)
        end

        @info Γ_diff
        if Γ_diff.relerr < reltol
            @info "Convergence reached"
            break
        end
    end
    (; vertex, Σ)
end
