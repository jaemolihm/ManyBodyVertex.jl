# FIXME: non-SU2

function iterate_parquet(Γ::AsymptoticVertex, ΠAscr, ΠPscr, basis_aux)
    basis_w = Γ.basis_k1_b
    ws = get_fitting_points(basis_w)
    (; Γ0_A, Γ0_P, Γ0_T) = Γ

    # BSE for channel A
    @info "Solving BSE for channel A"
    Γ_A_irr = get_irreducible_vertices(:A, Γ)
    @time Γ_A_irr_mat = Tuple(cache_vertex_matrix(getindex.(Γ_A_irr, i), :A, ws, basis_aux) for i in 1:2);

    @time K2_A = solve_BSE.(Γ_A_irr_mat, ΠAscr, Γ.Γ0_A, Ref(basis_w); basis_aux)
    @time K1_A = vertex_bubble_integral.(Γ.Γ0_A, ΠAscr, K2_A, Ref(basis_w)) .+ getproperty.(ΠAscr, :K1)

    if Γ.max_class >= 2
        @time K2p_A = solve_BSE_left.(Γ_A_irr_mat, ΠAscr, Γ.Γ0_A, Ref(basis_w); basis_aux)
    end
    if Γ.max_class >= 3
        @time K3_A = solve_BSE.(Γ_A_irr_mat, ΠAscr, Γ_A_irr_mat, Ref(basis_w); basis_aux)
    end

    # BSE for channel P
    @info "Solving BSE for channel P"
    Γ_P_irr = get_irreducible_vertices(:P, Γ)
    @time Γ_P_irr_mat = Tuple(cache_vertex_matrix(getindex.(Γ_P_irr, i), :P, ws, basis_aux) for i in 1:2);

    @time K2_P = solve_BSE.(Γ_P_irr_mat, ΠPscr, Γ.Γ0_P, Ref(basis_w); basis_aux)
    @time K1_P = vertex_bubble_integral.(Γ.Γ0_P, ΠPscr, K2_P, Ref(basis_w)) .+ getproperty.(ΠPscr, :K1)

    if Γ.max_class >= 2
        @time K2p_P = solve_BSE_left.(Γ_P_irr_mat, ΠPscr, Γ.Γ0_P, Ref(basis_w); basis_aux)
    end
    if Γ.max_class >= 3
        @time K3_P = solve_BSE.(Γ_P_irr_mat, ΠPscr, Γ_P_irr_mat, Ref(basis_w); basis_aux)
    end

    # Channel T: use crossing symmetry
    @info "Applying crossing symmetry for channel T"
    K1_T = su2_apply_crossing(K1_A)
    if Γ.max_class >= 2
        K2_T = su2_apply_crossing(K2_A)
        K2p_T = su2_apply_crossing(K2p_A)
    end
    if Γ.max_class >= 3
        K3_T = su2_apply_crossing(K3_A)
    end

    Γs = (; Γ0_A, Γ0_P, Γ0_T, K1_A, K1_P, K1_T)
    if Γ.max_class >= 2
        Γs = (; Γs..., K2_A, K2_P, K2_T, K2p_A, K2p_P, K2p_T)
    end
    if Γ.max_class >= 3
        Γs = (; Γs..., K3_A, K3_P, K3_T)
    end
    typeof(Γ)(; Γ.max_class, Γs...)
end

function compute_self_energy_SU2(Γ, G, basis=G.basis; temperature=nothing)
    F = mfRG.get_formalism(Γ)
    F === :MF && temperature === nothing && error("For MF, temperature must be provided")
    nind = get_nind(G)
    Γ_vw = zeros(eltype(Γ), nind, nind)

    nind > 1 && error("Multiorbital not implemented yet")

    vs = get_fitting_points(basis)
    Σ_data_iv = zeros(ComplexF64, nind, nind, length(vs))

    # FIXME
    wmax = 500
    for (iv, v1) in enumerate(vs)
        for v2 in -wmax:wmax
            G_v2 = G(v2)[1, 1]
            v, w = mfRG._bubble_frequencies_inv(Val(F), Val(:A), v1, v2)
            Γ_vw .= 0
            if Γ.K1_A !== nothing
                Γ_vw .= Γ.K1_A[1](0, v, w) ./ 2 .+ Γ.K1_A[2](0, v, w) .* 3 ./ 2
            end
            if Γ.K2p_A !== nothing
                Γ_vw .+= Γ.K2p_A[1](0, v, w) ./ 2 .+ Γ.K2p_A[2](0, v, w) .* 3 ./ 2
            end
            Σ_data_iv[:, :, iv] .+= Γ_vw .* G_v2
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

function setup_screened_bubble(G, Γ0_A, Γ0_P, basis_v_bubble, basis_w_bubble, basis_w; temperature, smooth_bubble)
    bubble_function = smooth_bubble ? compute_bubble_smoothed : compute_bubble
    @time ΠA_ = bubble_function(G, basis_v_bubble, basis_w_bubble, Val(:A); temperature)
    @time ΠP_ = bubble_function(G, basis_v_bubble, basis_w_bubble, Val(:P); temperature)
    ΠA = (ΠA_, ΠA_)
    ΠP = (ΠP_, ΠP_ * 2)
    K1_A = solve_BSE.(Γ0_A, ΠA, Γ0_A, Ref(basis_w))
    K1_P = solve_BSE.(Γ0_P, ΠP, Γ0_P, Ref(basis_w))
    K1_T = su2_apply_crossing(K1_A)
    ΠAscr = ScreenedBubble.(ΠA, Γ0_A, K1_A)
    ΠPscr = ScreenedBubble.(ΠP, Γ0_P, K1_P)
    (; ΠAscr, ΠPscr, K1_A, K1_P, K1_T)
end

function run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w, basis_aux, basis_1p=G0.basis;
        max_class, max_iter=5, reltol=1e-2, temperature=nothing, smooth_bubble=false)
    F = get_formalism(G0)
    T = eltype(G0)

    Γ0_A = su2_bare_vertex(U, Val(F), Val(:A))
    Γ0_P = su2_bare_vertex(U, Val(F), Val(:P))
    Γ0_T = su2_apply_crossing(Γ0_A)

    # 1st iteration
    ΠAscr, ΠPscr, K1_A, K1_P, K1_T = setup_screened_bubble(G0, Γ0_A, Γ0_P, basis_v_bubble,
        basis_w_bubble, basis_w; temperature, smooth_bubble)
    vertex = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T, K1_A, K1_P, K1_T)

    # Update bubble
    @info "Updating self-energy and the bubble"
    @time Σ = compute_self_energy_SU2(vertex, G0, basis_1p; temperature)
    G = solve_Dyson(G0, Σ)
    ΠAscr, ΠPscr, _ = setup_screened_bubble(G, Γ0_A, Γ0_P, basis_v_bubble,
        basis_w_bubble, basis_w; temperature, smooth_bubble)

    for i in 2:max_iter
        @info "== Iteration $i =="
        @time vertex_new = iterate_parquet(vertex, ΠAscr, ΠPscr, basis_aux)

        Γ_diff = get_difference_norm(vertex_new, vertex)

        vertex = vertex_new

        @info "Updating self-energy and the bubble"
        @time Σ = compute_self_energy_SU2(vertex, G; temperature)
        G = solve_Dyson(G0, Σ)
        ΠAscr, ΠPscr, _ = setup_screened_bubble(G, Γ0_A, Γ0_P, basis_v_bubble,
            basis_w_bubble, basis_w; temperature, smooth_bubble)

        @info Γ_diff
        if Γ_diff.relerr < reltol
            @info "Convergence reached"
            break
        end
    end
    (; vertex, Σ)
end
