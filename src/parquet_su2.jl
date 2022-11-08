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

function run_parquet(U, ΠA, ΠP, basis_w, basis_aux; max_class, max_iter=5)
    Γ0_A = su2_bare_vertex(U, Val(:KF), Val(:A))
    Γ0_P = su2_bare_vertex(U, Val(:KF), Val(:P))
    Γ0_T = su2_apply_crossing(Γ0_A)

    K1_A = solve_BSE.(Γ0_A, ΠA, Γ0_A, Ref(basis_w))
    K1_P = solve_BSE.(Γ0_P, ΠP, Γ0_P, Ref(basis_w))
    K1_T = su2_apply_crossing(K1_A)
    ΠAscr = ScreenedBubble.(ΠA, K1_A)
    ΠPscr = ScreenedBubble.(ΠP, K1_P)

    vertex = AsymptoticVertex{:KF, eltype(K1_A)}(; max_class, Γ0_A, Γ0_P, Γ0_T, K1_A, K1_P, K1_T)

    for i in 2:max_iter
        @info "Iteration $i"
        @time vertex_new = iterate_parquet(vertex, ΠAscr, ΠPscr, basis_aux)

        Γ_diff_norm = get_difference_norm(vertex_new, vertex)

        vertex = vertex_new

        @info Γ_diff_norm
        if maximum(Γ_diff_norm) < U * 1e-2
            @info "Convergence reached"
            break
        end
    end
    vertex
end
