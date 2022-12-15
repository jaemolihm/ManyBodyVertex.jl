# TODO: Merge with run_parquet (impurity only)

function run_parquet_nonlocal(G0, U, basis_v_bubble, basis_w_bubble, rbasis,
        basis_k1_b, basis_k2_b, basis_k2_f, basis_1p=get_basis(G0);
        max_class=3, max_iter=5, reltol=1e-2, temperature=nothing,
        smooth_bubble=get_formalism(G0) === :MF ? false : true,
        mixing_history=10, mixing_coeff=0.5, iterate_by_bse=false)

    smooth_bubble && error("smooth_bubble in run_parquet_nonlocal not implementd yet")

    F = get_formalism(G0)
    T = eltype(G0)

    Γ0_A = su2_bare_vertex(Val(F), Val(:A), U, rbasis)
    Γ0_P = su2_bare_vertex(Val(F), Val(:P), U, rbasis)
    Γ0_T = su2_apply_crossing(Γ0_A)

    # Initialize self-energy and Green function. Here Σ = 0, G = G0.
    Σ = RealSpaceGreen2P{F}(eltype(G0), basis_1p.freq, basis_1p.r, G0.norb)
    @time G = green_lazy_to_explicit(G0, basis_1p)

    # 1st iteration
    @time ΠA_ = compute_bubble_nonlocal_real_space(G, basis_v_bubble, basis_w_bubble, Val(:A), rbasis; temperature);
    @time ΠP_ = compute_bubble_nonlocal_real_space(G, basis_v_bubble, basis_w_bubble, Val(:P), rbasis; temperature);
    ΠA = (ΠA_, ΠA_)
    ΠP = (ΠP_ * -1, ΠP_)

    Γ = AsymptoticVertex{F, T}(; max_class, Γ0_A, Γ0_P, Γ0_T,
        basis_k1_b=(; freq=basis_k1_b, r=rbasis),
        basis_k2_b=(; freq=basis_k2_b, r=rbasis),
        basis_k2_f=(; freq=basis_k2_f, r=rbasis))

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
        @time Σ = compute_self_energy_SU2(Γ, G, ΠA, ΠP; temperature);
        @time G = solve_Dyson(G0, Σ)
        @time ΠA_ = compute_bubble_nonlocal_real_space(G, basis_v_bubble, basis_w_bubble, Val(:A), rbasis; temperature);
        @time ΠP_ = compute_bubble_nonlocal_real_space(G, basis_v_bubble, basis_w_bubble, Val(:P), rbasis; temperature);
        ΠA = (ΠA_, ΠA_)
        ΠP = (ΠP_ * -1, ΠP_)

        @info err
        if err.relerr < reltol
            @info "Convergence reached"
            break
        end
    end
    (; Γ, Σ, Π=(; A=ΠA, P=ΠP))
end
