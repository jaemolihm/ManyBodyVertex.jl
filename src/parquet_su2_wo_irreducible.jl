function iterate_parquet_asymptotic_single_channel_without_fully_irreducible(
        Π₀, Π, U₀, γ₀, I₀, Δγ, ΔI; max_class=3, basis_k1_b, basis_k2_f, basis_k2_b)

    C = get_channel(Π₀[1])
    ΔΠ = Π .- Π₀

    ws_12 = unique!(vcat(get_fitting_points(basis_k1_b.freq), get_fitting_points(basis_k2_b.freq)))
    ws_2 = get_fitting_points(basis_k2_b.freq)
    ΔI_mat = Tuple(cache_vertex_matrix(getindex.(ΔI, i), C, ws_12, basis_k2_f) for i in 1:2)
    I₀_mat = Tuple(cache_vertex_matrix(getindex.(I₀, i), C, ws_2, basis_k2_f) for i in 1:2)

    Kᴸ₀ = filter!(!isnothing, [U₀, γ₀.K1, γ₀.K2p])
    Kᴿ₀ = filter!(!isnothing, [U₀, γ₀.K1, γ₀.K2])
    Kᴸ = filter!(!isnothing, [U₀, γ₀.K1, γ₀.K2p, Δγ.K1, Δγ.K2p])
    Kᴿ = filter!(!isnothing, [U₀, γ₀.K1, γ₀.K2, Δγ.K1, Δγ.K2])

    # Average two equations
    # ΔK₁ = Kᴸ₀ ΔΠ Kᴿ  + Kᴸ₀ Π₀ ΔI Π  Kᴿ
    #     = Kᴸ  ΔΠ Kᴿ₀ + Kᴸ  Π  ΔI Π₀ Kᴿ₀
    ΔK1_new = (_mapreduce_bubble_integrals(Kᴸ₀, ΔΠ, Kᴿ , basis_k1_b) .+
               _mapreduce_bubble_integrals(Kᴸ , ΔΠ, Kᴿ₀, basis_k1_b)) ./ 2
    if !isempty(ΔI)
        tmpL = _mapreduce_bubble_integrals(Kᴸ₀, Π₀, [ΔI_mat], basis_k1_b)
        tmpR = _mapreduce_bubble_integrals([ΔI_mat], Π₀, Kᴿ₀, basis_k1_b)
        ΔK1_new = ΔK1_new .+ (_mapreduce_bubble_integrals([tmpL], Π, Kᴿ, basis_k1_b) .+
                              _mapreduce_bubble_integrals(Kᴸ, Π, [tmpR], basis_k1_b)) ./ 2
    end

    # ΔK₂ = Γᴸ₀ ΔΠ Kᴿ  + ΔI Π  Kᴿ  + Γᴸ₀ Π₀ ΔI Π  Kᴿ      <-- Use this formula
    #     = Γᴸ  ΔΠ Kᴿ₀ + ΔI Π₀ Kᴿ₀ + Γᴸ  Π  ΔI Π₀ Kᴿ₀

    # ΔK₂ = Kᴸ₀ ΔΠ Γᴿ  + Kᴸ₀ Π₀ ΔI + Kᴸ₀ Π₀ ΔI Π  Γᴿ
    #     = Kᴸ  ΔΠ Γᴿ₀ + Kᴸ  Π  ΔI + Kᴸ  Π  ΔI Π₀ Γᴿ₀     <-- Use this formula

    if max_class >= 2
        core₀ = Tuple(cache_vertex_matrix(getindex.(filter!(!isnothing,
                [γ₀.K3, I₀_mat]), i), C, ws_2, basis_k2_f) for i in 1:2)
        Γᴸ₀ = [γ₀.K2, core₀]
        Γᴿ₀ = [γ₀.K2p, core₀]

        ΔK2_new = _mapreduce_bubble_integrals(Γᴸ₀, ΔΠ, Kᴿ, basis_k2_b)
        ΔK2p_new = _mapreduce_bubble_integrals(Kᴸ, ΔΠ, Γᴿ₀, basis_k2_b)

        if !isempty(ΔI)
            tmpL₀ = _mapreduce_bubble_integrals(Γᴸ₀, Π₀, [ΔI_mat], basis_k2_b)
            tmpR₀ = _mapreduce_bubble_integrals([ΔI_mat], Π₀, Γᴿ₀, basis_k2_b)

            ΔK2_new = ΔK2_new .+ _mapreduce_bubble_integrals([tmpL₀, ΔI_mat], Π, Kᴿ, basis_k2_b)
            ΔK2p_new = ΔK2p_new .+ _mapreduce_bubble_integrals(Kᴸ, Π, [ΔI_mat, tmpR₀], basis_k2_b)
        end
    else
        ΔK2_new = nothing
        ΔK2p_new = nothing
    end

    # Average two equations
    # ΔK₃ = Γᴸ₀ ΔΠ Γᴿ  + ΔI Π  Γᴿ  + Γᴸ₀ Π₀ ΔI Π  Γᴿ  + Γᴸ₀ Π₀ ΔI
    #     + Γᴸ  ΔΠ Γᴿ₀ + ΔI Π₀ Γᴿ₀ + Γᴸ  Π  ΔI Π₀ Γᴿ₀ + Γᴸ  Π  ΔI
    if max_class >= 3
        core = Tuple(cache_vertex_matrix(getindex.(filter!(!isnothing,
                [γ₀.K3, I₀_mat, Δγ.K3, ΔI...]), i), C, ws_2, basis_k2_f) for i in 1:2)
        Γᴸ = [γ₀.K2 , Δγ.K2 , core]
        Γᴿ = [γ₀.K2p, Δγ.K2p, core]

        ΔK3_new = (_mapreduce_bubble_integrals(Γᴸ₀, ΔΠ, Γᴿ , basis_k2_b) .+
                   _mapreduce_bubble_integrals(Γᴸ , ΔΠ, Γᴿ₀, basis_k2_b))
        if !isempty(ΔI)
            tmpL  = _mapreduce_bubble_integrals(Γᴸ, Π , [ΔI_mat], basis_k2_b)
            tmpR  = _mapreduce_bubble_integrals([ΔI_mat], Π , Γᴿ, basis_k2_b)
            ΔK3_new = ΔK3_new .+ (tmpL .+ tmpL₀ .+ tmpR .+ tmpR₀ .+
                _mapreduce_bubble_integrals([tmpL₀], Π, Γᴿ, basis_k2_b) .+
                _mapreduce_bubble_integrals(Γᴸ, Π, [tmpR₀], basis_k2_b))
        end
        ΔK3_new = ΔK3_new ./ 2
    else
        ΔK3_new = nothing
    end

    (; K1=ΔK1_new, K2=ΔK2_new, K2p=ΔK2p_new, K3=ΔK3_new)
end

function iterate_parquet_without_irreducible(ΔΓ, ΠA::NTuple{2,BT}, ΠP, Γ₀, ΠA₀, ΠP₀) where {BT <: Bubble}
    # BSE for channel A
    @info "Solving BSE for channel A"
    @time ΔγA = iterate_parquet_asymptotic_single_channel_without_fully_irreducible(
        ΠA₀, ΠA, Γ₀.Γ0_A, get_reducible_vertices(:A, Γ₀), get_irreducible_vertices(:A, Γ₀),
        get_reducible_vertices(:A, ΔΓ), get_irreducible_vertices(:A, ΔΓ);
        ΔΓ.max_class, ΔΓ.basis_k1_b, ΔΓ.basis_k2_f, ΔΓ.basis_k2_b
    )

    # BSE for channel P
    @info "Solving BSE for channel P"
    @time ΔγP = iterate_parquet_asymptotic_single_channel_without_fully_irreducible(
        ΠP₀, ΠP, Γ₀.Γ0_P, get_reducible_vertices(:P, Γ₀), get_irreducible_vertices(:P, Γ₀),
        get_reducible_vertices(:P, ΔΓ), get_irreducible_vertices(:P, ΔΓ);
        ΔΓ.max_class, ΔΓ.basis_k1_b, ΔΓ.basis_k2_f, ΔΓ.basis_k2_b
    )

    # Channel T: use crossing symmetry
    @info "Applying crossing symmetry for channel T"
    ΔγT = map(su2_apply_crossing, ΔγA)

    ΔΓs = (; ΔΓ.Γ0_A, ΔΓ.Γ0_P, ΔΓ.Γ0_T, K1_A=ΔγA.K1, K1_P=ΔγP.K1, K1_T=ΔγT.K1)
    if ΔΓ.max_class >= 2
        ΔΓs = (; ΔΓs..., K2_A=ΔγA.K2, K2_P=ΔγP.K2, K2_T=ΔγT.K2, K2p_A=ΔγA.K2p, K2p_P=ΔγP.K2p, K2p_T=ΔγT.K2p)
    end
    if ΔΓ.max_class >= 3
        ΔΓs = (; ΔΓs..., K3_A=ΔγA.K3, K3_P=ΔγP.K3, K3_T=ΔγT.K3)
    end
    typeof(ΔΓ)(; ΔΓ.max_class, ΔΓ.basis_k1_b, ΔΓ.basis_k2_b, ΔΓ.basis_k2_f, ΔΓs...)
end



function run_parquet_without_irreducible(G0, Π₀, Γ₀, basis_1p=G0.basis;
        max_class=3, max_iter=5, reltol=1e-2, temperature=nothing,
        smooth_bubble=get_formalism(G0) === :MF ? false : true,
        mixing_history=10, mixing_coeff=0.5)
    F = get_formalism(G0)
    T = eltype(G0)

    ΠA₀, ΠP₀ = Π₀
    (; Γ0_A, Γ0_P, Γ0_T, basis_k1_b, basis_k2_b, basis_k2_f) = Γ₀
    basis_w_bubble = ΠA₀[1].basis_b
    basis_v_bubble = ΠA₀[1].basis_f

    # 1st iteration
    ΔΓ = AsymptoticVertex{F, T}(; max_class, Γ0_A=Γ0_A, Γ0_P=Γ0_P, Γ0_T=Γ0_T, basis_k1_b, basis_k2_b, basis_k2_f)

    # Initialize self-energy and Green function.
    Σ₀ = compute_self_energy_SU2(Γ₀, G0, ΠA₀, ΠP₀, (; freq=basis_1p); temperature)
    ΔΣ = similar(Σ₀)
    ΔΣ.data .= 0
    G = solve_Dyson(G0, Σ₀ + ΔΣ)
    ΠA, ΠP = setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)

    acceleration = AndersonAcceleration(; m=mixing_history)

    for i in 1:max_iter
        @info "== Iteration $i =="
        @time ΔΓ_new = iterate_parquet_without_irreducible(ΔΓ, ΠA, ΠP, Γ₀, ΠA₀, ΠP₀)

        err = get_difference_norm(ΔΓ_new, ΔΓ)

        # Mixing
        if i <= 2
            # For the first two iterations, new class of vertices may appear, so we do not
            # apply mixing.
            ΔΓ = ΔΓ_new
        else
            ΔΓ_vec = vertex_to_vector(ΔΓ)
            ΔΓ_diff_vec = vertex_to_vector(ΔΓ_new) .- ΔΓ_vec
            ΔΓ = vector_to_vertex(acceleration(ΔΓ_vec, mixing_coeff, ΔΓ_diff_vec), ΔΓ)
        end

        @info "Updating self-energy and the bubble"
        @time Σ₀ = compute_self_energy_SU2(Γ₀, G, ΠA, ΠP; temperature)
        @time ΔΣ = compute_self_energy_SU2(ΔΓ, G, ΠA, ΠP; temperature, exclude_UU=true)
        G = solve_Dyson(G0, Σ₀ + ΔΣ)
        ΠA, ΠP = setup_bubble_SU2(G, basis_v_bubble, basis_w_bubble; temperature, smooth_bubble)

        @info err
        if err.relerr < reltol
            @info "Convergence reached"
            break
        end
    end
    Σ = Σ₀ + ΔΣ
    (; ΔΓ, Σ)
end
