using mfRG
using PyPlot

# Compute the vertex of SIAM by solving parquet equations by fixed point iteration.
# Benchmark against Fig. 9.1 of E. Walter thesis (2021)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams");
rcParams["font.size"] = 15
rcParams["lines.linewidth"] = 3
rcParams["lines.markersize"] = 8

# 2022.11.08: The script (U=0.5*Δ) takes wall time ~ 4 mins to finish in fifi server
#             (including compilation)
# 2022.11.08: Wall time 2m 45s (1 thread, including compilation which takes ~50% of the time)

begin
    # 2022.11.08: run_parquet (max_class = 3) takes ~50 seconds per iteration in fifi server
    #             with the ~5% error grids. The timing is for 3rd and later iterations. The
    #             2nd iteration is very quick (~2 seconds).
    #             (Currently, multithreading is not used in run_parquet.)
    # 2022.11.30: Each iteration takes ~8 seconds with the ~5% error grids with 4 threads.
    #             parquet iteration        : ~3 seconds
    #             self-energy calculation  : ~0.5 seconds
    #             bubble update (smoothed) : ~4 seconds
    # 2022.12.21: Each iteration takes ~4 seconds with the ~5% error grids with 4 threads.
    #             parquet iteration        : ~2 seconds
    #             self-energy calculation  : ~1.1 seconds
    #             bubble update (smoothed) : ~0.6 seconds
    #             (The increase in self-energy calculation is due to changing to a more
    #             accurate implementation.)

    e = 0
    Δ = 10.0
    U = 0.5 * Δ
    # U = 2.5 * Δ
    temperature = 0.01 * U
    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, t=temperature)

    # # Parameters to ensure ~1% error (U = 2.5 * Δ takes ~ 485 sec with 4 threads)
    # vgrid_1p = get_nonequidistant_grid(10, 101) .* Δ;
    # wgrid_k1 = get_nonequidistant_grid(20, 71; w_s=10) .* Δ;
    # vgrid_k1 = get_nonequidistant_grid(20, 71; w_s=10) .* Δ;
    # vgrid_k3 = get_nonequidistant_grid(20, 51; w_s=10) .* Δ;

    # Parameters to ensure ~5% error
    vgrid_1p = get_nonequidistant_grid(10, 101) .* Δ;
    vgrid_k1 = get_nonequidistant_grid(10, 31) .* Δ;
    wgrid_k1 = get_nonequidistant_grid(10, 31) .* Δ;
    vgrid_k3 = get_nonequidistant_grid(10, 21) .* Δ;

    # # Very coarse parameters for debugging
    # vgrid_1p = get_nonequidistant_grid(10, 31) .* Δ;
    # vgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    # wgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    # vgrid_k3 = get_nonequidistant_grid(10, 5) .* Δ;

    basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
    basis_w = LinearSplineAndTailBasis(1, 3, wgrid_k1)
    basis_v_aux = LinearSplineAndTailBasis(1, 0, vgrid_k3)

    basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
    basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v_bubble_tmp, basis_w)

    # Run parquet calculation
    @time Γ, Σ, Π = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w, basis_w,
        basis_v_aux, basis_1p; max_iter=30, reltol=1e-3, temperature, mixing_history=5);
end;

begin
    # Plot self-energy as a function of energy
    vs = range(-10, 10, length=101) .* Δ
    ΣR = getindex.(Σ.(vs), 2, 1)
    ΣA = getindex.(Σ.(vs), 1, 2)
    ΣK = getindex.(Σ.(vs), 1, 1)
    ΣK_FDT = (ΣA .- ΣR) .* tanh.(vs / temperature / 2)
    plot(vs ./ Δ, real.(ΣR) ./ Δ, "-", label="Re ΣR")
    plot(vs ./ Δ, imag.(ΣR) ./ Δ, "-", label="Re ΣR")
    plot(vs ./ Δ, imag.(ΣK) ./ Δ, "-", label="Im ΣK")
    plot(vs ./ Δ, imag.(ΣK_FDT) ./ Δ, "--", label="Im ΣK_FDT")
    plot(vs ./ Δ, imag.(ΣK .- ΣK_FDT) ./ Δ .* 10, "--", label="Im (ΣK - ΣK_FDT) * 10")

    xlim(extrema(vs) ./ Δ)
    axhline(0, c="k", lw=1)
    xlabel("\$\\nu / \\Delta \$")
    ylabel("\$\\Sigma / \\Delta \$")
    legend()

    title("SIAM, KF, parquet approximation.\nU=$U, Δ=$Δ, T=$temperature")

    # savefig("siam_parquet_Keldysh_self_energy_U_$(U/Δ).png", dpi=50)
    display(gcf()); close(gcf())
end


begin
    # Plot vertex, to be compared with Fig. 9.1 of E. Walter thesis (2021).

    # Evaluate vertex at a grid of frequencies
    vs = range(-10, 10, length=101) .* Δ
    w = 0.0
    c = :T

    v1_ = vec(ones(length(vs))' .* vs)
    v2_ = vec(vs' .* ones(length(vs)))
    w_ = fill(w, length(v1_))

    function evaluate_vertex(Γ)
        Γ_dm = mfRG.su2_convert_spin_channel(c, Γ)
        x = Γ_dm[2](v1_, v2_, w_, Val(c))
        reshape(x, length(vs), length(vs), 4, 4)
    end
    x_k1_A = evaluate_vertex(Γ.K1_A)
    x_k2_A = evaluate_vertex(Γ.K2_A) .+ evaluate_vertex(Γ.K2p_A)
    x_k3_A = evaluate_vertex(Γ.K3_A)
    x_k1_P = evaluate_vertex(Γ.K1_P)
    x_k2_P = evaluate_vertex(Γ.K2_P) .+ evaluate_vertex(Γ.K2p_P)
    x_k3_P = evaluate_vertex(Γ.K3_P)
    x_k1_T = evaluate_vertex(Γ.K1_T)
    x_k2_T = evaluate_vertex(Γ.K2_T) .+ evaluate_vertex(Γ.K2p_T)
    x_k3_T = evaluate_vertex(Γ.K3_T)

    x_k1 = x_k1_A + x_k1_P + x_k1_T
    x_k2 = x_k2_A + x_k2_P + x_k2_T
    x_k3 = x_k3_A + x_k3_P + x_k3_T

    fig, plotaxes = subplots(4, 5, figsize=(12, 12))
    fig.subplots_adjust(right=0.9, hspace=0.03, wspace=0.)
    cbar_axes = [
        fig.add_axes([0.91, x().y0 + x().height * 0.15, 0.015, x().height * 0.7]) for x in getproperty.(plotaxes[:, 5], :get_position)
    ]

    for (i, x) in enumerate([x_k1, x_k2, x_k3, x_k1+x_k2+x_k3])
        if U / Δ == 0.5
            vmax = [0.0625, 0.026, 0.00288, 0.06][i]
        elseif U / Δ == 2.5
            vmax = [0.7, 0.7, 0.43, 0.735][i]
        end
        kwargs_plot = (; cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
            extent=[extrema(vs)..., extrema(vs)...] ./ Δ, )
        im = plotaxes[i, 1].imshow(real.(x[:, :, 3, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 2].imshow(imag.(x[:, :, 3, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 3].imshow(imag.(x[:, :, 1, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 4].imshow(real.(x[:, :, 1, 3]) ./ U; kwargs_plot...)
        im = plotaxes[i, 5].imshow(imag.(x[:, :, 1, 1]) ./ U; kwargs_plot...)
        cbar = fig.colorbar(im; cax=cbar_axes[i])
        cbar.ax.tick_params()
        plotaxes[i, 1].text(-0.5, 0.5, i<=3 ? "\$K_$i\$" : "\$\\Gamma\$"; transform=plotaxes[i, 1].transAxes, ha="center", va="center")
    end
    plotaxes[1, 1].set_title("Re Γ_1222")
    plotaxes[1, 2].set_title("Im Γ_1222")
    plotaxes[1, 3].set_title("Im Γ_1122")
    plotaxes[1, 4].set_title("Re Γ_1112")
    plotaxes[1, 5].set_title("Im Γ_1111")
    for ax in plotaxes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    end
    for ax in plotaxes[end, :]
        ax.set_xlabel("\$\\nu / \\Delta\$")
        ax.set_xticks([-5, 0, 5])
        ax.set_xticklabels([-5, 0, 5])
    end
    for ax in plotaxes[:, 1]
        ax.set_ylabel("\$\\nu' / \\Delta\$")
        ax.set_yticks([-5, 0, 5])
        ax.set_yticklabels([-5, 0, 5])
    end
    suptitle("SIAM, KF, parquet approximation. Full vertex Γ_↑↓↓↑ in t-channel representation.\n"
        * "ω=$w, U=$U, Δ=$Δ, T=$temperature";)
    # savefig("siam_parquet_Keldysh_vertex_U_$(U/Δ).png", dpi=50)
    display(fig); close(fig)
end

begin
    # Plot vertex, channel decomposed.

    fig, plotaxes = subplots(9, 5, figsize=(10, 18))
    fig.subplots_adjust(right=0.9, hspace=0.03, wspace=0.)
    cbar_axes = [
        fig.add_axes([0.91, x().y0 + x().height * 0.15, 0.015, x().height * 0.7]) for x in getproperty.(plotaxes[:, 5], :get_position)
    ]

    for (i, x) in enumerate([x_k1_A, x_k2_A, x_k3_A, x_k1_P, x_k2_P, x_k3_P, x_k1_T, x_k2_T, x_k3_T])
        vmax = max(maximum(abs.(real.(x))), maximum(abs.(imag.(x)))) / U
        kwargs_plot = (; cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
            extent=[extrema(vs)..., extrema(vs)...] ./ Δ, )
        im = plotaxes[i, 1].imshow(real.(x[:, :, 3, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 2].imshow(imag.(x[:, :, 3, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 3].imshow(imag.(x[:, :, 1, 4]) ./ U; kwargs_plot...)
        im = plotaxes[i, 4].imshow(real.(x[:, :, 1, 3]) ./ U; kwargs_plot...)
        im = plotaxes[i, 5].imshow(imag.(x[:, :, 1, 1]) ./ U; kwargs_plot...)
        cbar = fig.colorbar(im; cax=cbar_axes[i])
        cbar.ax.tick_params()
        labeltext = "\$K_$(mod1(i,3))^$(["a","p","t"][cld(i,3)])\$"
        plotaxes[i, 1].text(-0.5, 0.5, labeltext; transform=plotaxes[i, 1].transAxes, ha="center", va="center")
    end
    plotaxes[1, 1].set_title("Re Γ_1222")
    plotaxes[1, 2].set_title("Im Γ_1222")
    plotaxes[1, 3].set_title("Im Γ_1122")
    plotaxes[1, 4].set_title("Re Γ_1112")
    plotaxes[1, 5].set_title("Im Γ_1111")
    for ax in plotaxes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    end
    for ax in plotaxes[end, :]
        ax.set_xlabel("\$\\nu / \\Delta\$")
        ax.set_xticks([-5, 0, 5])
        ax.set_xticklabels([-5, 0, 5])
    end
    for ax in plotaxes[:, 1]
        ax.set_ylabel("\$\\nu' / \\Delta\$")
        ax.set_yticks([-5, 0, 5])
        ax.set_yticklabels([-5, 0, 5])
    end
    suptitle("SIAM, KF, parquet approximation. Full vertex Γ_↑↓↓↑ in t-channel representation.\n"
        * "ω=$w, U=$U, Δ=$Δ, T=$temperature"; y=0.94)
    display(fig); close(fig)
end
