using mfRG
using PyPlot

# Compute the vertex of SIAM by solving parquet equations by fixed point iteration.
# Obtain Fig. 9.1 of E. Walter thesis (2021)

# 2022.11.08: The script takes wall time ~ 4 mins to finish in fifi server (including compilation)
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
    t = 0.005 * Δ
    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, t)

    # Parameters to ensure ~5% error
    vgrid_1p = get_nonequidistant_grid(10, 101) .* Δ;
    vgrid_k1 = get_nonequidistant_grid(10, 21) .* Δ;
    wgrid_k1 = get_nonequidistant_grid(10, 31) .* Δ;
    vgrid_k3 = get_nonequidistant_grid(10, 25) .* Δ;

    # # Very coarse parameters for debugging
    # vgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    # wgrid_k1 = get_nonequidistant_grid(10, 5) .* Δ;
    # vgrid_k3 = get_nonequidistant_grid(10, 15) .* Δ;

    basis_v_bubble_tmp = LinearSplineAndTailBasis(2, 4, vgrid_k1)
    basis_w = LinearSplineAndTailBasis(1, 3, wgrid_k1)
    basis_v_aux = LinearSplineAndTailBasis(1, 0, vgrid_k3)

    basis_1p = LinearSplineAndTailBasis(1, 3, vgrid_1p)
    basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v_bubble_tmp, basis_w)

    # Run parquet calculation
    @time vertex, Σ, Π = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w, basis_w,
        basis_v_aux, basis_1p; max_class=3, max_iter=10, reltol=1e-2, temperature=t,
        smooth_bubble=true);
end;

begin
    # Evaluate vertex at a grid of frequencies
    vv = range(-10, 10, length=101) .* Δ
    w = 0.0
    x_k1 = zeros(ComplexF64, length(vv), length(vv), 4, 4)
    x_k2 = zeros(ComplexF64, length(vv), length(vv), 4, 4)
    x_k3 = zeros(ComplexF64, length(vv), length(vv), 4, 4)
    v1_ = vec(ones(length(vv))' .* vv)
    v2_ = vec(vv' .* ones(length(vv)))
    w_ = fill(w, length(v1_))
    @time for Γ in mfRG.get_vertices(vertex)
        Γ_dm = mfRG.su2_convert_spin_channel(:T, Γ)
        x = Γ_dm[2](v1_, v2_, w_, Val(:T))
        x = reshape(x, length(vv), length(vv), 4, 4)
        if Γ[1].basis_f1 isa ConstantBasis && Γ[1].basis_f2 isa ConstantBasis
            x_k1 .+= x
        elseif Γ[1].basis_f1 isa ConstantBasis || Γ[1].basis_f2 isa ConstantBasis
            x_k2 .+= x
        else
            x_k3 .+= x
        end
    end
    x_k1 ./= U
    x_k2 ./= U
    x_k3 ./= U
end;

begin
    fig, plotaxes = subplots(3, 5, figsize=(12, 8))
    fontsize = 16
    fig.subplots_adjust(right=0.9, hspace=0.03, wspace=0.)
    cbar_axes = [
        fig.add_axes([0.91, x().y0 + x().height * 0.15, 0.015, x().height * 0.7]) for x in getproperty.(plotaxes[:, 5], :get_position)
    ]

    for (i, x) in zip(1:3, [x_k1, x_k2, x_k3])
        vmax = (U/Δ/π)^i * [0.5, 1.0, 0.7][i]
        kwargs_plot = (; cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
            extent=[extrema(vv)..., extrema(vv)...] ./ Δ, )
        plotaxes[i, 1].imshow(real.(x[:, :, 3, 4]); kwargs_plot...)
        plotaxes[i, 2].imshow(imag.(x[:, :, 3, 4]); kwargs_plot...)
        plotaxes[i, 3].imshow(imag.(x[:, :, 1, 4]); kwargs_plot...)
        plotaxes[i, 4].imshow(real.(x[:, :, 1, 3]); kwargs_plot...)
        im = plotaxes[i, 5].imshow(imag.(x[:, :, 1, 1]); kwargs_plot...)
        cbar = fig.colorbar(im; cax=cbar_axes[i])
        cbar.ax.tick_params(labelsize=fontsize)
        plotaxes[i, 1].text(-0.5, 0.5, "\$K_$i\$"; fontsize, transform=plotaxes[i, 1].transAxes, ha="center", va="center")
    end
    plotaxes[1, 1].set_title("Re Γ_1222"; fontsize)
    plotaxes[1, 2].set_title("Im Γ_1222"; fontsize)
    plotaxes[1, 3].set_title("Im Γ_1122"; fontsize)
    plotaxes[1, 4].set_title("Re Γ_1112"; fontsize)
    plotaxes[1, 5].set_title("Im Γ_1111"; fontsize)
    for ax in plotaxes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    end
    for ax in plotaxes[3, :]
        ax.set_xlabel("\$\\nu / \\Delta\$"; fontsize)
        ax.set_xticks([-5, 0, 5])
        ax.set_xticklabels([-5, 0, 5]; fontsize)
    end
    for ax in plotaxes[:, 1]
        ax.set_ylabel("\$\\nu' / \\Delta\$"; fontsize)
        ax.set_yticks([-5, 0, 5])
        ax.set_yticklabels([-5, 0, 5]; fontsize)
    end
    suptitle("Γ_↑↓↓↑ / U, t channel, ω=$w, U/Δ=$(U/Δ), t/U=$(t/U)\n(Analogous to Fig. 9.1 (upper panel) of E. Walter thesis)"; y=0.98, fontsize)
    # savefig("siam_parquet_vertex_U_$(U/Δ).png")
    display(fig); close(fig)
end
