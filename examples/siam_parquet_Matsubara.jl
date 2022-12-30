using mfRG
using PyPlot
using Printf

# Compute the vertex of SIAM by solving parquet equations by fixed point iteration.
# Benchmark against P. Chalupa-Gantner et al, PRResearch 4, 023050 (2022)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams");
rcParams["font.size"] = 15
rcParams["lines.linewidth"] = 3
rcParams["lines.markersize"] = 8

# 2022.12.29: The script with nmax = 24 takes wall time ~ 4 mins to finish in fifi server
#             (including compilation, 1 thread)

begin
    # 2022.12.29: Each iteration takes ~7 seconds with nmax = 24 with 4 threads.
    #             parquet iteration        : ~4.5 seconds
    #             self-energy calculation  : ~2.0 seconds
    #             bubble update            : ~0.1 seconds

    # Parameters from the reference paper.
    e = 0
    Δ = π/5
    temperature = 0.1
    U = 1.0
    D = 10
    G0 = SIAMLazyGreen2P{:MF}(; e, Δ, temperature, D)

    # Control parameter for the basis set size
    # nmax = 12  # runtime ~6 seconds, error below 1 %
    nmax = 24  # runtime ~46 seconds, error below 0.03 %

    # Set frequency basis functions
    basis_w_k1 = ImagGridAndTailBasis(:Boson, 1, 0, 4 * nmax)
    basis_w = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_v_aux = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
    basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_w_k1)))
    basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_w_k1)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)

    # Run parquet calculation
    @time Γ, Σ, Π = run_parquet(G0, U, basis_v_bubble, basis_w_bubble, basis_w_k1, basis_w, basis_v_aux,
        basis_1p; max_iter=15, reltol=1e-2, temperature);
end;


begin
    # Plot self-energy
    ns = -21:20
    vs = @. 2π * temperature * (ns + 1/2)
    plot(vs, imag.(getindex.(Σ.(ns), 1, 1)) ./ U^2, "o-", label="This code, nmax=$nmax")

    # Data from reference paper (Fig. 3, first panel)
    Σ_Chalupa = [-0.038312, -0.051846, -0.052369, -0.049564, -0.046030, -0.042527, -0.039310, -0.036442]
    v_Chalupa = 2π * temperature * ((0:(length(Σ_Chalupa)-1)) .+ 1/2)
    Σ_Chalupa = vcat(.-reverse(Σ_Chalupa), Σ_Chalupa)
    v_Chalupa = vcat(.-reverse(v_Chalupa), v_Chalupa)
    plot(v_Chalupa, Σ_Chalupa, "*--", label="Chalupa-Gantner (2022)")

    xlim([0, 5])
    ylim([-0.056, -0.035])
    xlabel("\$\\nu\$")
    ylabel("Im \$\\Sigma / U^2\$")
    legend(loc="center right")
    title(@sprintf "SIAM, MF, parquet approximation\nΔ=%.2f, U=%.2f, T=%.2f" Δ U temperature)
    tight_layout()
    # savefig("siam_parquet_Matsubara_self_energy.png")
    display(gcf()); close(gcf())
end

begin
    # Plot vertex
    # To be compared with Fig. 3 (third panel) of the reference paper.

    # Evaluate vertex at a grid of frequencies
    w = 0
    vs = -15:14
    c = :A

    v1_ = vec(ones(length(vs))' .* vs)
    v2_ = vec(vs' .* ones(length(vs)))
    w_ = fill(w, length(v1_))
    function evaluate_vertex(Γ)
        Γ_dm = mfRG.su2_convert_spin_channel(c, Γ)
        x = Γ_dm[2](v1_, v2_, w_, Val(c))
        reshape(x, length(vs), length(vs))
    end
    x_k1 = evaluate_vertex(Γ.K1_A)
    x_k2 = evaluate_vertex(Γ.K2_A) + evaluate_vertex(Γ.K2p_A)
    x_k3 = evaluate_vertex(Γ.K3_A)

    # Plot
    fig, plotaxes = subplots(2, 2, figsize=(10, 8))
    for (i, x) in enumerate([x_k1, x_k2, x_k3, x_k1 + x_k2 + x_k3])
        if i <= 3
            cmap = "RdBu_r"
            vmin, vmax = -maximum(abs.(x)), maximum(abs.(x))
        else
            cmap = "plasma_r"
            vmin, vmax = -0.763, -0.533
        end
        kwargs_plot = (; cmap, origin="lower", vmin, vmax, extent=2π * temperature .* [minimum(vs), maximum(vs)+1, minimum(vs), maximum(vs)+1],)

        ax = plotaxes[i]
        im = ax.imshow(real.(x[:, :]); kwargs_plot...)
        colorbar(im; ax)
        ax.set_title(["Re K_{1a}", "Re K_{2a} + K_{2'a}", "Re K_{3a}", "Re γ_a"][i])
    end
    for ax in plotaxes
        ax.set_xlim([-6.5, 6.5])
        ax.set_ylim([-6.5, 6.5])
        ax.set_xticks(-6:3:6)
        ax.set_yticks(-6:3:6)
        ax.set_xlabel("\$\\nu / \\Delta\$")
        ax.set_ylabel("\$\\nu' / \\Delta\$")
    end
    suptitle("SIAM, MF, parquet approximation. a-reducible vertex in m spin channel\n"
            * @sprintf "ω=%.2f, Δ=%.2f U=%.2f, T=%.2f, nmax=%d" w Δ U temperature nmax)
    tight_layout()
    # savefig("siam_parquet_Matsubara_vertex.png")
    display(fig); close(fig)
end
