# using mfRG
# using Test
# using PyPlot
# using Printf
# using JLD2
using mfRG
using PyPlot
using StaticArrays
using Printf
using LinearAlgebra

# Compute the vertex of the Hubbard model using the parquet approximation.
# Benchmark against C. Hille et al, PRResearch 2, 033372 (2020).

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams");
rcParams["font.size"] = 14
rcParams["lines.linewidth"] = 3
rcParams["lines.markersize"] = 8

begin
    # Setup model and solve parquet equation.
    # For a rough convergence, use nmax = 12 and qgrid = (8, 8) (~ 6 mins with 4 threads)
    # For a good convergence, use nmax = 16 and qgrid = (16, 16) (~ 80 mins with 4 threads)
    # 2022.12.29: with nmax = 12, qgrid = (8, 8), 4 threads, ~30 seconds per iteration.
    #             parquet update  : 14 sec
    #             Anderson mixing : 10 sec
    #             self-energy     :  5 sec
    #             bubble update   :  2 sec

    nmax = 12
    qgrid = (8, 8)
    # nmax = 16
    # qgrid = (16, 16)

    basis_k1_b = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_k2_b = ImagGridAndTailBasis(:Boson, 1, 0, 2 * nmax)
    basis_k2_f = ImagGridAndTailBasis(:Fermion, 1, 0, nmax)
    basis_w_bubble = ImagGridAndTailBasis(:Boson, 1, 0, maximum(get_fitting_points(basis_k1_b)))
    basis_v_bubble = ImagGridAndTailBasis(:Fermion, 2, 4, maximum(get_fitting_points(basis_k1_b)))
    basis_1p = ImagGridAndTailBasis(:Fermion, 1, 3, nmax * 3 + 10)

    t = 1.
    μ = 0.
    U = 2.0
    temperature = 0.2
    G0 = mfRG.HubbardLazyGreen2P{:MF}(; temperature, t, μ)

    lattice = SMatrix{2, 2}([1. 0; 0 1])
    positions = [SVector(0., 0.)]
    bonds_L = [(1, 1, SVector(0, 0))]
    bonds_R = [(1, 1, SVector(0, 0))]
    qgrid_1p = (32, 32)
    rbasis = RealSpaceBasis(lattice, positions, bonds_L, bonds_R, qgrid)
    rbasis_1p = RealSpaceBasis2P(lattice, positions, qgrid_1p)

    @time Γ, Σ, Π = run_parquet_nonlocal(G0, U, basis_v_bubble, basis_w_bubble, rbasis,
        basis_k1_b, basis_k2_b, basis_k2_f, (; freq=basis_1p, r=rbasis_1p);
        max_iter=30, reltol=1e-4, temperature, mixing_history=5)

    op_suscep_L, op_suscep_R = mfRG.susceptibility_operator_SU2(Val(:MF), rbasis)
    @time chi = mfRG.compute_response_SU2(op_suscep_L, op_suscep_R, Γ, Π.A);
end;


begin
    # Plot self-energy
    fig, plotaxes = subplots(1, 2, figsize=(12, 5.5))

    # Left panel: self-energy as a function of momentum
    sca(plotaxes[1])
    xqs = vcat([SVector(x, 0.) for x in range(0., 0.5, length=31)[1:end-1]],
               [SVector(0.5, x) for x in range(0., 0.5, length=31)[1:end-1]],
               [SVector(x, x) for x in range(0.5, 0., length=31)])
    qs = vcat(0, cumsum(norm.(xqs[2:end] .- xqs[1:end-1])))

    nmax_plot = 12
    yy = map(xqs) do xq
        interpolate_to_q(Σ, xq, 1, 1)(0)[1, 1]
    end
    plot(qs, real.(yy); label="Re Σ, this code")
    plot(qs, imag.(yy); label="Im Σ, this code")

    # Data from Fig. 15 of C. Hille et al, PRResearch 2, 033372 (2020)
    xx = [0.00043764, 0.037637, 0.073961, 0.11160, 0.14748, 0.18468, 0.22013, 0.25821, 0.29365, 0.33085, 0.36718, 0.40438, 0.44070, 0.47746, 0.51422, 0.54967, 0.58643, 0.63851, 0.69015, 0.74136, 0.79344, 0.84639, 0.89759, 0.94923, 1.0000]
    yy_r = [-0.040397, -0.039934, -0.038612, -0.035702, -0.031207, -0.024198, -0.014744, -0.0046281, -0.000066116, 0.0045620, 0.014545, 0.024132, 0.031207, 0.035504, 0.038215, 0.039934, 0.040198, 0.039273, 0.035438, 0.025124, -0.00019835, -0.025322, -0.035570, -0.039405, -0.040264]
    yy_i = [-0.057117, -0.057941, -0.060549, -0.064989, -0.071991, -0.081648, -0.093181, -0.10307, -0.10691, -0.10307, -0.093135, -0.081648, -0.071991, -0.064943, -0.060458, -0.057941, -0.057025, -0.058719, -0.064165, -0.074691, -0.083295, -0.074645, -0.064165, -0.058719, -0.057117]
    plot(qs[end] .* xx, yy_r, "k--", label="Re Σ, Reference")
    plot(qs[end] .* xx, yy_i, "k-.", label="Im Σ, Reference")

    inds_highsym = findall(x -> x ∈ [SVector(0., 0.), SVector(0.5, 0.), SVector(0.5, 0.5)], xqs)
    for i in inds_highsym
        axvline(qs[i], c="k", lw=1)
    end
    xticks(qs[inds_highsym], ["Γ", "X", "M", "Γ"])
    ylabel("\$ \\Sigma (i\\nu = i\\pi T) \$")
    legend()
    axhline(0, c="k", lw=1)
    xlim(extrema(qs))

    # Right panel: self-energy as a function of energy
    sca(plotaxes[2])
    vv = 0:14
    xx = @. 2π * temperature * (vv + 1/2)
    for (i, xq) in enumerate([SVector(0.25, 0.25), SVector(0.5, 0.0)])
        yy = getindex.((interpolate_to_q(Σ, xq, 1, 1)).(vv), 1, 1)
        label = if xq == SVector(0.25, 0.25)
            "q=(π/2, π/2), this code"
        elseif xq == SVector(0.5, 0.)
            "q=(π, 0), this code"
        else
            nothing
        end
        plot(xx, imag.(yy), "o-"; label, marker=["o", "s"][i])
    end

    # Data from Fig. 14 of C. Hille et al, PRResearch 2, 033372 (2020) (Parquet)
    yy1 = [-0.083656, -0.11600, -0.12270, -0.11861, -0.11052, -0.10225, -0.093881, -0.086538, -0.079287, -0.073524, -0.067761, -0.063393, -0.059210, -0.055306, -0.052239]
    yy2 = [-0.10732, -0.12396, -0.12495, -0.11931, -0.11069, -0.10216, -0.093939, -0.086408, -0.079571, -0.073724, -0.068274, -0.063220, -0.059554, -0.055888, -0.052122]
    plot(xx, yy1, "kx--", label="q=(π/2, π/2), Ref.")
    plot(xx, yy2, "k+--", label="q=(π, 0), Ref.")

    xlabel("\$\\nu\$")
    ylabel("Im Σ")
    legend()
    xlim(left=0)

    fig.suptitle("Hubbard model, MF, parquet approximation.\n"
        * "Ref: C. Hille et al, PRResearch 2, 033372 (2020), Figs. 14,15\n"
        * "t=$t, T=$temperature, U=$U, μ=$μ, nmax=$nmax, qgrid=$qgrid")
    tight_layout()
    # fig.savefig("hubbard_parquet_Matsubara_self_energy.png", bbox_inches="tight")
    display(fig); close(fig)
end

begin
    # Evaluate vertex at a grid of frequencies
    b0 = (1, 1, SVector(0, 0))
    vv = -4:3
    w = 0
    qplotlist = Dict(:A => SVector(0.5, 0.5), :P => SVector(0., 0.), :T => SVector(0.5, 0.5))

    v1_ = vec(ones(length(vv))' .* vv)
    v2_ = vec(vv' .* ones(length(vv)))
    w_ = fill(w, length(v1_))

    xdata = Dict(c => zeros(ComplexF64, length(vv), length(vv)) for c in (:A, :P, :T))
    for Γ_ in mfRG.get_vertices(Γ)
        C = get_channel(Γ_[1])
        Γ_dm = mfRG.su2_convert_spin_channel(:A, Γ_)
        Γ_dm_q = mfRG.interpolate_to_q(Γ_dm[2], qplotlist[c], b0, b0)
        xdata[C] .+= reshape(Γ_dm_q(v1_, v2_, w_, C), length(vv), length(vv))
    end

    fig, plotaxes = subplots(1, 3, figsize=(12, 4))
    for (i, c) in zip(1:3, (:P, :T, :A))
        x = xdata[c]
        cmap = "viridis"
        vmin, vmax = [(1., 1.7), (-2.8, -1.0), (-6.6, -3.7)][i]
        kwargs_plot = (; cmap, origin="lower", vmin, vmax, extent=2π * temperature .* [minimum(vv), maximum(vv)+1, minimum(vv), maximum(vv)+1],)

        im = plotaxes[i, 1].imshow(real.(x[:, :]); kwargs_plot...)
        plotaxes[i, 1].set_title("Re γ_$c, q=$(qplotlist[c])")
        cbar = fig.colorbar(im; ax=plotaxes[i, 1])
    end
    for ax in plotaxes
        ax.set_xlabel("\$\\nu\$")
        ax.set_ylabel("\$\\nu'\$")
    end
    fig.suptitle("Hubbard model, MF, parquet approximation.\nChannel-reducible vertices in the magnetic spin channel.\n"
    * "Ref: C. Hille et al, PRResearch 2, 033372 (2020), Fig. 1\n"
    * "t=$t, T=$temperature, U=$U, μ=$μ, nmax=$nmax, qgrid=$qgrid")
    tight_layout()
    # savefig("hubbard_parquet_Matsubara_vertex.png", dpi=150, bbox_inches="tight")
    display(fig); close(fig)
end


# Susceptibility
# chi[1]: charge susceptibility
# chi[2]: magnetic susceptibility

begin
    # Plot susceptibility
    fig, plotaxes = subplots(1, 2, figsize=(12, 5.5))

    # Left panel: susceptibility as a function of momentum
    sca(plotaxes[1])
    xqs = vcat([SVector(x, 0.) for x in range(0., 0.5, length=31)[1:end-1]],
               [SVector(0.5, x) for x in range(0., 0.5, length=31)[1:end-1]],
               [SVector(x, x) for x in range(0.5, 0., length=31)])
    qs = vcat(0, cumsum(norm.(xqs[2:end] .- xqs[1:end-1])))
    labels = ["magnetic, full", "charge, full", "magnetic, disc.", "charge, disc."]
    for (i, x) in enumerate([chi.total[2], chi.total[1], chi.disconnected[2], chi.disconnected[1]])
        yy = map(xqs) do xq
            mfRG.interpolate_to_q(x, xq, b0, b0).(0)[1, 1]
        end
        plot(qs, real.(yy), ls=i ∈ [1, 3] ? "-" : "--", label=labels[i])
    end

    # Data from Fig. 9 of C. Hille et al, PRResearch 2, 033372 (2020) (Parquet)
    # The susceptibility of C. Hille et al has an additional factor of 1/2 in the definition.
    # Here, we multiply 2 to the reference data to recover our convention.
    # See Eq. (3a) of Tagliavini et al, Scipost Phys. 6, 009 (2019)
    xx = [-4.7184e-16, 0.018785, 0.091160, 0.055249, 0.12928, 0.16575, 0.20166, 0.23702, 0.27569, 0.31160, 0.34807, 0.38343, 0.42044, 0.45856, 0.49448, 0.53039, 0.54972, 0.56685, 0.58619, 0.61105, 0.63702, 0.66575, 0.69006, 0.71602, 0.76796, 0.81934, 0.87072, 0.92210, 0.97569, 1.0000]
    yy = [0.16449, 0.16510, 0.16753, 0.16449, 0.16024, 0.15235, 0.14203, 0.13536, 0.13171, 0.13293, 0.13961, 0.15235, 0.17542, 0.21548, 0.29256, 0.45341, 0.58938, 0.74416, 0.82185, 0.68953, 0.51593, 0.40728, 0.33869, 0.29256, 0.23612, 0.20212, 0.18149, 0.16813, 0.16510, 0.16267]
    xx .*= qs[end] / xx[end]
    plot(xx, yy .* 2, "k-.", label="magnetic, full, Ref.")

    inds_highsym = findall(x -> x ∈ [SVector(0., 0.), SVector(0.5, 0.), SVector(0.5, 0.5)], xqs)
    axvline.(qs[inds_highsym], c="k", lw=1)
    xlim(extrema(qs))
    xticks(qs[inds_highsym], ["Γ", "X", "M", "Γ"])
    ylabel("\$ \\chi_M (i\\omega = 0) \$")
    axhline(0, c="k", lw=1)
    legend()

    # Right panel: susceptibility as a function of frequency
    sca(plotaxes[2])
    ww = -5:5
    xx = @. ww * 2π * temperature

    z = interpolate_to_q(chi.total[2], SVector(0.5, 0.5), b0, b0)
    plot(xx, real.(getindex.(z.(ww), 1, 1)), "o-", label="Full, this code")

    z = interpolate_to_q(chi.disconnected[2], SVector(0.5, 0.5), b0, b0)
    plot(xx, real.(getindex.(z.(ww), 1, 1)), "o-", label="Disconnected, this code")

    # Data from Fig. 10 of C. Hille et al, PRResearch 2, 033372 (2020) (Parquet)
    # The susceptibility of C. Hille et al has an additional factor of 1/2 in the definition.
    # Here, we multiply 2 to the reference data to recover our convention.
    # See Eq. (3a) of Tagliavini et al, Scipost Phys. 6, 009 (2019)
    yy = [0.025699, 0.036477, 0.053057, 0.083731, 0.16166, 0.81907, 0.16166, 0.083731, 0.052228, 0.036477, 0.024870]
    plot(xx, yy .* 2, "kx--", label="Reference")

    legend()
    ylabel("\$\\chi\$_AF (q=(0.5, 0.5))")
    xlabel("\$\\omega\$")

    fig.suptitle("Hubbard model, MF, parquet approximation.\n"
        * "Ref: C. Hille et al, PRResearch 2, 033372 (2020), Figs. 9,10\n"
        * "t=$t, T=$temperature, U=$U, μ=$μ, nmax=$nmax, qgrid=$qgrid")
    tight_layout()
    # fig.savefig("hubbard_parquet_Matsubara_susceptibility.png", bbox_inches="tight")
    display(fig); close(fig)
end

