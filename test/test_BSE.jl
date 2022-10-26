using Test
using mfRG

@testset "BSE static" begin
    using LinearAlgebra
    using mfRG: vertex_bubble_integral

    U = 4.0
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, -4:2.:4)
    basis_w = LinearSplineAndTailBasis(1, 3, -4:4.:4)

    # Set static vertx
    Γ0 = Vertex4P{:KF, :A}(basis1, basis1, basis1)
    Γ0_kv = vertex_keldyshview(Γ0)
    for ks in CartesianIndices((2, 2, 2, 2))
        k1, k2, k3, k4 = ks.I
        if mod(k1 + k2 + k3 + k4, 2) == 1
            Γ0_kv[:, :, 1, 1, 1, 1, ks, :] .= U / 2
        end
    end

    # Set bubble. Use random numbers for testing.
    Π = Bubble{:KF}(basis2, basis_w)
    Π.data .= rand(eltype(Π.data), size(Π.data)...)

    # Solve BSE by direct inversion
    overlap = basis_integral(basis1, basis1, basis2);
    Γ_direct = Vertex4P{:KF, :A}(basis1, basis1, basis_w)
    for (iw, w) in enumerate(basis_w.grid)
        Γ0_mat = to_matrix(Γ0, w)
        Π_mat = to_matrix(Π, w, basis1, basis1)

        # Direct solution: Γ = inv(I - Γ0 * Π) * Γ0
        Γ_mat = (I - Γ0_mat * Π_mat) \ Γ0_mat
        Γ_direct.data[:, :, iw + mfRG.ntails(basis_w)] .= Γ_mat
    end

    # Solve BSE by an iterative solver
    Γ = solve_BSE(Γ0, Π, Γ0, basis_w)

    # Check that the iterative solution and the direct solution agree on the grid.
    Γ_data = to_matrix.(Ref(Γ), basis_w.grid) .+ Ref(to_matrix(Γ0, 0.));
    Γ_data_direct = to_matrix.(Ref(Γ_direct), basis_w.grid);
    @test norm(Γ_data_direct .- Γ_data) < 1e-10

    # Check that the BSE is satisfied. Check only the interpolated part, the tail part can
    # be different. The reason is that in solve_BSE the BSE is solved at each w and then
    # fitted, while in vertex_bubble_integral they are fitted after a single multiplication.
    inds_interp = mfRG.ntails(basis_w)+1:size(basis_w, 2)

    # Test Γ = Γ0 Π Γ0 + Γ0 Π Γ
    Γ0_Π_Γ0 = vertex_bubble_integral(Γ0, Π, Γ0, basis_w)
    Γ_test1 = vertex_bubble_integral(Γ0, Π, Γ, basis_w)
    Γ_test1.data .+= Γ0_Π_Γ0.data
    @test norm((Γ_test1.data .- Γ.data)[:, :, inds_interp]) < 1e-10

    # Test Γ = Γ0 Π Γ0 + Γ Π Γ0
    Γ_test2 = vertex_bubble_integral(Γ, Π, Γ0, basis_w)
    Γ_test2.data .+= Γ0_Π_Γ0.data
    @test norm((Γ_test2.data .- Γ.data)[:, :, inds_interp]) < 1e-10

    # Test Γ = Γ0 Πscr Γ0 + Γ0 Π Γ0 Π Γ0
    Πscr = mfRG.ScreenedBubble(Π, Γ)
    Γ_test3 = vertex_bubble_integral(Γ0, Πscr, Γ0, basis_w)
    Γ_test3.data .+= vertex_bubble_integral(Γ0, Π, Γ0_Π_Γ0, basis_w).data
    @test norm((Γ_test3.data .- Γ.data)[:, :, inds_interp]) < 1e-10
end
