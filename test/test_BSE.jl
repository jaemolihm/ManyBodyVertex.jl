using Test
using mfRG

@testset "BSE static" begin
    using LinearAlgebra

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
        Γ0_mat = vertex_to_matrix(Γ0, w)
        Π_mat = bubble_to_matrix(Π, w, overlap)

        # Direct solution: Γ = inv(I - Γ0 * Π) * Γ0
        Γ_mat = (I - Γ0_mat * Π_mat) \ Γ0_mat
        Γ_direct.data[:, :, iw + mfRG.ntails(basis_w)] .= Γ_mat
    end

    # Solve BSE by an iterative solver
    Γ_iter = solve_BSE(Γ0, Π, Γ0, basis_w)

    # Check that the iterative solution and the direct solution agree on the grid.
    Γ_data_iter = vertex_to_matrix.(Ref(Γ_iter), basis_w.grid) .+ Ref(vertex_to_matrix(Γ0, 0.));
    Γ_data_direct = vertex_to_matrix.(Ref(Γ_direct), basis_w.grid);
    @test norm(Γ_data_iter .- Γ_data_direct) < 1e-10
end
