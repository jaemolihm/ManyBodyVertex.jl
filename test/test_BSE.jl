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
    Π = Bubble{:KF, :A}(basis2, basis_w)
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


@testset "vertex caching" begin
    using mfRG: channel, get_bare_vertex, siam_get_bubble, ScreenedBubble, cache_vertex_matrix

    function test_cached(ΓL, Π, ΓR, basis_w, basis_aux)
        # Test vertex_bubble_integral with cached vertex matrix.
        # Currently, caching works only if the vertex to be cached is in a different channel
        # with Π. So, test caching only in such a case.
        ws = get_fitting_points(basis_w)
        x_ref = vertex_bubble_integral(ΓL, Π, ΓR, basis_w; basis_aux)
        if channel(ΓL) !== channel(Π)
            ΓL_cache = cache_vertex_matrix(ΓL, channel(Π), ws, basis_aux);
            x = vertex_bubble_integral(ΓL_cache, Π, ΓR, basis_w; basis_aux);
            @test x.data ≈ x_ref.data
        end
        if channel(ΓR) !== channel(Π)
            ΓR_cache = cache_vertex_matrix(ΓR, channel(Π), ws, basis_aux);
            x = vertex_bubble_integral(ΓL, Π, ΓR_cache, basis_w; basis_aux);
            @test x.data ≈ x_ref.data
        end
        if channel(ΓL) !== channel(Π) && channel(ΓR) !== channel(Π)
            x = vertex_bubble_integral(ΓL_cache, Π, ΓR_cache, basis_w; basis_aux);
            @test x.data ≈ x_ref.data
        end
    end

    U = 1.0
    e = 0.5
    Δ = 0.8
    t = 0.1
    basis_f = LinearSplineAndTailBasis(2, 4, -2:0.4:2)
    basis_b = LinearSplineAndTailBasis(1, 0, -3:1.5:3)
    basis_aux = LinearSplineAndTailBasis(1, 0, -4:1.0:4)

    ΠA = siam_get_bubble(basis_f, basis_b, Val(:KF), Val(:A); e, Δ, t)
    ΠP = siam_get_bubble(basis_f, basis_b, Val(:KF), Val(:P); e, Δ, t)
    Γ0_A = get_bare_vertex(U, Val(:KF), Val(:A))
    Γ0_P = get_bare_vertex(U, Val(:KF), Val(:P))
    Γ1_A = solve_BSE(Γ0_A, ΠA, Γ0_A, basis_b)
    Γ1_P = solve_BSE(Γ0_P, ΠP, Γ0_P, basis_b)
    Γ1_T = apply_crossing(Γ1_A)
    ΠAscr = ScreenedBubble(ΠA, Γ1_A)
    ΠPscr = ScreenedBubble(ΠP, Γ1_P)

    test_cached(Γ1_A, ΠA, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_P, ΠAscr, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_P, ΠPscr, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_A, ΠP, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_A, ΠPscr, Γ1_T, basis_b, basis_aux)

    # Test caching of multiple vertices
    x_ref = (vertex_bubble_integral(Γ1_P, ΠA, Γ1_P, basis_b; basis_aux)
           + vertex_bubble_integral(Γ1_T, ΠA, Γ1_P, basis_b; basis_aux))

    ws = get_fitting_points(basis_b)
    ΓL_cache = cache_vertex_matrix([Γ1_P, Γ1_T], channel(ΠA), ws, basis_aux);
    x = vertex_bubble_integral(ΓL_cache, ΠA, Γ1_P, basis_b; basis_aux)
    @test x.data ≈ x_ref.data
end
