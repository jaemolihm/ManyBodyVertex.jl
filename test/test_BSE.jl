using Test
using ManyBodyVertex

@testset "BSE static" begin
    using LinearAlgebra

    for F in (:KF, :MF), C in (:A, :P, :T)
        U = 4.0
        temperature = 0.3

        if F === :KF
            basis1 = ConstantBasis()
            basis2 = LinearSplineAndTailBasis(2, 4, -4:2.:4)
            basis_w = LinearSplineAndTailBasis(1, 3, -4:4.:4)
        elseif F === :MF
            basis1 = ImagConstantBasis()
            basis2 = ImagGridAndTailBasis(:Fermion, 2, 4, 5)
            basis_w = ImagGridAndTailBasis(:Boson, 1, 3, 5)
        end

        # Set static vertx
        Γ0 = get_bare_vertex(Val(F), C, U)

        # Set bubble. Use random numbers for testing.
        Π = Bubble{F}(C, basis2, basis_w; temperature)
        Π.data .= rand(eltype(Π.data), size(Π.data)...)

        # Solve BSE by direct inversion
        overlap = basis_integral(basis1, basis1, basis2);
        Γ_direct = Vertex4P{F}(C, basis1, basis1, basis_w)
        for (iw, w) in enumerate(basis_w.grid)
            Γ0_mat = to_matrix(Γ0, w)
            Π_mat = to_matrix(Π, w, basis1, basis1)

            # Direct solution: Γ = inv(I - Γ0 * Π) * Γ0
            Γ_mat = (I - Γ0_mat * Π_mat) \ Γ0_mat
            Γ_direct.data[:, :, iw + ManyBodyVertex.ntails(basis_w)] .= Γ_mat
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
        inds_interp = ManyBodyVertex.ntails(basis_w)+1:size(basis_w, 2)

        # Test Γ = Γ0 Π Γ0 + Γ0 Π Γ
        Γ0_Π_Γ0 = vertex_bubble_integral(Γ0, Π, Γ0, basis_w)
        Γ_test1 = vertex_bubble_integral(Γ0, Π, Γ, basis_w)
        Γ_test1.data .+= Γ0_Π_Γ0.data
        @test norm((Γ_test1.data .- Γ.data)[:, :, inds_interp]) < 1e-10

        # Test Γ = Γ0 Π Γ0 + Γ Π Γ0
        Γ_test2 = vertex_bubble_integral(Γ, Π, Γ0, basis_w)
        Γ_test2.data .+= Γ0_Π_Γ0.data
        @test norm((Γ_test2.data .- Γ.data)[:, :, inds_interp]) < 1e-10

        # Test Γ = Γ0 Πscr Γ0
        Πscr = ManyBodyVertex.ScreenedBubble(Π, Γ0, Γ)
        Γ_test3 = vertex_bubble_integral(Γ0, Πscr, Γ0, basis_w)
        @test norm((Γ_test3.data .- Γ.data)[:, :, inds_interp]) < 1e-10

        # Test left BSE
        Γ_left = solve_BSE_left(Γ0, Π, Γ0, basis_w)
        @test Γ_left.data ≈ Γ.data
    end
end


@testset "vertex caching" begin
    using ManyBodyVertex: get_bare_vertex, ScreenedBubble, cache_vertex_matrix

    function test_cached(ΓL, Π, ΓR, basis_w, basis_aux)
        # Test vertex_bubble_integral with cached vertex matrix.
        # Currently, caching works only if the vertex to be cached is in a different channel
        # with Π. So, test caching only in such a case.
        ws = get_fitting_points(basis_w)
        x_ref = vertex_bubble_integral(ΓL, Π, ΓR, basis_w; basis_aux)

        # Cache ΓL
        ΓL_cache = cache_vertex_matrix(ΓL, get_channel(Π), ws, basis_aux);
        x = vertex_bubble_integral(ΓL_cache, Π, ΓR, basis_w; basis_aux);
        @test x.data ≈ x_ref.data

        # Cache ΓR
        ΓR_cache = cache_vertex_matrix(ΓR, get_channel(Π), ws, basis_aux);
        x = vertex_bubble_integral(ΓL, Π, ΓR_cache, basis_w; basis_aux);
        @test x.data ≈ x_ref.data

        # Cache ΓL and ΓR
        x = vertex_bubble_integral(ΓL_cache, Π, ΓR_cache, basis_w; basis_aux);
        @test x.data ≈ x_ref.data
    end

    U = 1.0
    e = 0.5
    Δ = 0.8
    temperature = 0.1
    basis_f = LinearSplineAndTailBasis(2, 4, -2:0.8:2)
    basis_b = LinearSplineAndTailBasis(1, 0, -3:1.5:3)
    basis_aux = LinearSplineAndTailBasis(1, 0, -4:1.0:4)

    G0 = SIAMLazyGreen2P{:KF}(; e, Δ, temperature)
    ΠA = compute_bubble(G0, G0, basis_f, basis_b, :A; temperature)
    ΠP = compute_bubble(G0, G0, basis_f, basis_b, :P; temperature)
    Γ0_A = get_bare_vertex(Val(:KF), :A, U)
    Γ0_P = get_bare_vertex(Val(:KF), :P, U)
    Γ1_A = solve_BSE(Γ0_A, ΠA, Γ0_A, basis_b)
    Γ1_P = solve_BSE(Γ0_P, ΠP, Γ0_P, basis_b)
    Γ1_T = apply_crossing(Γ1_A)
    ΠAscr = ScreenedBubble(ΠA, Γ0_A, Γ1_A)
    ΠPscr = ScreenedBubble(ΠP, Γ0_P, Γ1_P)

    test_cached(Γ1_A, ΠA, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_P, ΠAscr, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_P, ΠPscr, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_A, ΠP, Γ0_A, basis_b, basis_aux)
    test_cached(Γ1_A, ΠPscr, Γ1_T, basis_b, basis_aux)

    # Test caching of multiple vertices
    x_ref = (vertex_bubble_integral(Γ1_P, ΠA, Γ1_P, basis_b; basis_aux)
           + vertex_bubble_integral(Γ1_T, ΠA, Γ1_P, basis_b; basis_aux))

    ws = get_fitting_points(basis_b)
    ΓL_cache = cache_vertex_matrix([Γ1_P, Γ1_T], get_channel(ΠA), ws, basis_aux);
    x = vertex_bubble_integral(ΓL_cache, ΠA, Γ1_P, basis_b; basis_aux)
    @test x.data ≈ x_ref.data
end
