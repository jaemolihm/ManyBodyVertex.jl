using Test
using ManyBodyVertex

@testset "Vertex4P" begin
    using LinearAlgebra

    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, -2:0.5:2)
    basis3 = LinearSplineAndTailBasis(2, 4, [-1., 1.])
    n1 = size(basis1, 2)
    n2 = size(basis2, 2)
    n3 = size(basis3, 2)

    @test_throws ArgumentError Vertex4P{:KF}(:X, basis1, basis1, basis2)

    Γ = Vertex4P{:KF}(:A, basis1, basis1, basis2)
    @test ManyBodyVertex.nkeldysh(Γ) == 2
    @test size(Γ.data) == (n1 * 2^2, n1 * 2^2, n2)
    @test size(keldyshview(Γ)) == (n1, n1, 1, 1, 1, 1, 2, 2, 2, 2, n2)
    @test size(to_matrix(Γ, 0.3)) == (n1 * 2^2, n1 * 2^2)

    Γ = Vertex4P{:ZF}(:A, basis1, basis1, basis2, 3)
    @test ManyBodyVertex.nkeldysh(Γ) == 1
    @test size(Γ.data) == (n1 * 3^2, n1 * 3^2, n2)
    @test size(to_matrix(Γ, 0.3)) == (n1 * 3^2, n1 * 3^2)

    Γ2 = Γ + 2 * Γ + Γ * 3 + Γ / 2
    @test Γ2.data ≈ Γ.data .* 6.5

    Γ_sim = similar(Γ)
    @test Γ_sim isa typeof(Γ)
    @test Γ_sim.basis_f1 === Γ.basis_f1
    @test Γ_sim.basis_f2 === Γ.basis_f2
    @test Γ_sim.basis_b === Γ.basis_b
    @test Γ_sim.norb === Γ.norb
    @test size(Γ_sim.data) == size(Γ.data)

    # Test fitting
    for basis2 in [LinearSplineAndTailBasis(0, -1, -3:0.5:1), LinearSplineAndTailBasis(2, 4, -2:0.5:2)]
        Γ = Vertex4P{:ZF}(:A, basis1, basis1, basis2)
        ws = get_fitting_points(basis2)
        Γ_data = reshape(1 ./ (ws.^2 .+ 1), 1, 1, length(ws))
        fit_bosonic_basis_coeff!(Γ, Γ_data, ws)
        y = Γ.basis_b[ws, :] * Γ.data[1, 1, :]
        @test norm(y - Γ_data[1, 1, :]) / norm(Γ_data) < 1e-3
        if basis2 isa LinearSplineAndTailBasis && ManyBodyVertex.ntails(basis2) == 0
            # If there is no tail, fitting must be exact (interpolative)
            @test y ≈ Γ_data[1, 1, :]
        end
    end
end

@testset "Vertex channel" begin
    # Test evaluation of vertex at different channels
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(0, 1, [-2., 2.])
    Γ = Vertex4P{:ZF}(:A, Float64, basis1, basis1, basis2)
    vec(Γ.data) .= [1., 2., 3., 4., 5., 6.]
    @test Γ(1., 1., 3., :A)[1, 1] ≈ 1 + 2 * 2/3
    @test Γ(1., 1., 3., :P)[1, 1] ≈ 5.
    @test Γ(1., 1., 3., :T)[1, 1] ≈ 5.5
    @test Γ(0., 1., 3., :A) == Γ(0., 1., 3.)
    for C in (:A, :P, :T)
        @inferred Γ(0., 1., 3., C)
    end

    # Test apply_crossing
    Γ_A = Vertex4P{:KF}(:A, Float64, basis1, basis2, basis2)
    Γ_A.data .= rand(size(Γ_A.data)...)
    Γ_T = apply_crossing(Γ_A)
    @test Γ_T isa Vertex4P{:KF}
    @test Γ_A(0.2, 0.3, 0.4) ≈ .-Γ_T(0.2, 0.3, 0.4)
end
