using Test
using mfRG

@testset "Vertex4P" begin
    using LinearAlgebra

    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, -2:0.5:2)
    basis3 = LinearSplineAndTailBasis(2, 4, [-1., 1.])
    n1 = size(basis1, 2)
    n2 = size(basis2, 2)
    n3 = size(basis3, 2)

    @test_throws ArgumentError Vertex4P{:KF, :X}(basis1, basis1, basis2)

    Γ = Vertex4P{:KF, :A}(basis1, basis1, basis2)
    @test mfRG.nkeldysh(Γ) == 2
    @test size(Γ.data) == (n1 * 2^2, n1 * 2^2, n2)
    @test size(vertex_keldyshview(Γ)) == (n1, n1, 1, 1, 1, 1, 2, 2, 2, 2, n2)
    @test size(vertex_to_matrix(Γ, 0.3)) == (n1 * 2^2, n1 * 2^2)

    Γ = Vertex4P{:ZF, :A}(basis1, basis1, basis2, 3)
    @test mfRG.nkeldysh(Γ) == 1
    @test size(Γ.data) == (n1 * 3^2, n1 * 3^2, n2)
    @test size(vertex_to_matrix(Γ, 0.3)) == (n1 * 3^2, n1 * 3^2)

    Π = Bubble{:KF}(basis3, basis2, 3)
    overlap = basis_integral(basis1, basis1, basis3)
    @test size(Π.data) == (n3, 6^2, 6^2, n2)
    @test size(bubble_to_matrix(Π, 0.3, overlap)) == (n1 * 6^2, n1 * 6^2)

    # Test fitting
    for basis2 in [LinearSplineAndTailBasis(0, -1, -3:0.5:1), LinearSplineAndTailBasis(2, 4, -2:0.5:2)]
        Γ = Vertex4P{:ZF, :A}(basis1, basis1, basis2)
        ws = get_fitting_points(basis2)
        Γ_data = reshape(1 ./ (ws.^2 .+ 1), 1, 1, length(ws))
        fit_bosonic_basis_coeff!(Γ, Γ_data, ws)
        y = Γ.basis_b[ws, :] * Γ.data[1, 1, :]
        @test norm(y - Γ_data[1, 1, :]) / norm(Γ_data) < 1e-3
        if basis2 isa LinearSplineAndTailBasis && mfRG.ntails(basis2) == 0
            # If there is no tail, fitting must be exact (interpolative)
            @test y ≈ Γ_data[1, 1, :]
        end
    end
end
