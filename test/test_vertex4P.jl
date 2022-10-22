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
    @test size(to_matrix(Γ, 0.3)) == (n1 * 2^2, n1 * 2^2)

    Γ = Vertex4P{:ZF, :A}(basis1, basis1, basis2, 3)
    @test mfRG.nkeldysh(Γ) == 1
    @test size(Γ.data) == (n1 * 3^2, n1 * 3^2, n2)
    @test size(to_matrix(Γ, 0.3)) == (n1 * 3^2, n1 * 3^2)

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

@testset "Vertex channel" begin
    # Test evaluation of vertex at different channels
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(0, 1, [-2., 2.])
    Γ = Vertex4P{:ZF, :A}(Float64, basis1, basis1, basis2)
    vec(Γ.data) .= [1., 2., 3., 4.]
    @test Γ(1., 1., 3., Val(:A))[1, 1] ≈ 1 + 2/3
    @test Γ(1., 1., 3., Val(:P))[1, 1] ≈ 3.
    @test Γ(1., 1., 3., Val(:T))[1, 1] ≈ 3.5
    @test Γ(0., 1., 3., Val(:A)) == Γ(0., 1., 3.)
    for c in (:A, :P, :T)
        @inferred Γ(0., 1., 3., Val(c))
    end

    # Test apply_crossing
    Γ_A = Vertex4P{:KF, :A}(Float64, basis1, basis2, basis2)
    Γ_A.data .= rand(size(Γ_A.data)...)
    Γ_T = apply_crossing(Γ_A)
    @test Γ_T isa Vertex4P{:KF, :T}
    @test Γ_A(0.2, 0.3, 0.4) ≈ .-Γ_T(0.2, 0.3, 0.4)
end
