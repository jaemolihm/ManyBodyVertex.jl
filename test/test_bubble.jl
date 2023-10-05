using Test
using ManyBodyVertex

@testset "Bubble" begin
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, -2:0.5:2)
    basis3 = LinearSplineAndTailBasis(2, 4, [-1., 1.])
    n1 = nbasis(basis1)
    n2 = nbasis(basis2)
    n3 = nbasis(basis3)

    Π = Bubble{:KF}(:A, basis3, basis2, 3)
    overlap = basis_integral(basis1, basis2, basis3)
    @test size(Π.data) == (n3, 6^2, 6^2, n2)
    @test size(to_matrix(Π, 0.3, overlap)) == (n1 * 6^2, n2 * 6^2)
    @test to_matrix(Π, 0.3, basis1, basis2) ≈ to_matrix(Π, 0.3, overlap)
end
