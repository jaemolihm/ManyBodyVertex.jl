using Test
using mfRG

@testset "Bubble" begin
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, -2:0.5:2)
    basis3 = LinearSplineAndTailBasis(2, 4, [-1., 1.])
    n1 = size(basis1, 2)
    n2 = size(basis2, 2)
    n3 = size(basis3, 2)

    Π = Bubble{:KF}(basis3, basis2, 3)
    overlap = basis_integral(basis1, basis1, basis3)
    @test size(Π.data) == (n3, 6^2, 6^2, n2)
    @test size(to_matrix(Π, 0.3, overlap)) == (n1 * 6^2, n1 * 6^2)
end
