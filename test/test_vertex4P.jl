using Test
using mfRG

@testset "Vertex4P" begin
    basis1 = ConstantBasis()
    basis2 = LinearSplineAndTailBasis(2, 4, vgrid)
    basis3 = LinearSplineAndTailBasis(2, 4, [-1., 1.])
    n1 = size(basis1, 2)
    n2 = size(basis2, 2)
    n3 = size(basis3, 2)

    Γ = zeros_vertex(Val(:KF), basis1, basis2)
    @test mfRG.nkeldysh(Γ) == 2
    @test size(Γ.data) == (n1 * 2^2, n1 * 2^2, n2)
    @test size(vertex_keldyshview(Γ)) == (n1, n1, 1, 1, 1, 1, 2, 2, 2, 2, n2)
    @test size(vertex_to_matrix(Γ, 0.3)) == (n1 * 2^2, n1 * 2^2)

    Γ = zeros_vertex(Val(:ZF), basis1, basis2, 3)
    @test mfRG.nkeldysh(Γ) == 1
    @test size(Γ.data) == (n1 * 3^2, n1 * 3^2, n2)
    @test size(vertex_to_matrix(Γ, 0.3)) == (n1 * 3^2, n1 * 3^2)

    Π = zeros_bubble(Val(:KF), basis3, basis2, 3)
    overlap = basis_integral(basis1, basis1, basis3)
    @test size(Π.data) == (n3, 6^2, 6^2, n2)
    @test size(bubble_to_matrix(Π, 0.3, overlap)) == (n1 * 6^2, n1 * 6^2)
end
