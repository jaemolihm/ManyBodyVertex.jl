using Test
using mfRG

@testset "basis functions" begin
    basis = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.)
    @test size(basis, 2) == 5
    @test basis[-4.0, :] ≈ [0, 0, 1, 0, 0]
    @test basis[-2.0, :] ≈ [0, 0, 0.5, 0.5, 0]
    @test basis[ 0.0, :] ≈ [0, 0, 0, 1, 0]
    @test basis[ 2.0, :] ≈ [0, 0, 0, 0.5, 0.5]
    @test basis[ 4.0, :] ≈ [0, 0, 0, 0, 1]
    for x in (nextfloat(4.), prevfloat(-4.), 10., -10.)
        @test basis[x, :] ≈ [1/x^2, 1/x^3, 0, 0, 0]
    end

    points = get_fitting_points(basis)
    @test points ≈ [-8, -4, -4, 0, 4, 4, 8]
    @test points[2] == prevfloat(points[3])
    @test points[6] == nextfloat(points[5])

    basis = ConstantBasis()
    @test all(basis[-1:1, 1] .≈ 1)
    @test_throws BoundsError basis[1, 2]

    # Test basis_integral
    basis = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.)
    basis2 = LinearSplineAndTailBasis(0, 0, [-1., 1.])
    @test basis_integral(basis) ≈ [0.5, 0, 2, 4, 2]
    @test basis_integral(basis, basis2) ≈ [1/2 0 0; 0 0 0; 15/8 5/48 1/48; 9/4 7/8 7/8; 15/8 1/48 5/48]
    @test basis_integral(basis, ConstantBasis()) ≈ basis_integral(basis)
end
