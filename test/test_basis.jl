using Test
using mfRG

@testset "basis functions" begin
    basis = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.)
    @test size(basis, 2) == 7
    @test basis[-4.0, :] ≈ [0, 0, 0, 0, 1, 0, 0]
    @test basis[-2.0, :] ≈ [0, 0, 0, 0, 0.5, 0.5, 0]
    @test basis[ 0.0, :] ≈ [0, 0, 0, 0, 0, 1, 0]
    @test basis[ 2.0, :] ≈ [0, 0, 0, 0, 0, 0.5, 0.5]
    @test basis[ 4.0, :] ≈ [0, 0, 0, 0, 0, 0, 1]
    for x in (nextfloat(4.), prevfloat(-4.), 10., -10.)
        @test basis[x, :] ≈ [(4/x)^2, (4/x)^3, sign(x) * (4/x)^2, sign(x) * (4/x)^3, 0, 0, 0]
    end
    @test all(maximum(basis[vcat(basis.grid, prevfloat(-4.), nextfloat(4.)), :], dims=1) .≈ 1)

    points = get_fitting_points(basis)
    @test points ≈ [-12, -8, -4, -4, 0, 4, 4, 8, 12]
    @test allunique(points)

    basis = ConstantBasis()
    @test all(basis[-1:1, 1] .≈ 1)
    @test_throws BoundsError basis[1, 2]

    # Test basis_integral
    basis = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.)
    basis2 = LinearSplineAndTailBasis(0, 0, [-1., 1.])
    @test basis_integral(basis) ≈ [8, 0, 0, 4, 2, 4, 2]
    @test basis_integral(basis, basis2) ≈ [8 0 0 0;
                                           0 4 0 0;
                                           0 8 0 0;
                                           4 0 0 0;
                                           15/8 -15/8 5/48 1/48;
                                           9/4 0 7/8 7/8;
                                           15/8 15/8 1/48 5/48]
    @test basis_integral(basis, ConstantBasis()) ≈ basis_integral(basis)
end

@testset "Imaginary basis" begin
    using mfRG: InfRange

    # Test InfRange
    @test 3 ∈ InfRange()
    @test 3.5 ∉ InfRange()
    @test length(InfRange()) > typemax(Int)
    @test (1:2) != InfRange()
    @test InfRange() == InfRange()

    # Test ImagConstantBasis
    b = ImagConstantBasis()
    @test axes(b) == (InfRange(), 1:1)
    @test size(b) == (length(InfRange()), 1)
    @test b[10, 1] == 1

    # Test ImagGridAndTailBasis
    b = ImagGridAndTailBasis(0, 1, -2, 1)
    @test axes(b) == (InfRange(), 1:6)
    @test size(b) == (length(InfRange()), 6)
    @test b[0, :] ≈ [0, 0, 0, 0, 1, 0]
    @test b[2, :] ≈ [1, 1/2, 0, 0, 0, 0]
    @test b[-3, :] ≈ [1, -1/3, 0, 0, 0, 0]
    @test get_fitting_points(b) == [-6, -4, -2, -1, 0, 1, 4, 6]

    @test frequency_index_bounds(3, :Boson) == (-3, 3)
    @test frequency_index_bounds(3, :Fermion) == (-3, 2)
end

@testset "basis_for_bubble" begin
    using mfRG: ntails
    basis_v = LinearSplineAndTailBasis(2, 3, -4.:1:4.)
    basis_w = LinearSplineAndTailBasis(1, 3, [-3., 3.])
    basis_v_bubble, basis_w_bubble = basis_for_bubble(basis_v, basis_w)
    @test ntails(basis_v_bubble) == ntails(basis_v)
    @test basis_v_bubble.grid[end] > maximum(get_fitting_points(basis_w)) * 0.5
    @test basis_v_bubble.grid[1] < minimum(get_fitting_points(basis_w)) * 0.5
    @test ntails(basis_w_bubble) == 0
    @test basis_w_bubble.grid[end] >= maximum(get_fitting_points(basis_w))
    @test basis_w_bubble.grid[1] <= minimum(get_fitting_points(basis_w))

    # TODO: Imaginary basis
end
