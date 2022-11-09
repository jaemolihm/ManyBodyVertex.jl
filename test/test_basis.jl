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

    # Test ImagGridAndTailBasis for bosons
    b_boson = ImagGridAndTailBasis(:Boson, 0, 1, 2)  # grid for -2:2
    @test axes(b_boson) == (InfRange(), 1:9)
    @test b_boson[-2, :] ≈ [0, 0, 0, 0, 1, 0, 0, 0, 0]
    @test b_boson[ 1, :] ≈ [0, 0, 0, 0, 0, 0, 0, 1, 0]
    for m in [-11, -5, 3, 10]
        x = (2 + 1) / m
        @test b_boson[m, :] ≈ [1, x, sign(m) * 1, sign(m) * x, 0, 0, 0, 0, 0]
    end
    @test all(maximum(b_boson[-5:5, :], dims=1) .≈ 1)
    ws = get_fitting_points(b_boson)
    @test ws == vcat(-9, -6, -3, -2:2, 3, 6, 9)
    @test ws == .-reverse(ws)  # Test symmetry of fitting points

    # Test ImagGridAndTailBasis for Fermions
    b_fermion = ImagGridAndTailBasis(:Fermion, 0, 1, 2)  # grid for -2:1
    @test axes(b_fermion) == (InfRange(), 1:8)
    @test b_fermion[-2, :] ≈ [0, 0, 0, 0, 1, 0, 0, 0]
    @test b_fermion[ 1, :] ≈ [0, 0, 0, 0, 0, 0, 0, 1]
    for m in [-11, -5, 3, 10]
        x = (2 + 1/2) / (m + 1/2)
        @test b_fermion[m, :] ≈ [1, x, sign(m) * 1, sign(m) * x, 0, 0, 0, 0]
    end
    @test all(maximum(b_fermion[-5:5, :], dims=1) .≈ 1)
    vs = get_fitting_points(b_fermion)
    @test vs == vcat(-8, -5, -3, -2:1, 2, 4, 7)
    @test vs .+ 1/2 == .-reverse(vs .+ 1/2)  # Test symmetry of fitting points

    # Special case: Fermions with only tails
    @test get_fitting_points(ImagGridAndTailBasis(:Fermion, 0, 1, 0)) == -3:2

    # Test basis_integral
    basis = ImagGridAndTailBasis(:Fermion, 2, 3, 2)
    basis2 = ImagGridAndTailBasis(:Fermion, 0, 0, 1)
    @test basis_integral(basis) ≈ [6.129471951252935, 0, 0, 3.690688306902043, 1, 1, 1, 1]
    @test basis_integral(basis, basis2) ≈ [6.129471951252935 0 0 0;
                                            0 3.690688306902043 0 0;
                                            0 6.129471951252935 0 0;
                                            3.690688306902043 0 0 0;
                                            1 -1 0 0;
                                            0 0 1 0;
                                            0 0 0 1;
                                            1 1 0 0]
    @test basis_integral(basis, ImagConstantBasis()) ≈ basis_integral(basis)
end

@testset "integrate imag" begin
    using mfRG: integrate_imag
    # zeta(n) = 1/1^n + 1/2^n + 1/3^n + ...
    zeta = [0, π^2/6, 1.202056903159594285, π^4/90, 1.036927755143369926, π^6/945]
    for n in 2:5
        f(x) = (1/x)^n
        y = zeta[n]
        @test integrate_imag(f, 1, typemax(Int))[1] ≈ y atol=2e-14
        @test mfRG.integrate_imag(f, typemin(Int), -1)[1] ≈ (-1)^n * y atol=2e-14

        n0 = 30
        y = zeta[n] - sum(f, 1:n0-1)
        @test integrate_imag(f, n0, typemax(Int))[1] ≈ y atol=2e-14
        @test integrate_imag(f, typemin(Int), -n0)[1] ≈ (-1)^n * y atol=2e-14

        y = sum(f, 1:n0)
        @test integrate_imag(f, 1, n0)[1] ≈ y atol=2e-14
        @test integrate_imag(f, -n0, -1)[1] ≈ (-1)^n * y atol=2e-14
    end
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
