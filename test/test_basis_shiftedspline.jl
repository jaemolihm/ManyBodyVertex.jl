using Test
using ManyBodyVertex

@testset "ShiftedSplineBasis" begin
    using QuadGK
    basis = ShiftedSplineBasis(2, 3, -4.:4.0:4.)
    @test size(basis, 2) == 10

    ManyBodyVertex.set_basis_shift!(basis, 0)
    @test basis[-4.0, :] ≈ repeat([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    @test basis[-2.0, :] ≈ repeat([0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0])
    @test basis[ 0.0, :] ≈ repeat([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    @test basis[ 2.0, :] ≈ repeat([0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0])
    @test basis[ 4.0, :] ≈ repeat([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    for x in (4.1, -4.1, 10., -10.)
        @test if x > 0
            basis[x, :] ≈ vcat([(4/x)^2, (4/x)^3, 0, 0], zeros(6))
        else
            basis[x, :] ≈ vcat([0, 0, (-4/x)^2, (-4/x)^3], zeros(6))
        end
    end

    ManyBodyVertex.set_basis_shift!(basis, 0)
    vs = vcat(basis.basis.grid, -4-1e-10, 4+1e-10)
    @test all(maximum(basis[vs, :], dims=1) .≈ 1)

    ManyBodyVertex.set_basis_shift!(basis, 1)
    vs = vcat(basis.basis.grid, -4.5-1e-10, 4.5+1e-10)
    @test all(maximum(basis[vs, :], dims=1) .≈ 1)

    ManyBodyVertex.set_basis_shift!(basis, 0)
    @test get_fitting_points(basis) ≈ [-12, -8, -4, -4, -4, 0, 0, 4, 4, 4, 8, 12]
    ManyBodyVertex.set_basis_shift!(basis, 4)
    @test get_fitting_points(basis) ≈ [-18, -12, -6, -6, -2, -2, 2, 2, 6, 6, 12, 18]
    ManyBodyVertex.set_basis_shift!(basis, 10)
    @test get_fitting_points(basis) ≈ [-27, -18, -9, -9, -5, -1, -1, -0.5, 0, 0.5, 1, 1, 5,
                                       9, 9, 18, 27]

    b1 = ShiftedSplineBasis(2, 3, -4.:4.0:4.);
    b2 = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.);
    ManyBodyVertex.set_basis_shift!(b1, 0)
    @test basis_integral(b1)[1:4] ≈ basis_integral(b2)[1:4]
    @test basis_integral(b1)[5:end] ≈ [0, 2, 2, 2, 2, 0]

    @test basis_integral(b1, b1)[1:4, 1:4] ≈ basis_integral(b2, b2)[1:4, 1:4]
    @test basis_integral(b1, b1)[1:4, 5:end] ≈ zeros(4, 6)
    @test basis_integral(b1, b1)[5:end, 1:4] ≈ zeros(6, 4)

    ManyBodyVertex.set_basis_shift!(b1, 3)
    ov = basis_integral(b1)
    @test ov ≈ basis_integral(b1, ConstantBasis())
    for i in 1:4
        @test ov[i] ≈ quadgk(x -> b1[x, i], -Inf, -5.5)[1] + quadgk(x -> b1[x, i], 5.5, Inf)[1] atol=1e-6
    end
    for i in 5:10
        @test ov[i] ≈ quadgk(x -> b1[x, i], -5.5, 5.5)[1] atol=1e-6
    end
end


@testset "ShiftedSplineBasis interpolation" begin
    # Test interpolation of doubly-peaked functions using ShiftedSplineBasis works.
    basis = ShiftedSplineBasis(2, 4, get_nonequidistant_grid(5., 31))
    f(x, w) = 1 / ((x - w/2)^2 + 2) / ((x + w/2)^2 + 0.5)
    for w in [-20, -10, -5, -1, 0, 2/45, 1, 5, 10, 20]
        ManyBodyVertex.set_basis_shift!(basis, w)
        xs = get_fitting_points(basis)
        coeff = basis[xs, :] \ f.(xs, w)
        xs_test = range(-30, 30, length=1001)
        y_exact = f.(xs_test, w)
        y_interp = basis[xs_test, :] * coeff
        @test maximum(y_exact .- y_interp) / maximum(y_exact) < 0.01
    end
end
