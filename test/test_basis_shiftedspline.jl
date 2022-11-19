using Test
using mfRG

@testset "ShiftedSplineBasis" begin
    using QuadGK
    basis = ShiftedSplineBasis(2, 3, -4.:4.0:4.)
    @test size(basis, 2) == 14
    basis.w = 0
    @test basis[-4.0, :] ≈ repeat([0, 0, 0, 0, 1, 0, 0], 2)
    @test basis[-2.0, :] ≈ repeat([0, 0, 0, 0, 0.5, 0.5, 0], 2)
    @test basis[ 0.0, :] ≈ repeat([0, 0, 0, 0, 0, 1, 0], 2)
    @test basis[ 2.0, :] ≈ repeat([0, 0, 0, 0, 0, 0.5, 0.5], 2)
    @test basis[ 4.0, :] ≈ repeat([0, 0, 0, 0, 0, 0, 1], 2)
    for x in (nextfloat(4.), prevfloat(-4.), 10., -10.)
        @test if x > 0
            basis[x, :] ≈ vcat([(4/x)^2, (4/x)^3, 0, 0, 0, 0, 0], zeros(7))
        else
            basis[x, :] ≈ vcat(zeros(7), [0, 0, (-4/x)^2, (-4/x)^3, 0, 0, 0])
        end
    end

    basis.w = 0
    vs = vcat(basis.grid, prevfloat(-4.), nextfloat(4.))
    @test all(maximum(basis[vs, vcat(1:2, 5:7, 10:11, 12:14)], dims=1) .≈ 1)

    basis.w = 1
    @test all(maximum(basis[vs .+ 0.5, vcat(1:2, 5:7)], dims=1) .≈ 1)
    @test all(maximum(basis[vs .- 0.5, vcat(10:11, 12:14)], dims=1) .≈ 1)

    @test (basis.w = 2; basis[-4.0, :]) ≈ (basis.w = 0; vcat(basis[-5.0, 1:7], basis[-3.0, 1:7]))

    basis.w = 0
    @test get_fitting_points(basis) ≈ [-12, -8, -4, -4, 0, 4, 4, 8, 12]
    basis.w = 4
    @test get_fitting_points(basis) ≈ [-14, -10, -6, -6, -2, 2, 6, 6, 10, 14]
    basis.w = 10
    @test get_fitting_points(basis) ≈ [-17, -13, -9, -9, -5, -1, -1, -1/3,
                                            1/3, 1, 1, 5, 9, 9, 13, 17]


    b1 = ShiftedSplineBasis(2, 3, -4.:4.0:4.);
    b2 = LinearSplineAndTailBasis(2, 3, -4.:4.0:4.);
    b1.w = 0
    x_ref = repeat(basis_integral(b2), 2)
    x_ref[vcat(3:4, 8:9)] .= 0  # zero tails due to overlapping region
    @test basis_integral(b1) ≈ x_ref

    x2_ref = basis_integral(b2, b2);
    x = [x2_ref x2_ref; x2_ref x2_ref]
    x[vcat(3:4, 8:9), :] .= 0  # zero tails due to overlapping region
    x[:, vcat(3:4, 8:9)] .= 0  # zero tails due to overlapping region
    @test basis_integral(b1, b1) ≈ x
    x = [x2_ref; x2_ref]
    x[vcat(3:4, 8:9), :] .= 0  # zero tails due to overlapping region
    @test basis_integral(b1, b2) ≈ x

    b1.w = 3.
    ov = basis_integral(b1)
    @test ov ≈ basis_integral(b1, ConstantBasis())
    for i in 1:4
        @test ov[i] ≈ quadgk(x -> b1[x, i], -Inf, -2.5)[1] + quadgk(x -> b1[x, i], 5.5, Inf)[1] atol=1e-6
    end
    for i in 8:11
        @test ov[i] ≈ quadgk(x -> b1[x, i], -Inf, -3.5)[1] + quadgk(x -> b1[x, i], 2.5, Inf)[1] atol=1e-6
    end

    inds_spline = [5, 6, 7, 12, 13, 14]
    @test (b1.w = 0; basis_integral(b1)[inds_spline]) ≈ (b1.w = 3.; basis_integral(b1)[inds_spline])
end
