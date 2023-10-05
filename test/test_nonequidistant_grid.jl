using ManyBodyVertex
using Test

@testset "nonequidistant grid" begin
    wmax = 5.
    n = 11
    grid = get_nonequidistant_grid(wmax, n)
    @test all(extrema(grid) .≈ (-wmax, wmax))
    @test length(grid) == n
    @test diff(grid) ≈ minimum(diff(grid)) .* vcat(9:-2:1, 1:2:9)

    @test grid ≈ get_nonequidistant_grid(5., 11, w_s=1e5) atol=1e-4

    factor = 6.
    w_s = 1.7
    @test grid .* factor ≈ get_nonequidistant_grid(5. * factor, n)
    @test get_nonequidistant_grid(wmax, n; w_s) .* factor ≈ get_nonequidistant_grid(
        wmax * factor, n, w_s=w_s * factor)
end
