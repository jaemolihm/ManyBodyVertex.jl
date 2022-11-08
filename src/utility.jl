"""
    get_nonequidistant_grid(wmax, n, w_s=Inf)
Generate nonequidistant real-frequency grid. Using the transformation function
``f(Ω) = w_s * sign(Ω) * Ω^2 / sqrt(1 - Ω^2)`` (Eq. (6.3) of reference), uniformly sample
`n` points of `Ω` in `[Ω_min, Ω_max] = [f⁻¹(-wmax), f⁻¹(wmax)]`, and then convert to `w` as
`w_i = f(Ω_i)`.

If `w_s == Inf`, the step size is proportional to |w| (linearly increasing step size).

The generated grid changes linearly when one scales both `wmax` and `w_s`.

Ref: Eq. (6.3) of E. Walter LMU thesis (2021)
"""
function get_nonequidistant_grid(wmax, n; w_s=Inf)
    # w_s == Inf means step size proportional to w (linearly increasing step size).
    f(x) = isinf(w_s) ? sign(x) * x^2 : w_s * sign(x) * x^2 / sqrt(1 - x^2)
    inv_f(y) = isinf(w_s) ? sqrt(y) : sqrt((-y^2 + sqrt(y^4 + 4 * w_s^2 * y^2)) / 2 / w_s^2)
    @assert f(inv_f(2.0)) ≈ 2.0
    f.(range(-inv_f(wmax), inv_f(wmax), length=n))
end
