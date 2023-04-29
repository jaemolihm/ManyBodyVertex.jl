# 221130
* Optimized `compute_self_energy_SU2` and `compute_bubble_smoothed`.

# 221208 - 221209
* Implemented better (faster convergence w.r.t. frequency grid) of the self-energy
* Make the formula work also for parquet without irreducible vertex.
* Ref: Eq. (12) of C. Hille et al, PRResearch 2 033372 (2020)
* `workspace/221208_better_self_energy.jl`

# 221210 - 221213
* Implemented bosonic momentum dependent vertex, bubble, and self-energy
* Benchmark against C. Hille et al, PRResearch 2 033372 (2020) (particle-hole symmetric only)
* Future works
  * High-frequency part of self-energy is local -> use this information?
  * Nonlocal K3 vertex can be well approximated by SBE

# 221214 - 221215
* Implemented response function calculation (susceptibility): `response.jl`.

# 221221
* Optimized `compute_bubble_smoothed` by defining and using `basis_integral_bubble`.

# 221226
* Implemented analytic continuation from KF to MF: `analytic_continuation_KF_to_MF`.
* Added test of analytic continuation (bare GF, parquet results): `test_analytic_continuation.jl`.
* Questions
  * Why is convergence of chi AC w.r.t. vgrid_k1, wgrid_k1, and vgrid_k3 density too slow?
  * Why is convergence of Hartree self-energy in KF much slower than other self-energy?

# TODO
- [ ] Particle-hole nonsymmetric case. Need Hartree self-energy term.
- [ ] Cache bubble * vertex
- [ ] ZF
- [ ] Hubbard atom

