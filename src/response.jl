using Dictionaries

# TODO: Add test
# TODO: Add example

"""
    susceptibility_operator_SU2(::Val{F}, rbasis, norb=1) where {F}
Return the vertex for computing charge and magnetic susceptibility under SU2 spin symmetry.
In the susceptibility vertex obtained by `X = compute_response_SU2(...)`, `X.total[1]` is
the charge (density) susceptibility ``<ρ(q) ρ(-q)>`` and `X.total[2]` is the spin
susceptibility ``<Sx(q) Sx(-q)> = <Sy(q) Sy(-q)> = <Sz(q) Sz(-q)>``.

The susceptibility value can be accessed as follows.
```
b0 = (1, 1, zeros(SVector{Dim, Int}))
X_charge_q_w = interpolate_to_q(X.total[1], xq, b0, b0)(0, 0, w)[1, 1]
X_spin_q_w   = interpolate_to_q(X.total[2], xq, b0, b0)(0, 0, w)[1, 1]
```
"""
function susceptibility_operator_SU2(::Val{F}, rbasis::RealSpaceBasis{Dim}, norb=1) where {F, Dim}
    F !== :MF && error("Only MF implemented yet")
    # TODO: KF
    # TODO: impurity-only version

    R0 = zeros(SVector{Dim, Int})
    ibL = findfirst(x -> x == (1, 1, R0), rbasis.bonds_L)
    ibL === nothing && error("onsite bond not found in bonds_L")
    ibR = findfirst(x -> x == (1, 1, R0), rbasis.bonds_R)
    ibR === nothing && error("onsite bond not found in bonds_R")
    iR_B = findfirst(x -> x == R0, rbasis.R_B_replicas[ibL, ibR])
    iR_B === nothing && error("onsite R_B not found")

    basis = F === :MF ? ImagConstantBasis() : ConstantBasis()
    A_d = Vertex4P{F, :A}(basis, basis, basis, norb)
    A_m = Vertex4P{F, :A}(basis, basis, basis, norb)
    if F === :MF
        # TODO: Resolve orbital
        @views reshape(A_d.data[1, :, 1], norb, norb) .= I(norb)
        @views reshape(A_m.data[1, :, 1], norb, norb) .= I(norb)
    end
    A = (RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => A_d,))),
         RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => A_m,))))
    A
end

"""
    compute_response_SU2(op1, op2, Γ, Π, basis_response=Γ.basis_k1_b)
Compute linear response function ``X(q) = <op1(q) op2(-q)>``.
"""
function compute_response_SU2(op1, op2, Γ, Π, basis_response=Γ.basis_k1_b)
    C = channel(op1[1])
    ws = get_fitting_points(basis_response.freq)
    disconnected = -1 .* vertex_bubble_integral.(op1, Π, op2, Ref(basis_response))
    connected = mapreduce(.+, [get_irreducible_vertices(C, Γ), [Γ.K1_A],
            [Γ.K2_A], [Γ.K2p_A], [Γ.K3_A], [Γ.Γ0_A]]) do Γ_
        @time Γ_cache = Tuple(cache_vertex_matrix(getindex.(Γ_, i), C, ws, Γ.basis_k2_f) for i in 1:2)
        @time tmp = vertex_bubble_integral.(Γ_cache, Π, op2, Ref(basis_response))
        @time -1 .* vertex_bubble_integral.(op1, Π, tmp, Ref(basis_response))
    end;
    (; total=disconnected .+ connected, disconnected, connected)
end
