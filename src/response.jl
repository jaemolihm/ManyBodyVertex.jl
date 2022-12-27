using Dictionaries

"""
# Response
For the KF, the physical (retarded) response is the k = (1, 2) component.
"""


# TODO: Add example


"""
    susceptibility_operator_SU2(::Val{F}, norb=1) where {F}
Return a (local) Vertex4P object for computing charge and magnetic susceptibility under SU2
spin symmetry.
"""
function susceptibility_operator_SU2(::Val{F}, norb=1) where {F}
    basis = F === :MF ? ImagConstantBasis() : ConstantBasis()
    op_L_d = Vertex4P{F, :A}(basis, basis, basis, norb)
    op_L_m = Vertex4P{F, :A}(basis, basis, basis, norb)
    op_R_d = Vertex4P{F, :A}(basis, basis, basis, norb)
    op_R_m = Vertex4P{F, :A}(basis, basis, basis, norb)
    if F === :MF
        # TODO: Resolve orbital
        @views reshape(op_L_d.data[1, :, 1], norb, norb) .= I(norb)
        @views reshape(op_L_m.data[1, :, 1], norb, norb) .= I(norb)
        @views reshape(op_R_d.data[:, 1, 1], norb, norb) .= I(norb)
        @views reshape(op_R_m.data[:, 1, 1], norb, norb) .= I(norb)
    elseif F === :KF
        op_L_d_kv = keldyshview(op_L_d)
        op_L_m_kv = keldyshview(op_L_m)
        op_R_d_kv = keldyshview(op_R_d)
        op_R_m_kv = keldyshview(op_R_m)
        for ks in CartesianIndices((2, 2, 2, 2))
            @views if mod(sum(ks.I), 2) == 1
                op_L_d_kv[1, 1, 1, 1, :, :, ks, 1] .= I(norb) ./ sqrt(2)
                op_L_m_kv[1, 1, 1, 1, :, :, ks, 1] .= I(norb) ./ sqrt(2)
                op_R_d_kv[1, 1, :, :, 1, 1, ks, 1] .= I(norb) ./ sqrt(2)
                op_R_m_kv[1, 1, :, :, 1, 1, ks, 1] .= I(norb) ./ sqrt(2)
            end
        end
    else
        error("Wrong formalism $F. Only :KF an :MF are allowed")
    end
    (op_L_d, op_L_m), (op_R_d, op_R_m)
end

"""
    susceptibility_operator_SU2(::Val{F}, rbasis, norb=1) where {F}
Return a RealSpaceVertex for computing charge and magnetic susceptibility under SU2 spin
symmetry. In the susceptibility vertex obtained by `X = compute_response_SU2(...)`,
`X.total[1]` is the charge (density) susceptibility ``<ρ(q) ρ(-q)>`` and `X.total[2]` is the
spin susceptibility ``<Sx(q) Sx(-q)> = <Sy(q) Sy(-q)> = <Sz(q) Sz(-q)>``.

The susceptibility value can be accessed as follows.
```
b0 = (1, 1, zeros(SVector{Dim, Int}))
X_charge_q_w = interpolate_to_q(X.total[1], xq, b0, b0)(0, 0, w)[1, 1]
X_spin_q_w   = interpolate_to_q(X.total[2], xq, b0, b0)(0, 0, w)[1, 1]
```
"""
function susceptibility_operator_SU2(::Val{F}, rbasis::RealSpaceBasis{Dim}, norb=1) where {F, Dim}
    R0 = zeros(SVector{Dim, Int})
    ibL = findfirst(x -> x == (1, 1, R0), rbasis.bonds_L)
    ibL === nothing && error("onsite bond not found in bonds_L")
    ibR = findfirst(x -> x == (1, 1, R0), rbasis.bonds_R)
    ibR === nothing && error("onsite bond not found in bonds_R")
    iR_B = findfirst(x -> x == R0, rbasis.R_B_replicas[ibL, ibR])
    iR_B === nothing && error("onsite R_B not found")

    op_loc_L, op_loc_R = susceptibility_operator_SU2(Val(F), norb)
    op_L = (RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => op_loc_L[1],))),
            RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => op_loc_L[2],))))
    op_R = (RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => op_loc_R[1],))),
            RealSpaceVertex{:A}(rbasis, dictionary(((ibL, ibR, iR_B) => op_loc_R[2],))))
    op_L, op_R
end


"""
    compute_response_SU2(op1, op2, Γ, Π, basis_response=Γ.basis_k1_b)
Compute linear response function ``X(q) = <op1(q) op2(-q)>``.
"""
function compute_response_SU2(op1, op2, Γ, Π, basis_response=Γ.basis_k1_b)
    C = channel(op1[1])
    ws = get_fitting_points(basis_response.freq)
    disconnected = -1 .* response_4p_to_2p.(vertex_bubble_integral.(op1, Π, op2, Ref(basis_response)))

    vertices = [get_irreducible_vertices(C, Γ), [Γ.K1_A], [Γ.K2_A], [Γ.K2p_A], [Γ.K3_A], [Γ.Γ0_A]]
    filter!(!Base.Fix1(all, isnothing), vertices)
    connected = mapreduce(.+, vertices) do Γ_
        # FIXME: different real space
        Γ_cache = Tuple(cache_vertex_matrix(getindex.(Γ_, i), C, ws, Γ.basis_k2_f) for i in 1:2)
        tmp = vertex_bubble_integral.(Γ_cache, Π, op2, Ref(basis_response))
        -1 .* response_4p_to_2p.(vertex_bubble_integral.(op1, Π, tmp, Ref(basis_response)))
    end;
    (; total=disconnected .+ connected, disconnected, connected)
end


"""
    response_4p_to_2p(X)
Convert a response function in the 4-point vertex form to a bosonic 2-point Green function.
"""
function response_4p_to_2p(X::Vertex4P{F}) where {F}
    if F === :MF
        Green2P{F}(X.basis_b, X.norb, X.data)
    else
        data = zeros(eltype(X), X.norb, 2, X.norb, 2, nb_b(X))
        data_vertex_kv = keldyshview(X)
        # Map two Keldysh indices to one: (2, 1) (or (1, 2)) -> 1, (1, 1) (or (2, 2)) -> 2
        # This mapping is designed to satisfy k1 + k2 = k12 (mod 2).
        for (i2, ks2) in enumerate(((2, 1), (1, 1)))
            for (i1, ks1) in enumerate(((2, 1), (1, 1)))
                # Select orbital index 1.
                data[1, i1, 1, i2, :] .= data_vertex_kv[1, 1, 1, 1, 1, 1, ks1..., ks2..., :]
            end
        end
        nind = get_nind(X)
        Green2P{F}(X.basis_b, X.norb, reshape(data, nind, nind, :))
    end
end

function response_4p_to_2p(X::RealSpaceVertex{F, RC}) where {F, RC}
    RealSpaceVertex{RC}(X.rbasis, response_4p_to_2p.(X.vertices_R))
end


"""
    compute_occupation_matrix(G, temperature=nothing)
Compute the equilibrium occupation matrix (density matrix). 0 is empty and 1 is fully filled.
"""
function compute_occupation_matrix(G, temperature=nothing)
    # FIXME: I/2 should not be added for nonlocal G
    F = get_formalism(G)
    overlap = basis_integral(G.basis; skip_divergence=true)
    coeff = integral_coeff(Val(F), temperature)
    F === :KF && (coeff /= 2)

    @ein n[i, j] := G.data[i, j, ib] * overlap[ib]
    n = n::Matrix{eltype(G)}
    n .= n .* coeff .+ I(G.norb) ./ 2

    if F === :KF  # select the Keldysh component
        n = reshape(n, G.norb, 2, G.norb, 2)[:, 2, :, 2]
    end

    # Impose Hermiticity
    (n .+ n') ./ 2
end

"""
    compute_occupation(G, temperature=nothing)
Compute the total occupation of the electrons (zero to norb).
"""
compute_occupation(G, temperature=nothing) = real(tr(compute_occupation_matrix(G, temperature)))
