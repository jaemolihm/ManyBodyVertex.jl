"""
# Spin channel
Under SU(2) symmetry, there are six nonzero components to the vertex, which are related as
`Γ↑↑↑↑ = Γ↓↓↓↓`, `Γ↑↑↓↓ = Γ↓↓↑↑`, `Γ↑↓↓↑ = Γ↓↑↑↓ = Γ↑↑↑↑ - Γ↑↑↓↓`.

We use different spin parametrizations for different channels:
- A: density channel `Γd = Γ↑↑↑↑ + Γ↑↑↓↓` and magnetic channel `Γm = Γ↑↑↑↑ - Γ↑↑↓↓`.
- P: singlet channel `Γs = Γ↑↑↓↓ - Γ↑↓↓↑` and triplet channel `Γt = Γ↑↑↓↓ + Γ↑↓↓↑`.
- T: T-density channel `Γtd = Γ↑↑↑↑ + Γ↑↓↓↑` and T-magnetic channel `Γtm = Γ↑↑↑↑ - Γ↑↓↓↑`.
For each parametrization, vertices in the two channels are stored as a 2-Tuple.

We use the "dm" parametrization for the A and T channel and the "st" parametrization for the
P channel, to make the bubble integral diagonal. We store all the vertices and bubbles as
length-2 tuples. In the P channel, factor of -1 needs to be multiplied to the bubble of the
"s" channel.

Conversion between the two spin parametrizations are done using the
`su2_convert_spin_channel` function.
"""

"""
    su2_convert_spin_channel(C_out, Γ, C_in=channel(Γ[1]))
Convert between the "dm" (for channel A and T) and "st" (for channel P) spin representations.
"""
function su2_convert_spin_channel(C_out::Symbol, Γ, C_in::Symbol=get_channel(Γ[1]))
    if C_in === C_out
        Γ
    elseif (C_in, C_out) === (:A, :P)
        (Γ[1] / 2 - Γ[2] * 3/2, Γ[1] / 2 + Γ[2] / 2)
    elseif (C_in, C_out) === (:P, :A)
        (Γ[1] / 2 + Γ[2] * 3/2, Γ[1] * -1/2 + Γ[2] / 2)
    elseif (C_in, C_out) === (:T, :A) || (C_in, C_out) === (:A, :T)
        (Γ[1] / 2 + Γ[2] * 3/2, Γ[1] / 2 - Γ[2] / 2)
    elseif (C_in, C_out) === (:T, :P) || (C_in, C_out) === (:P, :T)
        (Γ[1] * -1/2 + Γ[2] * 3/2, Γ[1] / 2 + Γ[2] / 2)
    else
        error("Wrong channels $C_out and $C_in")
    end
end

function su2_apply_crossing(Γ)
    get_channel(Γ[1]) === :A || error("su2_apply_crossing implemented only for A -> T")
    apply_crossing.(Γ)
end
su2_apply_crossing(::Nothing) = nothing

function su2_bare_vertex(F::Val, C::Symbol, U::Number, args...)
    # SU2 symmetric bare vertex: +1, -1, 0 for spin channels d, m, p.
    if C === :A  # (d, m)
        (get_bare_vertex(F, C, U, args...), get_bare_vertex(F, C, -U, args...))
    elseif C === :P  # (s, t)
        (get_bare_vertex(F, C, 2 * U, args...),get_bare_vertex(F, C, 0 * U, args...))
    elseif C === :T  # (td, tm)
        (get_bare_vertex(F, C, -U, args...), get_bare_vertex(F, C, U, args...))
    else
        error("Wrong channel $C")
    end
end
