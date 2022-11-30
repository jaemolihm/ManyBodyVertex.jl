"""
# Spin channel
Under SU(2) symmetry, there are six nonzero components to the vertex, which are related as
`Γ↑↑↑↑ = Γ↓↓↓↓`, `Γ↑↑↓↓ = Γ↓↓↑↑`, `Γ↑↓↓↑ = Γ↓↑↑↓ = Γ↑↑↑↑ - Γ↑↑↓↓`.

We use two types of parametrizations.
- "dm": density channel `Γd = Γ↑↑↑↑ + Γ↑↑↓↓` and magnetic channel `Γm = Γ↑↑↑↑ - Γ↑↑↓↓`.
- "st": singlet channel `Γs = Γ↑↑↓↓ - Γ↑↓↓↑` and magnetic channel `Γt = Γ↑↑↓↓ + Γ↑↓↓↑`.
For each parametrization, vertices in the two channels are stored as a 2-Tuple.

We use the "dm" parametrization for the A and T channel and the "st" parametrization for the
P channel, to make the bubble integral diagonal. We store all the vertices and bubbles as
length-2 tuples. In the P channel, factor of -1 needs to be multiplied to the bubble of the
"s" channel.

Conversion between the two spin parametrizations are done using the
`su2_convert_spin_channel` function.
"""

"""
    su2_convert_spin_channel(C_out, Γ)
Convert between the "dm" (for channel A and T) and "st" (for channel P) spin representations.
"""
function su2_convert_spin_channel(C_out::Symbol, Γ)
    C_in = channel(Γ[1])
    if C_in ∈ (:A, :T) && C_out == :P  # dm -> st
        (Γ[1] / 2 - Γ[2] * 3/2, Γ[1] / 2 + Γ[2] / 2)
    elseif C_in == :P && C_out ∈ (:A, :T)  # st -> dm
        (Γ[1] / 2 + Γ[2] * 3/2, Γ[1] * -1/2 + Γ[2] / 2)
    else
        Γ
    end
end

function su2_apply_crossing(Γ)
    channel(Γ[1]) === :A || error("su2_apply_crossing implemented only for A -> T")
    Γa_d, Γa_m = Γ
    Γt_d = 1/2 * apply_crossing(Γa_d) + 3/2 * apply_crossing(Γa_m)
    Γt_m = 1/2 * apply_crossing(Γa_d) - 1/2 * apply_crossing(Γa_m)
    Γt_d, Γt_m
end
su2_apply_crossing(::Nothing) = nothing

function su2_bare_vertex(U::Number, F::Val, C::Val)
    # SU2 symmetric bare vertex: +1, -1, 0 for spin channels d, m, p.
    if C === Val(:A) || C === Val(:T)  # (d, m)
        (get_bare_vertex(U, F, C), -1 * get_bare_vertex(U, F, C))
    elseif C === Val(:P)  # (s, t)
        (2 * get_bare_vertex(U, F, C), 0 * get_bare_vertex(U, F, C))
    else
        error("Wrong channel $C")
    end
end
