"""
    basis_overlap_analytic_continuation(basis_KF, basis_MF, temperature)
Compute ``∫dw' basis_KF(w') / (iw - w')`` and use fit_basis_coeff for `basis_MF`.
"""
function basis_overlap_analytic_continuation(basis_KF, basis_MF, temperature)
    ns_MF = get_fitting_points(basis_MF)
    if basis_MF.particle_type === :Fermion
        ws_MF = @. (ns_MF + 1/2) * 2π * temperature
    else
        ws_MF = @. ns_MF * 2π * temperature
    end

    overlap = zeros(complex(eltype(basis_KF)), nbasis(basis_KF), length(ws_MF))
    for ib in 1:nbasis(basis_KF)
        l, r = endpoints(support_bounds(basis_KF, ib))
        for (iw, w_MF) in enumerate(ws_MF)
            if w_MF == 0 && l <= 0 <= r
                # Special cases for zero KF and MF frequencies (only for bosons)
                if l == 0
                    basis_KF[0, ib] ≈ 0 || error("basis not zero at endpoint")
                    val, err = quadgk(w -> (basis_KF[w, ib] - basis_KF[0, ib]) / (-w), 0, r)
                elseif r == 0
                    val, err = quadgk(w -> (basis_KF[w, ib] - basis_KF[0, ib]) / (-w), l, 0)
                else  # l != 0, r != 0
                    val1, err1 = quadgk(w -> (basis_KF[w, ib] - basis_KF[0, ib]) / (-w), l, 0)
                    val2, err2 = quadgk(w -> (basis_KF[w, ib] - basis_KF[0, ib]) / (-w), 0, r)
                    val3 = basis_KF[0, ib] * log(abs(r / l))
                    val = val1 + val2 + val3
                end
            else
                val, err = quadgk(w -> basis_KF[w, ib] / (im * w_MF - w), l, r)
            end
            overlap[ib, iw] = val
        end
    end
    fit_basis_coeff(overlap, basis_MF, ns_MF, 2)
end

"""
    analytic_continuation_KF_to_MF(G_KF::Green2P{:KF}, basis_MF, temperature, type)
Apply analytic continuation for a 2-point correlator (Green function) or vertex
(self-energy) to convert from Keldysh to Matsubara formalism.
# Inputs
-`G_KF`: correlator in the Keldysh formalism.
-`basis_MF`: basis to use for the output Matsubara formalism correlator.
-`temperature`: temperature to use for the Matsubara formalism.
-`type`: `:Green` or `:Vertex`.
"""
function analytic_continuation_KF_to_MF(G_KF::Green2P{:KF}, basis_MF, temperature, type)
    type ∈ (:Green, :Vertex) || error("Wrong type $type, must be :Green or :Vertex.")
    overlap = basis_overlap_analytic_continuation(G_KF.basis, basis_MF, temperature)

    G_R = keldyshview(G_KF)[2, 1, :, :, :]
    G_A = keldyshview(G_KF)[1, 2, :, :, :]
    spectral_function = imag.(G_A .- G_R) ./ 2π

    # For vertex (self-energy), definition of G_A and G_R are reversed, so we need -1 sign.
    type === :Vertex && (spectral_function .*= -1)

    norb = G_KF.norb
    data_MF = reshape(reshape(spectral_function, norb^2, :) * overlap, norb, norb, :)
    offset = reshape(G_KF.offset, norb, 2, norb, 2)[:, 2, :, 1]
    Green2P{:MF}(basis_MF, norb, data_MF, offset)
end
