"""
    basis_integral_self_energy(b1, b2, b3, v, C::Symbol)
Basis integral for calculating self-energy
```
       -<- ±(v+w) -<-
      /            \
      2  ---------  3
      |             |
v -<- 1  ---------  4 -<- v
```

# Input
- `b1`: Basis for the fermionic frequency of the vertex
- `b2`: Basis for the bosonic frequency of the vertex
- `b3`: Basis for the 1-particle propagator

# Integral formulas
- A, T channal: ``overlap = ∫dw b1(v + w/2) * b2(w) * b3(v + w)``
- P channal: ``overlap = ∫dw b1(v + w/2) * b2(w) * b3(-v - w)``

For `b1` with Matsubara frequencies, the bound `a <= (v + w/2) <= b` becomes
`2(a - v) - 1 <= w <= 2(b - v)` (see `_bubble_frequencies_inv`: `v + w/2` is implemented as
`_bubble_frequencies_inv(v, v+w)[1] = fld(v + v + w + 1, 2)`).
"""
function basis_integral_self_energy(b1::AbstractImagBasis, b2, b3, v, C::Symbol)
    @assert C ∈ (:A, :P, :T)

    bases = (b1, b2, b3)
    sizes = nbasis.(bases)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        i1, i2, i3 = inds.I
        # Integrate the function only on the interval where the function is nonzero.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        l1, r1 = endpoints(support_bounds(b1, i1))
        l2, r2 = endpoints(support_bounds(b2, i2))
        l3, r3 = endpoints(support_bounds(b3, i3))
        l1 != typemin(Int) && (l1 = 2 * (l1 - v) - 1)
        r1 != typemax(Int) && (r1 = 2 * (r1 - v))
        if C === :P
            # l3 <= -v - w - 1 <= r3
            r3_ = l3 != typemin(Int) ? -l3 - v - 1 : typemax(Int)
            l3_ = r3 != typemax(Int) ? -r3 - v - 1 : typemin(Int)
            l3, r3 = l3_, r3_
        else
            # l3 <= v + w <= r3
            l3 != typemin(Int) && (l3 = l3 - v)
            r3 != typemax(Int) && (r3 = r3 - v)
        end
        interval = max(l1, l2, l3) .. min(r1, r2, r3)
        isempty(interval) && continue

        function f(w)
            v_w_over_2 = fld(2v + w + 1, 2)
            if C === :P
                b1[v_w_over_2, i1] * b2[w, i2] * b3[-v-w-1, i3]
            else
                b1[v_w_over_2, i1] * b2[w, i2] * b3[v+w, i3]
            end
        end
        l, r = interval.left, interval.right
        res = integrate_imag(f, l, r)
        res !== nothing && (overlap[inds] += res.val)
    end
    overlap
end


function basis_integral_self_energy(b1::AbstractRealFreqBasis, b2, b3, v, C::Symbol)
    @assert C ∈ (:A, :P, :T)

    bases = (b1, b2, b3)
    sizes = nbasis.(bases)
    overlap = zeros(eltype(bases[1]), sizes)
    for inds in CartesianIndices(sizes)
        i1, i2, i3 = inds.I
        # Integrate the function only on the interval where the function is nonzero.
        # Directly integrating over (-Inf, Inf) is inefficient, and may even become
        # inaccurate when the support is narrow.
        l1, r1 = endpoints(support_bounds(b1, i1))
        l2, r2 = endpoints(support_bounds(b2, i2))
        l3, r3 = endpoints(support_bounds(b3, i3))
        l1 = 2 * (l1 - v)
        r1 = 2 * (r1 - v)
        if C === :P
            # l3 <= -v - w <= r3
            l3, r3 = -r3 - v, -l3 - v
        else
            # l3 <= v + w <= r3
            l3 = l3 - v
            r3 = r3 - v
        end
        interval = max(l1, l2, l3) .. min(r1, r2, r3)
        isempty(interval) && continue

        function f(w)
            if C === :P
                b1[v + w/2, i1] * b2[w, i2] * b3[-v-w, i3]
            else
                b1[v + w/2, i1] * b2[w, i2] * b3[v+w, i3]
            end
        end
        l, r = interval.left, interval.right
        (l >= r || l ≈ r) && continue
        val, err = quadgk(f, l, r)
        overlap[inds] += val
    end
    overlap
end
