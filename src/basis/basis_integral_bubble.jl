
"""
    basis_integral_bubble(b1, b2, b3, w, C::Symbol)
Basis integral for calculating bubbles.
```
          1  ->- ±(v+w/2) ->- 4
Π(v, w) = |                   |
          2  -<-   v-w/2  -<- 3
```

# Input
- `b1`: Basis for the fermionic frequency of the vertex
- `b2`: Basis for the first 1-particle propagator
- `b3`: Basis for the second 1-particle propagator
- `w`: Bosonic frequency of the bubble
- `C`: Channel (`:A` or `:P` or `:T`)

# Integral formulas
``overlap = ∫dv b1(v) * b2(v1) * b3(v2)`` where ``v1, v2 = _bubble_frequencies(F, C, v, w)``.
- A, T channal: ``overlap = ∫dw b1(v) * b2(v + w/2) * b3(v - w/2)``
- P channal: ``overlap = ∫dw b1(v) * b2(-v - w/2) * b3(v - w/2)``
"""
function basis_integral_bubble(b1::AbstractImagBasis, b2, b3, w, C::Symbol)
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
        # l3 <= floor(Int, v-w/2) <= r3
        l3 != typemin(Int) && (l3 = l3 - floor(Int, -w/2))
        r3 != typemax(Int) && (r3 = r3 - floor(Int, -w/2))
        if C === :P
            # l2 <= ceil(Int, -v-1-w/2) <= r2
            r2_ = l2 != typemin(Int) ? -l2 - 1 + ceil(Int, -w/2) : typemax(Int)
            l2_ = r2 != typemax(Int) ? -r2 - 1 + ceil(Int, -w/2) : typemin(Int)
            l2, r2 = l2_, r2_
        else
            # l2 <= floor(Int, v+w/2) <= r2
            l2 != typemin(Int) && (l2 = l2 - floor(Int, w/2))
            r2 != typemax(Int) && (r2 = r2 - floor(Int, w/2))
        end
        interval = max(l1, l2, l3) .. min(r1, r2, r3)
        isempty(interval) && continue

        function f(v)
            v1, v2 = _bubble_frequencies(Val(:MF), C, v, w)
            b1[v, i1] * b2[v1, i2] * b3[v2, i3]
        end
        l, r = endpoints(interval)
        res = integrate_imag(f, l, r)
        res !== nothing && (overlap[inds] += res.val)
    end
    overlap
end


function basis_integral_bubble(b1::AbstractRealFreqBasis, b2, b3, w, C::Symbol)
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
        l3, r3 = l3 + w/2, r3 + w/2  # l3 <= v - w/2 <= r3
        if C === :P
            l2, r2 = -r2 - w/2, -l2 - w/2  # l2 <= -v - w/2 <= r2
        else
            l2, r2 = l2 - w/2, r2 - w/2  # l2 <= v + w/2 <= r2
        end
        interval = max(l1, l2, l3) .. min(r1, r2, r3)
        isempty(interval) && continue

        function f(v)
            v1, v2 = _bubble_frequencies(Val(:KF), C, v, w)
            b1[v, i1] * b2[v1, i2] * b3[v2, i3]
        end
        l, r = endpoints(interval)
        (l >= r || l ≈ r) && continue
        val, err = quadgk(f, l, r)
        overlap[inds] += val
    end
    overlap
end
