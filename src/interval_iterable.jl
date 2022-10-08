using ContinuumArrays: Inclusion
using IntervalSets: AbstractInterval
using DomainSets: UnionDomain, components, EmptySpace, Point
using QuasiArrays: domain

"""
    interval_iterable(x::Inclusion)
Return the list of intervals that belong to `domain(x)` in an iterable form.
"""
interval_iterable(x::Inclusion) = interval_iterable(domain(x))
interval_iterable(x::AbstractInterval) = (x,)
interval_iterable(x::UnionDomain) = components(x)
interval_iterable(x::Point) = (boundingbox(x),)
interval_iterable(x::EmptySpace) = ()

# Fix DomainSets.jl
# See https://github.com/JuliaApproximation/DomainSets.jl/issues/121
using DomainSets
DomainSets.intersectdomain1(d1::Point, d2::Domain) = d1.x ∈ d2 ? d1 : EmptySpace{eltype(d1)}()
DomainSets.intersectdomain1(d1::Domain, d2::Point) = d2.x ∈ d1 ? d2 : EmptySpace{eltype(d2)}()
