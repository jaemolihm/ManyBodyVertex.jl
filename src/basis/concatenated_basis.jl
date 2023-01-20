struct RealConcatenatedBasis{T, B1, B2} <: AbstractRealFreqBasis{T}
    basis1::B1
    basis2::B2
    function RealConcatenatedBasis(b1::B1, b2::B2) where {B1 <: Basis{T}, B2 <: Basis{T}} where {T}
        new{T, B1, B2}(b1, b2)
    end
end

struct ImagConcatenatedBasis{T, B1, B2} <: AbstractImagBasis{T}
    basis1::B1
    basis2::B2
    function ImagConcatenatedBasis(b1::B1, b2::B2) where {B1 <: Basis{T}, B2 <: Basis{T}} where {T}
        new{T, B1, B2}(b1, b2)
    end
end

const _ConcatenatedBasis{T} = Union{ImagConcatenatedBasis{T}, RealConcatenatedBasis{T}} where {T}

ntails(f::_ConcatenatedBasis) = ntails(f.basis1) + ntails(f.basis2)
nbasis(f::_ConcatenatedBasis) = nbasis(f.basis1) + nbasis(f.basis2)

@inline function index_concatenated_basis(f::_ConcatenatedBasis, n::Integer)
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n ∈ axes(f.basis1, 2)
        (f.basis1, n)
    else
        (f.basis2, n - nbasis(f.basis1))
    end
end

@inline function support_bounds(f::_ConcatenatedBasis, n::Integer)
    basis, i = index_concatenated_basis(f, n)
    support_bounds(basis, i)
end
@inline function Base.getindex(f::_ConcatenatedBasis{T}, x::Number, n::Integer) where {T}
    @boundscheck x ∈ axes(f, 1) || throw(BoundsError())
    @boundscheck n ∈ axes(f, 2) || throw(BoundsError())
    if n ∈ axes(f.basis1, 2)
        getindex(f.basis1, x, n)
    else
        n -= nbasis(f.basis1)
        zero(T)
        T(getindex(f.basis2, x, n))::T
    end
end

function concat_constant_basis(basis::AbstractRealFreqBasis{T}) where {T}
    RealConcatenatedBasis(ConstantBasis(T), basis)
end
function concat_constant_basis(basis::AbstractImagBasis{T}) where {T}
    ImagConcatenatedBasis(ImagConstantBasis(T), basis)
end
