abstract type AbstractFrequencyVertex{F, T} end
Base.eltype(::Type{<:AbstractFrequencyVertex{F, T}}) where {F, T} = T
get_formalism(::Type{<:AbstractFrequencyVertex{F, T}}) where {F, T} = F
get_formalism(::T) where {T <: AbstractFrequencyVertex} = get_formalism(T)
nkeldysh(F::Symbol) = F === :KF ? 2 : 1
nkeldysh(::AbstractFrequencyVertex{F}) where {F} = nkeldysh(F)

data_fieldnames(::T) where {T <: AbstractFrequencyVertex} = data_fieldnames(T)

function Base.:+(A::T, B::T) where {T <: AbstractFrequencyVertex}
    _check_basis_identity(A, B)
    C = similar(A)
    for name in data_fieldnames(A)
        getproperty(C, name) .= getproperty(A, name) .+ getproperty(B, name)
    end
    C
end

function Base.:-(A::T, B::T) where {T <: AbstractFrequencyVertex}
    _check_basis_identity(A, B)
    C = similar(A)
    for name in data_fieldnames(A)
        getproperty(C, name) .= getproperty(A, name) .- getproperty(B, name)
    end
    C
end

function Base.:*(x::Number, A::T) where {T <: AbstractFrequencyVertex}
    B = similar(A)
    for name in data_fieldnames(A)
        getproperty(B, name) .= getproperty(A, name) .* x
    end
    B
end
function Base.:/(A::T, x::Number) where {T <: AbstractFrequencyVertex}
    B = similar(A)
    for name in data_fieldnames(A)
        getproperty(B, name) .= getproperty(A, name) ./ x
    end
    B
end
Base.:*(A::T, x::Number)  where {T <: AbstractFrequencyVertex} = x * A

function Base.copy(A::T) where {T <: AbstractFrequencyVertex}
    B = similar(A)
    for name in data_fieldnames(A)
        getproperty(B, name) .= getproperty(A, name)
    end
    B
end
