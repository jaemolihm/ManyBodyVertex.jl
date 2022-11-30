# Copied from DFTK.jl/src/scf/potential_mixing.jl (v0.5.15)
# The only change is to comment out the to_cpu call.

# Quick and dirty Anderson implementation ... lacks important things like
# control of the condition number of the anderson matrix.
# Not particularly optimised. Also should be moved to NLSolve ...
#
# Accelerates the iterative solution of f(x) = 0 according to a
# damped preconditioned scheme
#    xₙ₊₁ = xₙ + αₙ P⁻¹ f(xₙ)
# Where f(x) computes the residual (e.g. SCF(x) - x)
# Further define
#    preconditioned residual  Pf(x) = P⁻¹ f(x)
#    fixed-point map          g(x)  = V + α Pf(x)
# where the α may vary between steps.
#
# Finds the linear combination xₙ₊₁ = g(xₙ) + ∑ᵢ βᵢ (g(xᵢ) - g(xₙ))
# such that |Pf(xₙ) + ∑ᵢ βᵢ (Pf(xᵢ) - Pf(xₙ))|² is minimal
struct AndersonAcceleration
    m::Int                  # maximal history size
    iterates::Vector{Any}   # xₙ
    residuals::Vector{Any}  # Pf(xₙ)
    maxcond::Real           # Maximal condition number for Anderson matrix
end
AndersonAcceleration(;m=10, maxcond=1e6) = AndersonAcceleration(m, [], [], maxcond)

function Base.push!(anderson::AndersonAcceleration, xₙ, αₙ, Pfxₙ)
    push!(anderson.iterates,  vec(xₙ))
    push!(anderson.residuals, vec(Pfxₙ))
    if length(anderson.iterates) > anderson.m
        popfirst!(anderson.iterates)
        popfirst!(anderson.residuals)
    end
    @assert length(anderson.iterates) <= anderson.m
    @assert length(anderson.iterates) == length(anderson.residuals)
    anderson
end

# Gets the current xₙ, Pf(xₙ) and damping αₙ
# JML: Here, we need to solve f(x) = iterate_parquet(x) - x = 0, and use no preconditioning
# P=I. So, we need Pfxₙ = iterate_parquet(xₙ) - xₙ.
function (anderson::AndersonAcceleration)(xₙ, αₙ, Pfxₙ)
    xs   = anderson.iterates
    Pfxs = anderson.residuals

    # Special cases with fast exit
    anderson.m == 0 && return xₙ .+ αₙ .* Pfxₙ
    if isempty(xs)
        push!(anderson, xₙ, αₙ, Pfxₙ)
        return xₙ .+ αₙ .* Pfxₙ
    end

    M = hcat(Pfxs...) .- vec(Pfxₙ)  # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
    # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ

    # Ensure the condition number of M stays below maxcond, else prune the history
    Mfac = qr(M)
    while size(M, 2) > 1 && cond(Mfac.R) > anderson.maxcond
        M = M[:, 2:end]  # Drop oldest entry in history
        popfirst!(anderson.iterates)
        popfirst!(anderson.residuals)
        Mfac = qr(M)
    end

    xₙ₊₁ = vec(xₙ) .+ αₙ .* vec(Pfxₙ)
    βs   = -(Mfac \ vec(Pfxₙ))
    # βs = to_cpu(βs)  # GPU computation only : get βs back on the CPU so we can iterate through it
    for (iβ, β) in enumerate(βs)
        xₙ₊₁ .+= β .* (xs[iβ] .- vec(xₙ) .+ αₙ .* (Pfxs[iβ] .- vec(Pfxₙ)))
    end

    push!(anderson, xₙ, αₙ, Pfxₙ)
    reshape(xₙ₊₁, size(xₙ))
end
