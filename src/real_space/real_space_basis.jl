using LinearAlgebra
using StaticArrays

"""
# Real-space basis
By Fourier transforming the vertex in the A channel representation from the momentum space
to the real space, we can define lattice vectors in the channel representation:
```
  Γ(k, k'; q)
= Γ(k-q/2, -k-q/2, k'+q/2, -k'+q/2)
= 1/N_R ∑_{R1, R2, R3, R4} exp(i(R1, R2, R3, R4) • (k-q/2, -k-q/2, k'+q/2, -k'+q/2)) Γ(R1, R2, R3, R4)
= ∑_{R, R', R_B} exp(ikR) * exp(-ik'R') * exp(iq(R_B + (R - R')/2)) * Γ(R, R'; R_B)
```
where `R = R2 - R1`, `R' = R3 - R4`, and `R_B = R1 - R4`.
These relations and its inverse are implemented in `lattice_vectors_to_standard` and
`lattice_vectors_to_channel`.
(The subscript `B` in `R_B` stands for "bosonic".)

Real-space representations in the P and T channels are identical except that the lattice
vectors R1, R2, R3, R4 needs to be reordered according to `indices_to_channel`.

Physically, `Γ(R, R'; R_B)` describes a correlation between the one density operator
`d(R_B) d†(R_B + R)` and another operator `d(R') d†(0)`, which are separated apart by `R_B`.
We call each pair of creation/annihilation operators a "bond", as it describes two sites
(e.g. for `d(R_B) d†(R_B + R)`, one at `R_B` and the other at `R_B + R`). We use the bonds
as the basis for the truncation-of-unity expansion of the fermionic momenta of the vertices
and the bubbles.

## Minimal-distance replica selection for the bosonic lattice vector
For the Fourier transformation of the bosonic frequency, one can use minimal-distance
replica selection to determine which `R_B` should be nonzero. For that, we use the distance
between the center of the two bonds. Including the atomic positions, the distance is
``R_B + (R + τ₁ + τ₂) / 2 - (R' + τ₃ + τ₄) / 2``. The replica search is implemented in
`find_minimal_distance_replica`.
"""

# TODO: Truncation for choosing the bond bases

const Bond{Dim} = Tuple{Int, Int, SVector{Dim, Int}}

function lattice_vectors_to_standard(C::Val, R, Rp, R_B)
    # Uniform translation of (R1, R2, R3, R4) is meaningless. We arbitraily choose R4 to
    # be zero. All operations should only use the differences between these vectors.
    R4 = zero(R)
    R1 = R_B + R4
    R3 = Rp + R4
    R2 = R + R1
    indices_to_standard(C, (R1, R2, R3, R4))
end

function lattice_vectors_to_channel(C::Val, R1234)
    R1, R2, R3, R4 = indices_to_channel(C, R1234)
    R = R2 - R1
    Rp = R3 - R4
    R_B = R1 - R4
    (; R, Rp, R_B)
end

"""
    RealSpaceBasis{Dim, T}
Use outer constructor `RealSpaceBasis(lattice, positions, bonds_L, bonds_R, qgrid)`.
- `Dim`: The physical dimension of the system
- `T`: Floating-point type for representing the lattice and the q point vectors.

# Fields
- `lattice`: lattice vector. `lattice[:, i]` is the Cartesian coordinates of the i-th lattice vector.
- `positions`: atomic positions in reduced coordinates
- `bonds_L`, `bonds_R`: list of bond bases vectors to use for the expansion of fermionic momenta.
- `qgrid`: size of the q-point grid
- `qpts`: list of q vectors in reciprocal reduced coordinates
- `R_B_list`: bosonic lattice vectors.
- `R_B_replicas`: list of lattice vectors to be used for Fourier transformation.
    `R_B_replicas[ibL, ibR][:]` is the list of minimal-distance replicas of
    `R_B = R_B_list[iRB]` for bonds `bonds_L[ibL]` and `bonds_R[ibR]`.
- `R_B_replica_inds`: index of original `R_B` in `R_B_list`. `R_B_replicas[ibL, ibR][i]` is
    a replica of `R_B_list[R_B_replica_inds[ibL, ibR][i]`.
- `R_B_ndegen`: degeneracy of the R_B replica. Need to be divided in the Fourier
    transformation R → q.
"""
struct RealSpaceBasis{Dim, T}
    lattice::SMatrix{Dim, Dim, T}
    positions::Vector{SVector{Dim, T}}
    bonds_L::Vector{Bond{Dim}}
    bonds_R::Vector{Bond{Dim}}
    qgrid::NTuple{Dim, Int}
    qpts::Vector{SVector{Dim, T}}
    R_Bs::Vector{SVector{Dim, Int}}
    R_B_replicas::Matrix{Vector{SVector{Dim, Int}}}
    R_B_replica_inds::Matrix{Vector{Int}}  # TODO: Remove
    R_B_ndegen::Matrix{Vector{Int}}
end

function Base.show(io::IO, rbasis::RealSpaceBasis{Dim}) where {Dim}
    println(io, "RealSpaceBasis($(Dim)D)")
    println(io, "    lattice : $(rbasis.lattice)")
    println(io, "    bonds_L : $(length(rbasis.bonds_L)) bonds")
    println(io, "    bonds_R : $(length(rbasis.bonds_R)) bonds")
    println(io, "    qgrid   : $(rbasis.qgrid)")
end

function RealSpaceBasis(lattice::SMatrix{Dim, Dim}, positions, bonds_L, bonds_R, qgrid) where {Dim}
    natoms = length(positions)
    all(1 .<= getindex.(bonds_L, 1) .<= natoms) || error("Wrong 1st atomic index of bonds_L")
    all(1 .<= getindex.(bonds_L, 2) .<= natoms) || error("Wrong 2nd atomic index of bonds_L")
    all(1 .<= getindex.(bonds_R, 1) .<= natoms) || error("Wrong 1st atomic index of bonds_R")
    all(1 .<= getindex.(bonds_R, 2) .<= natoms) || error("Wrong 2nd atomic index of bonds_R")

    lattice_ = convert.(AbstractFloat, lattice)
    qpts = vec([SVector(x ./ qgrid) for x in Iterators.product(range.(0, qgrid .- 1)...)])
    R_Bs, R_B_replicas, R_B_replica_inds, R_B_ndegen = find_minimal_distance_replica(qgrid, lattice_,
        positions, bonds_L, bonds_R; nsearch=3)
    RealSpaceBasis(lattice_, positions, bonds_L, bonds_R, qgrid, qpts, R_Bs, R_B_replicas,
        R_B_replica_inds, R_B_ndegen)
end

function center_position(bond::Bond, positions)
    i1, i2, R = bond
    (positions[i1] + positions[i2] + R) / 2
end
center_position(iatm::Integer, positions) = positions[iatm]

"""
    find_minimal_distance_replica(qgrid, lattice, bonds_L, bonds_R, positions; nsearch=3)
Find the minimal-distance replica of R_B in the supercell of size `qgrid`.
Minimize `|R_B + R_center_L - R_center_R|` where `R_center` is the center of the bonds.

# Inputs
- `qgrid`: Size of the momentum-space grid for which the corresponding real-space vectors are found.
- `lattice`: lattice vectors
- `positions`: atomic positions in the reduced coordinates
- `bonds_L`, `bonds_R`: left and right bond bases
- `nsearch=3`: Size of the supercell to search the replica (-nsearch:nsearch)
"""
function find_minimal_distance_replica(qgrid, lattice::SMatrix{Dim, Dim, T}, positions,
                                       bonds_L, bonds_R; nsearch=3) where {Dim, T}
    R_super_list = vec([SVector(x .* qgrid) for x in Iterators.product(
            ntuple(x -> -nsearch:nsearch, Val(Dim))...)])

    R_Bs = vec(SVector.(Iterators.product(range.(0, qgrid .- 1)...)))

    R_B_replicas = [SVector{Dim, Int}[] for _ in eachindex(bonds_L), _ in eachindex(bonds_R)]
    R_B_replica_inds = [Int[] for _ in eachindex(bonds_L), _ in eachindex(bonds_R)]
    R_B_ndegen = [Int[] for _ in eachindex(bonds_L), _ in eachindex(bonds_R)]

    distances = zeros(T, length(R_super_list))
    is_minimal_distance = falses(length(R_super_list))
    for ibR in eachindex(bonds_R), ibL in eachindex(bonds_L)
        ΔR = center_position(bonds_L[ibL], positions) - center_position(bonds_L[ibR], positions)

        # Find all R_super that minimize norm(lattice * (R_B + R_super + ΔR))
        for (iR_B, R_B) in enumerate(R_Bs)
            distances .= norm.(Ref(lattice) .* (R_super_list .+ Ref(R_B + ΔR)))
            distance_min = minimum(distances)
            is_minimal_distance .= distances .<= distance_min * (1 + sqrt(eps(T)))
            ndegen = sum(is_minimal_distance)
            append!(R_B_replicas[ibL, ibR], Ref(R_B) .+ R_super_list[is_minimal_distance])
            append!(R_B_replica_inds[ibL, ibR], fill(iR_B, ndegen))
            append!(R_B_ndegen[ibL, ibR], fill(ndegen, ndegen))
        end
    end
    R_Bs, R_B_replicas, R_B_replica_inds, R_B_ndegen
end

"""
    get_indices(rbasis::RealSpaceBasis)
Return list of real-space indices `(ibL, ibR, iR_B)`.
"""
function get_indices(rbasis::RealSpaceBasis)
    mapreduce(vcat, Iterators.product(eachindex(rbasis.bonds_L), eachindex(rbasis.bonds_R))) do (ibL, ibR)
        [(ibL, ibR, iR_B) for iR_B in eachindex(rbasis.R_B_replicas[ibL, ibR])]
    end
end
