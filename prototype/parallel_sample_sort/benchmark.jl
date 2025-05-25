
import AcceleratedKernels as AK
using BenchmarkTools

using Profile
using PProf

using Random
Random.seed!(0)


# Compile
v = rand(1_000_000)
AK.sort!(v)


# Collect a profile
Profile.clear()
# v = rand(1_000_000)
# @profile AK.sort!(v)

v = rand(UInt32(0):UInt32(1_000_000), 1_000_000)
ix = Vector{Int}(undef, 1_000_000)
@profile AK.sortperm!(ix, v)
pprof()


println("\nBase vs AK sort (Int):")
display(@benchmark Base.sort!(v) setup=(v = rand(1:1_000_000, 1_000_000)))
display(@benchmark AK.sort!(v) setup=(v = rand(1:1_000_000, 1_000_000)))


println("\nBase vs AK sort (Float64):")
display(@benchmark Base.sort!(v) setup=(v = rand(Float64, 1_000_000)))
display(@benchmark AK.sort!(v) setup=(v = rand(Float64, 1_000_000)))


println("\nBase vs AK sortperm (UInt32):")
display(@benchmark Base.sortperm!(ix, v) setup=(v = rand(UInt32(0):UInt32(1_000_000), 1_000_000); ix = Vector{Int}(undef, 1_000_000)))
display(@benchmark AK.sortperm!(ix, v) setup=(v = rand(UInt32(0):UInt32(1_000_000), 1_000_000); ix = Vector{Int}(undef, 1_000_000)))


println("\nBase vs AK sortperm (Float64):")
display(@benchmark Base.sortperm!(ix, v) setup=(v = rand(Float64, 1_000_000); ix = Vector{Int}(undef, 1_000_000)))
display(@benchmark AK.sortperm!(ix, v) setup=(v = rand(Float64, 1_000_000); ix = Vector{Int}(undef, 1_000_000)))
