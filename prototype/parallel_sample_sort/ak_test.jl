
import AcceleratedKernels as AK
using Random
Random.seed!(0)

v = rand(1:100, 1_000_000)
AK.sort!(v)
@assert issorted(v)

v = rand(1:100, 1_000_000)
ix = AK.sortperm(v)
@assert issorted(v[ix])
