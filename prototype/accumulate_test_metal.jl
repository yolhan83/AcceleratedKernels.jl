
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using Metal

import AcceleratedKernels as AK


Random.seed!(0)


v = Metal.ones(Int32, 100)

v2 = AK.accumulate!(+, copy(v), init=zero(eltype(v)), block_size=1024)

@assert Array(v2) == cumsum(Array(v))

v2
