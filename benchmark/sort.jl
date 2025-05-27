import AcceleratedKernels as AK
using KernelAbstractions

using BenchmarkTools
using Random
Random.seed!(0)


# Choose the Array backend:
#
# using CUDA
# const ArrayType = CuArray
#
# using AMDGPU
# const ArrayType = ROCArray
#
# using oneAPI
# const ArrayType = oneArray
#
# using Metal
# const ArrayType = MtlArray
#
# using OpenCL
# const ArrayType = CLArray
#
const ArrayType = Array


println("Using ArrayType: ", ArrayType)


n = 1_000_000


# Memory-bound, so not much improvement expected when multithreading
println("\n===\nBenchmarking sort! on $n UInt32 - Base vs. AK")
display(@benchmark Base.sort!(v) setup=(v = ArrayType(rand(UInt32(1):UInt32(1_000_000), n))))
display(@benchmark AK.sort!(v) setup=(v = ArrayType(rand(UInt32(1):UInt32(1_000_000), n))))


# Lexicographic sorting of tuples - more complex comparators
ntup = 5
println("\n===\nBenchmarking sort! on $n NTuple{$ntup, Int64} - Base vs. AK")
display(@benchmark Base.sort!(v) setup=(v = ArrayType(rand(NTuple{ntup, Int64}, n))))
display(@benchmark AK.sort!(v) setup=(v = ArrayType(rand(NTuple{ntup, Int64}, n))))


# Memory-bound again
println("\n===\nBenchmarking sort! on $n Float32 - Base vs. AK")
display(@benchmark Base.sort!(v) setup=(v = ArrayType(rand(Float32, n))))
display(@benchmark AK.sort!(v) setup=(v = ArrayType(rand(Float32, n))))


# More complex by=sin
println("\n===\nBenchmarking sort!(by=sin) on $n Float32 - Base vs. AK")
display(@benchmark Base.sort!(v, by=sin) setup=(v = ArrayType(rand(Float32, n))))
display(@benchmark AK.sort!(v, by=sin) setup=(v = ArrayType(rand(Float32, n))))

