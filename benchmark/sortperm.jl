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
ix = ArrayType(ones(Int, n))


println("\n===\nBenchmarking sortperm! on $n UInt32 - Base vs. AK")
display(@benchmark Base.sortperm!($ix, v) setup=(v = ArrayType(rand(UInt32(1):UInt32(1_000_000), n))))
display(@benchmark AK.sortperm!($ix, v) setup=(v = ArrayType(rand(UInt32(1):UInt32(1_000_000), n))))


println("\n===\nBenchmarking sortperm! on $n Int64 - Base vs. AK")
display(@benchmark Base.sortperm!($ix, v) setup=(v = ArrayType(rand(Int64(1):Int64(1_000_000), n))))
display(@benchmark AK.sortperm!($ix, v) setup=(v = ArrayType(rand(Int64(1):Int64(1_000_000), n))))


println("\n===\nBenchmarking sortperm! on $n Float32 - Base vs. AK")
display(@benchmark Base.sortperm!($ix, v) setup=(v = ArrayType(rand(Float32, n))))
display(@benchmark AK.sortperm!($ix, v) setup=(v = ArrayType(rand(Float32, n))))


println("\n===\nBenchmarking sortperm!(by=sin) on $n Float32 - Base vs. AK")
display(@benchmark Base.sortperm!($ix, v, by=sin) setup=(v = ArrayType(rand(Float32, n))))
display(@benchmark AK.sortperm!($ix, v, by=sin) setup=(v = ArrayType(rand(Float32, n))))

