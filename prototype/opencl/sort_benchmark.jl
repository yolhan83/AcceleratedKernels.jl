using BenchmarkTools
using Random

using OpenCL, pocl_jll
import AcceleratedKernels as AK


Random.seed!(0)
OpenCL.versioninfo()


# Generate random numbers
n = 10_000_000
d = CLArray{Int64}(undef, n);


function aksort!(d, temp)
    AK.sort!(d, temp=temp, block_size=512)
    AK.synchronize(AK.get_backend(d))
    d
end


println("AcceleratedKernels Sort:")
temp = similar(d)
display(@benchmark aksort!($d, temp) setup=(rand!(d)))


println("Base Sort:")
dh = Array(d)
temph = Array(temp)
display(@benchmark aksort!($dh, temph) setup=(rand!(dh)))


# println("BUC / CUDA Thrust Sort:")
# display(@benchmark buc_sort!($d) setup=(rand!(d)))

