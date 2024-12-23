module PlatformDependentMetalExt


using Metal
import AcceleratedKernels as AK


# On Metal use the ScanPrefixes accumulation algorithm by default as the DecoupledLookback algorithm
# cannot be supported due to Metal's weaker memory consistency guarantees.
function AK.accumulate!(
    op, v::AbstractArray, backend::MetalBackend;
    init,
    inclusive::Bool=true,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=1024,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    AK._accumulate_impl!(
        op, v, backend,
        init=init, inclusive=inclusive,
        alg=alg,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


end   # module PlatformDependentMetalExt