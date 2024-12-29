module AcceleratedKernelsMetalExt


using Metal
import AcceleratedKernels as AK


# On Metal use the ScanPrefixes accumulation algorithm by default as the DecoupledLookback algorithm
# cannot be supported due to Metal's weaker memory consistency guarantees.
function AK.accumulate!(
    op, v::AbstractArray, backend::MetalBackend;
    init,
    dims::Union{Nothing, Int}=nothing,
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
        init=init, dims=dims, inclusive=inclusive,
        alg=alg,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


# Two-argument version for compatibility with Base.accumulate!
function AK.accumulate!(
    op, dst::AbstractArray, src::AbstractArray, backend::MetalBackend;
    init,
    dims::Union{Nothing, Int}=nothing,
    inclusive::Bool=true,

    # Algorithm choice
    alg::AK.AccumulateAlgorithm=AK.ScanPrefixes(),

    # GPU settings
    block_size::Int=1024,
    temp::Union{Nothing, AbstractArray}=nothing,
    temp_flags::Union{Nothing, AbstractArray}=nothing,
)
    copyto!(dst, src)
    AK._accumulate_impl!(
        op, dst, backend,
        init=init, dims=dims, inclusive=inclusive,
        alg=alg,
        block_size=block_size, temp=temp, temp_flags=temp_flags,
    )
end


end   # module AcceleratedKernelsMetalExt