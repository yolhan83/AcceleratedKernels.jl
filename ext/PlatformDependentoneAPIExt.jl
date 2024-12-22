module PlatformDependentoneAPIExt


using oneAPI
import AcceleratedKernels as AK


# On oneAPI, use `cooperative=false` by default as on some Intel GPUs concurrent global writes hang
# the device.
function AK.any(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=false,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    AK._any_impl(
        pred, v, backend;
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
        cooperative=cooperative,
        temp=temp,
        switch_below=switch_below,
    )
end


function AK.all(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=false,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    AK._all_impl(
        pred, v, backend;
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
        cooperative=cooperative,
        temp=temp,
        switch_below=switch_below,
    )
end


end   # module PlatformDependentoneAPI