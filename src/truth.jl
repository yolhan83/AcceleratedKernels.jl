@kernel cpu=false inbounds=true function _any_global!(out, pred, @Const(v))
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    # Technically this is a race, but it doesn't matter as all threads would write the same value.
    # For example, CUDA F4.2 says "If a non-atomic instruction executed by a warp writes to the
    # same location in global memory for more than one of the threads of the warp, only one thread
    # performs a write and which thread does it is undefined."
    temp[0x1] = 0x0
    @synchronize()

    # The ndrange check already protects us from out of bounds access
    if pred(v[i])
        temp[0x1] = 0x1
    end

    @synchronize()
    if temp[0x1] != 0x0
        out[0x1] = 0x1
    end
end


"""
    any(
        pred, v::AbstractArray, backend::Backend=get_backend(v);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        cooperative::Bool=true,

        # GPU settings passed to mapreduce, only used if cooperative=false
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Check if any element of `v` satisfies the predicate `pred` (i.e. some `pred(v[i]) == true`).
Optimised differently to `mapreduce` due to shortcircuiting behaviour of booleans.

**Other names**: not often implemented standalone on GPUs, typically included as part of a
reduction.

On the CPU, parallelisation is only worth it for large arrays, relatively expensive predicates,
and/or rare occurrence of true; use `max_tasks` and `min_elems` to only use parallelism when worth
it in your application. When only one thread is needed, there is no overhead.

On the GPU, the `cooperative` flag controls whether to use an optimised implementation which works
by concurrent writing to a global flag; there is only one platform we are aware of (old Intel UHD
620 integrated graphics cards) where such writes hang. In such cases, set `cooperative=false` to
use a `mapreduce` implementation, for which you can also use `temp` and `switch_below`.

# Platform-Specific Notes
On oneAPI, `cooperative=false` is the default as on some Intel GPUs concurrent global writes hang
the device.

# Examples
```julia
import AcceleratedKernels as AK
using CUDA

v = CuArray(rand(Float32, 100_000))
AK.any(x -> x < 1, v)
```
"""
function any(
    pred, v::AbstractArray, backend::Backend=get_backend(v);

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=true,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    _any_impl(
        pred, v, backend;
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
        cooperative=cooperative,
        temp=temp,
        switch_below=switch_below,
    )
end


function _any_impl(
    pred, v::AbstractArray, backend::Backend;

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=true,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if backend isa GPU
        @argcheck block_size > 0

        # Some platforms crash when multiple threads write to the same memory location in a global
        # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
        # CUDA). If not cooperative, we need to do a mapreduce
        if cooperative
            out = KernelAbstractions.zeros(backend, Int8, 1)
            _any_global!(backend, block_size)(out, pred, v, ndrange=length(v))
            outh = @allowscalar(out[1])
            return outh == 0 ? false : true
        else
            # FIXME: pass the backend when added
            return mapreduce(
                pred,
                (x, y) -> x || y,
                v,
                backend;
                init=false,
                block_size=block_size,
                temp=temp,
                switch_below=switch_below,
            )
        end
    else
        overall = false
        task_partition(length(v), max_tasks, min_elems) do irange
            for i in irange
                if pred(v[i])
                    # Again, this is technically a thread race, but it doesn't matter as all threads
                    # would write the same value; no data corruption can occur
                    overall = true
                    break
                end
            end
        end
        return overall
    end
end




"""
    all(
        pred, v::AbstractArray, backend::Backend=get_backend(v);

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # GPU settings
        block_size::Int=256,
        cooperative::Bool=true,

        # GPU settings passed to mapreduce, only used if cooperative=false
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    )

Check if all elements of `v` satisfy the predicate `pred` (i.e. all `pred(v[i]) == true`).
Optimised differently to `mapreduce` due to shortcircuiting behaviour of booleans.

**Other names**: not often implemented standalone on GPUs, typically included as part of a
reduction.

On the CPU, parallelisation is only worth it for large arrays, relatively expensive predicates,
and/or rare occurrence of false; use `max_tasks` and `min_elems` to only use parallelism when worth
it in your application. When only one thread is needed, there is no overhead.

On the GPU, the `cooperative` flag controls whether to use an optimised implementation which works
by concurrent writing to a global flag; there is only one platform we are aware of (old Intel UHD
620 integrated graphics cards) where such writes hang. In such cases, set `cooperative=false` to
use a `mapreduce` implementation, for which you can also use `temp` and `switch_below`.

# Platform-Specific Notes
On oneAPI, `cooperative=false` is the default as on some Intel GPUs concurrent global writes hang
the device.

# Examples
```julia
import AcceleratedKernels as AK
using Metal

v = MtlArray(rand(Float32, 100_000))
AK.all(x -> x > 0, v)
````
"""
function all(
    pred, v::AbstractArray, backend::Backend=get_backend(v);

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=true,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    _all_impl(
        pred, v, backend;
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
        cooperative=cooperative,
        temp=temp,
        switch_below=switch_below,
    )
end


function _all_impl(
    pred, v::AbstractArray, backend::Backend;

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # GPU settings
    block_size::Int=256,
    cooperative::Bool=true,

    # GPU settings passed to mapreduce, only used if cooperative=false
    temp::Union{Nothing, AbstractArray}=nothing,
    switch_below::Int=0,
)
    if backend isa GPU
        @argcheck block_size > 0

        # Some platforms crash when multiple threads write to the same memory location in a global
        # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
        # CUDA). If not cooperative, we need to do a mapreduce
        if cooperative
            out = KernelAbstractions.zeros(backend, Int8, 1)
            _any_global!(backend, block_size)(out, (!pred), v, ndrange=length(v))
            outh = @allowscalar(out[1])
            return outh == 0 ? true : false
        else
            return mapreduce(
                pred,
                (x, y) -> x && y,
                v,
                backend;
                init=true,
                block_size=block_size,
                temp=temp,
                switch_below=switch_below,
            )
        end
    else
        overall = true
        task_partition(length(v), max_tasks, min_elems) do irange
            for i in irange
                if !pred(v[i])
                    overall = false
                    break
                end
            end
        end
        return overall
    end
end
