@kernel inbounds=true cpu=false unsafe_indices=true function _slicing_mean1d!(src,@Const(m))
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N
    if i <= length(src)
        src[i] -= m
    end
end


@kernel inbounds=true cpu=false unsafe_indices=true function _slicing_meannd!(src, @Const(m), @Const(dims))
    src_strides = strides(src)
    m_strides   = strides(m)
    nd          = length(src_strides)
    N       = @groupsize()[1]
    iblock  = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1
    tid     = ithread + iblock * N

    if tid < length(src)
        tmp    = tid
        midx0  = typeof(tid)(0)  
        KernelAbstractions.Extras.@unroll for i in nd:-1i16:1i16
            idxi = tmp ÷ src_strides[i]   
            if i != dims
                midx0 += idxi * m_strides[i]
            end
            tmp = tmp % src_strides[i]
        end
        src[tid + 0x1] -= m[midx0 + 0x1]
    end
end
function _slicing_meannd_cpu!(src::AbstractArray{T,N}, m, dims::Int;max_tasks,min_elems) where {T<:Real, N}
    src_strides ::NTuple{N,Int} = strides(src)
    m_strides  ::NTuple{N,Int} = strides(m)
    nd          = length(src_strides)
    foreachindex(src; max_tasks=max_tasks, min_elems=min_elems) do isrc
        tmp   = isrc - 1
        midx0  = 0
        KernelAbstractions.Extras.@unroll for i in nd:-1:1
            @inbounds idxi = tmp ÷ src_strides[i]
            if i != dims
                @inbounds midx0 += idxi * m_strides[i]
            end
            @inbounds tmp = tmp % src_strides[i]
        end
        @inbounds src[isrc] -= m[midx0 + 1]
    end
end

"""
    var!(
        src::AbstractArray{T}, backend::Backend=get_backend(src);
        dims::Union{Nothing, Int}=nothing,
        corrected ::Bool = true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
        prefer_threads::Bool=true,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    ) where {T<:Real}

    Compute the varience of `src` along dimensions `dims`. Can change `src` when using non-integer types
    If `dims` is `nothing`, reduce `src` to a scalar. If `dims` is an integer, reduce `src` along that
    dimension. The return type will be the same as the element type of `src` if it is a float type, or `Float32`
    if it is an integer type.
    ## CPU settings
    Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
    arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
    cases are scaling linearly with the number of threads.

    ## GPU settings
    The `block_size` parameter controls the number of threads per block.

    The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
    (`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
    required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
    array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
    except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
    zero, check against `Base.reduce` for CPU arrays for exact behavior.

    The `switch_below` parameter controls the threshold below which the reduction is performed on the
    CPU and is only used for 1D reductions (i.e. `dims=nothing`).
"""
function var!(
    src::AbstractArray{T,N},backend::Backend=get_backend(src);
    dims::Union{Nothing, Int}=nothing,
    corrected ::Bool = true,
    # CPU settings - ignored here
    max_tasks::Int = Threads.nthreads(),
    min_elems::Int = 1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int = 256,
    temp::Union{Nothing, AbstractArray} = nothing,
    switch_below::Int=0,
)  where {T<:Real,N}  
    m = mean(
        src,backend;
        dims=dims,
        # CPU settings - ignored here
        max_tasks = max_tasks,
        min_elems = min_elems,
        prefer_threads = prefer_threads,
        # GPU settings
        block_size = block_size,
        temp = temp,
        switch_below = switch_below
    )
    if T<:Integer
        src = Float32.(src)
    end
    if isnothing(dims)
        src .-= m
    else
        if use_gpu_algorithm(backend, prefer_threads)
            if N == 1
                _slicing_mean1d!(backend,block_size)(src,m;ndrange=length(src))
            else 
                _slicing_meannd!(backend,block_size)(src,m,dims;ndrange=length(src))
            end
        else
            if N ==1
                foreachindex(src; max_tasks=max_tasks, min_elems=min_elems) do i
                    src[i] -= m
                end
            else
                _slicing_meannd_cpu!(src,m,dims;max_tasks=max_tasks,min_elems=min_elems)
            end
        end
    end
    ntmp = isnothing(dims) ? nothing : m
    res = mapreduce(x->x*x,+,src,backend;
        init=zero(eltype(src)),
        dims=dims,
        max_tasks=max_tasks,
        min_elems=min_elems,
        prefer_threads=prefer_threads,
        block_size=block_size,
        temp=ntmp,
        switch_below=switch_below)
    if isnothing(dims) || N == 1
        res /= (length(src) - ifelse(corrected , 1 , 0))
        return res 
    end
    s = eltype(src)(1)/ (size(src,dims) - ifelse(corrected , 1 , 0))
    res .*= s
    return res
end

"""
    var!(
        src::AbstractArray{T}, backend::Backend=get_backend(src);
        dims::Union{Nothing, Int}=nothing,
        corrected ::Bool = true,

        # CPU settings
        max_tasks::Int=Threads.nthreads(),
        min_elems::Int=1,
        prefer_threads=prefer_threads,

        # GPU settings
        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
        switch_below::Int=0,
    ) where {T<:Real}

    Compute the varience of `src` along dimensions `dims`.
    If `dims` is `nothing`, reduce `src` to a scalar. If `dims` is an integer, reduce `src` along that
    dimension. The return type will be the same as the element type of `src` if it is a float type, or `Float32`
    if it is an integer type.
    ## CPU settings
    Use at most `max_tasks` threads with at least `min_elems` elements per task. For N-dimensional
    arrays (`dims::Int`) multithreading currently only becomes faster for `max_tasks >= 4`; all other
    cases are scaling linearly with the number of threads.

    ## GPU settings
    The `block_size` parameter controls the number of threads per block.

    The `temp` parameter can be used to pass a pre-allocated temporary array. For reduction to a scalar
    (`dims=nothing`), `length(temp) >= 2 * (length(src) + 2 * block_size - 1) ÷ (2 * block_size)` is
    required. For reduction along a dimension (`dims` is an integer), `temp` is used as the destination
    array, and thus must have the exact dimensions required - i.e. same dimensionwise sizes as `src`,
    except for the reduced dimension which becomes 1; there are some corner cases when one dimension is
    zero, check against `Base.reduce` for CPU arrays for exact behavior.

    The `switch_below` parameter controls the threshold below which the reduction is performed on the
    CPU and is only used for 1D reductions (i.e. `dims=nothing`).
"""
function var(
    src::AbstractArray{T},backend::Backend=get_backend(src);
    dims::Union{Nothing, Int}=nothing,
    corrected ::Bool = true,
    # CPU settings - ignored here
    max_tasks::Int = Threads.nthreads(),
    min_elems::Int = 1,
    prefer_threads::Bool=true,

    # GPU settings
    block_size::Int = 256,
    temp::Union{Nothing, AbstractArray} = nothing,
    switch_below::Int=0,
)  where {T<:Real}  
    return var!(copy(src),backend;
        dims=dims,
        corrected = corrected,
        # CPU settings - ignored here
        max_tasks = max_tasks,
        min_elems = min_elems,
        prefer_threads = prefer_threads,
        # GPU settings
        block_size = block_size,
        temp = temp,
        switch_below = switch_below
    ) 
end