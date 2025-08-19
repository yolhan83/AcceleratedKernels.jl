@kernel inbounds=true cpu=false unsafe_indices=true function _slicing_mean1d!(src,m)
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N
    if i <= length(src)
        src[i] -= m
    end
end
@kernel inbounds=true cpu=false unsafe_indices=true function _slicing_mean2d!(src,m,dims)
    @assert 1<=dims<=2
    Nx,Ny = @groupsize()
    iblock,jblock = @index(Group, NTuple)
    ithread,jthread = @index(Local, NTuple)
    i = ithread + (iblock - 0x1) * Nx
    j = jthread + (jblock - 0x1) * Ny
    if i<=size(src,1) && j<=size(src,2)
        if dims == 1
            src[i,j] -= m[1,j]
        else
            src[i,j] -= m[i,1]
        end
    end
end
@kernel inbounds=true cpu=false unsafe_indices=true function _slicing_mean3d!(src,m,dims)
    @assert 1<=dims<=3
    Nx,Ny,Nz = @groupsize()
    iblock,jblock,kblock = @index(Group, NTuple)
    ithread,jthread,kthread = @index(Local, NTuple)
    i = ithread + (iblock - 0x1) * Nx
    j = jthread + (jblock - 0x1) * Ny
    k = kthread + (kblock - 0x1) * Nz
    if i<=size(src,1) && j<=size(src,2) && k<=size(src,3)
        if dims == 1
            src[i,j,k] -= m[1,j,k]
        elseif dims == 2
            src[i,j,k] -= m[i,1,k]
        else
            src[i,j,k] -= m[i,j,1]
        end
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
    # use a special kernel ? what if more than 3 dims ?
    if backend isa GPU && N<=3
        if N == 1
            slicing_mean1d!(backend,block_size)(src,m;ndrange=size(src))
        elseif N == 2
            slicing_mean2d!(backend,block_size)(src,m,dims;ndrange=size(src))
        elseif N == 3
            slicing_mean3d!(backend,block_size)(src,m,dims;ndrange=size(src))
        end
    else
        for sl in eachslice(src,dims=dims)
            sl .-= selectdim(m,dims,1)
        end
    end
    res = mapreduce(x->x*x,+,src,backend;
            init=zero(eltype(src)),
            dims=dims,
            max_tasks=max_tasks,
            min_elems=min_elems,
            prefer_threads=prefer_threads,
            block_size=block_size,
            temp=temp,
            switch_below=switch_below)
    res ./= (size(src,dims) - ifelse(corrected , 1 , 0))
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