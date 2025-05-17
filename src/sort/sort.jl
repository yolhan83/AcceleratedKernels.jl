include("utils.jl")
include("merge_sort.jl")
include("merge_sort_by_key.jl")
include("merge_sortperm.jl")
include("cpu_sample_sort.jl")


# All other algorithms have the same naming convention as Julia Base ones; provide similar
# interface here too.


"""
    sort!(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),

        # GPU settings
        block_size::Int=256,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sort!(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    # CPU settings
    max_tasks=Threads.nthreads(),

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    _sort_impl!(
        v, backend,
        lt=lt, by=by, rev=rev, order=order,
        max_tasks=max_tasks,
        block_size=block_size,
        temp=temp,
    )
end


function _sort_impl!(
    v::AbstractArray, backend::Backend;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,

    max_tasks=Threads.nthreads(),
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if backend isa GPU
        merge_sort!(
            v, backend,
            lt=lt, by=by, rev=rev, order=order,
            block_size=block_size,
            temp=temp,
        )
    else
        sample_sort!(
            v;
            lt=lt, by=by, rev=rev, order=order,
            max_tasks=max_tasks,
            temp=temp,
        )
    end
end


"""
    sort(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),

        # GPU settings
        block_size::Int=256,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sort(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    # CPU settings
    max_tasks=Threads.nthreads(),

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    vcopy = copy(v)
    sort!(
        vcopy, backend,
        lt=lt, by=by, rev=rev, order=order,
        max_tasks=max_tasks,
        block_size=block_size,
        temp=temp,
    )
end


"""
    sortperm!(
        ix::AbstractArray,
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),

        # GPU settings
        block_size::Int=256,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sortperm!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    # CPU settings
    max_tasks=Threads.nthreads(),

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    _sortperm_impl!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        max_tasks=max_tasks,
        block_size=block_size,
        temp=temp,
    )
end


function _sortperm_impl!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend;

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,

    max_tasks=Threads.nthreads(),
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if backend isa GPU
        merge_sortperm_lowmem!(
            ix, v,
            lt=lt, by=by, rev=rev, order=order,
            block_size=block_size, temp=temp,
        )
    else
        sample_sortperm!(
            ix, v;
            lt=lt, by=by, rev=rev, order=order,
            max_tasks=max_tasks,
            temp=temp,
        )
    end
end


"""
    sortperm(
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Union{Nothing, Bool}=nothing,
        order::Base.Order.Ordering=Base.Order.Forward,

        # CPU settings
        max_tasks=Threads.nthreads(),

        # GPU settings
        block_size::Int=256,

        # Temporary buffer, same size as `v`
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sortperm(
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    # CPU settings
    max_tasks=Threads.nthreads(),

    # GPU settings
    block_size::Int=256,

    # Temporary buffer, same size as `v`
    temp::Union{Nothing, AbstractArray}=nothing,
)
    ix = similar(v, Int)
    sortperm!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        max_tasks=max_tasks,
        block_size=block_size,
        temp=temp,
    )
end
