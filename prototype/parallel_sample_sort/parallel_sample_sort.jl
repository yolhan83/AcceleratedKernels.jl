
using StaticArrays
using SyncBarriers
using BenchmarkTools
import AcceleratedKernels as AK

using Profile
using PProf

using AllocCheck

using Random
Random.seed!(0)




function _sample_sort_histogram!(
    v::AbstractVector{T}, ord,
    splitters::Vector{T}, histograms::Matrix{Int},
    itask, irange,
) where T
    @inbounds begin
        for i in irange
            ibucket = 1 + AK._searchsortedlast(splitters, v[i], 1, length(splitters), ord)
            histograms[ibucket, itask] += 1
        end
    end
    nothing
end


function _sample_sort_move_buckets!(
    v, dest, ord,
    splitters, global_offsets, task_offsets,
    itask, max_tasks, irange,
)
    # Copy the elements into the destination buffer following splitters
    @inbounds begin

        # Compute the destination indices for this task into each bucket
        offsets = @view task_offsets[1:max_tasks, itask]
        for it in 1:max_tasks
            offsets[it] += global_offsets[it] + 1
        end

        for i in irange
            # Find the bucket for this element
            ibucket = 1 + AK._searchsortedlast(splitters, v[i], 1, length(splitters), ord)

            # Get the current destination index for this element, then increment
            dest[offsets[ibucket]] = v[i]
            offsets[ibucket] += 1
        end
    end

    nothing
end


function _sample_sort_compute_offsets!(histograms, max_tasks)
    @inbounds begin
        # Sum up histograms and compute global offsets for each task
        offsets = @view histograms[1:max_tasks, max_tasks + 1]
        for itask in 1:max_tasks
            for j in 1:max_tasks
                offsets[j] += histograms[j, itask]
            end
        end
        AK.accumulate!(+, offsets, init=0, inclusive=false)

        # Compute each task's local offset into each bucket
        for itask in 1:max_tasks
            AK.accumulate!(
                +, @view(histograms[itask, 1:max_tasks]),
                init=0,
                inclusive=false,
            )
        end
    end

    offsets
end



function _sample_sort_sort_bucket!(
    v, dest, offsets, itask, max_tasks;
    lt, by, rev, order    
)
    @inbounds begin
        istart = offsets[itask] + 1
        istop = itask == max_tasks ? length(dest) : offsets[itask + 1]

        if istart == istop
            v[istart] = dest[istart]
            return
        elseif istart > istop
            return
        end

        # At the end we will have to move elements from dest back to v anyways; for every
        # odd-numbered itask, move elements first, to avoid false sharing from threads
        if isodd(itask)
            copyto!(v, istart, dest, istart, istop - istart + 1)
            sort!(view(v, istart:istop), lt=lt, by=by, rev=rev, order=order)
        else
            # For even-numbered itasks, sort first, then move elements back to v
            sort!(view(dest, istart:istop), lt=lt, by=by, rev=rev, order=order)
            copyto!(v, istart, dest, istart, istop - istart + 1)
        end
    end

    return
end




# @check_allocs ignore_throw=false
function _sample_sort_parallel!(
    v, dest, ord,
    splitters, histograms,
    max_tasks;
    lt, by, rev, order,
)
    # Compute the histogram for each task
    tp = AK.TaskPartitioner(length(v), max_tasks, 1)
    AK.itask_partition(tp) do itask, irange
        _sample_sort_histogram!(
            v, ord,
            splitters, histograms,
            itask, irange,
        )
    end

    # Compute the global and local (per-bucket) offsets for each task
    offsets = @view histograms[1:max_tasks, max_tasks + 1]
    _sample_sort_compute_offsets!(histograms, max_tasks)

    # Move the elements into the destination buffer
    AK.itask_partition(tp) do itask, irange
        _sample_sort_move_buckets!(
            v, dest, ord,
            splitters, offsets, histograms,
            itask, max_tasks, irange,
        )
    end

    # Sort each bucket in parallel
    AK.itask_partition(tp) do itask, irange
        _sample_sort_sort_bucket!(
            v, dest, offsets, itask, max_tasks;
            lt=lt, by=by, rev=rev, order=order,
        )
    end

    # # Debug: single-threaded version
    # tp = AK.TaskPartitioner(length(v), max_tasks, 1)
    # for itask in 1:max_tasks
    #     irange = tp[itask]
    #     _sample_sort_histogram!(
    #         v, ord,
    #         splitters, histograms,
    #         itask, irange,
    #     )
    # end
    # _sample_sort_compute_offsets!(histograms, max_tasks)
    # offsets = @view histograms[1:max_tasks, max_tasks + 1]
    # for itask in 1:max_tasks
    #     irange = tp[itask]
    #     _sample_sort_move_buckets!(
    #         v, dest, ord,
    #         splitters, offsets, histograms,
    #         itask, max_tasks, irange,
    #     )
    # end
    # for itask in 1:max_tasks
    #     _sample_sort_sort_bucket!(
    #         v, dest, offsets, itask, max_tasks;
    #         lt=lt, by=by, rev=rev, order=order,
    #     )
    # end

    nothing
end



@inline function sample_sort!(
    v;
    max_tasks=Threads.nthreads(),

    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,

    temp=nothing
)
    oversampling_factor = 16
    num_elements = length(v)

    if num_elements < 2
        return v
    end

    if max_tasks == 1 || num_elements < oversampling_factor * max_tasks
        return sort!(v, lt=lt, by=by, rev=rev, order=order)
    end

    # Create a temporary buffer for the sorted output
    if temp === nothing
        dest = similar(v)
    else
        # TODO add checks
        dest = temp
    end

    # Take equally spaced samples, save them in dest
    num_samples = oversampling_factor * max_tasks
    isamples = IntLinSpace(1, num_elements, num_samples)
    @inbounds for i in 1:num_samples
        dest[i] = v[isamples[i]]
    end

    # Sort samples and choose splitters
    sort!(view(dest, 1:num_samples), lt=lt, by=by, rev=rev, order=order)
    splitters = Vector{eltype(v)}(undef, max_tasks - 1)
    for i in 1:(max_tasks - 1)
        splitters[i] = dest[div(i * num_samples, max_tasks)]
    end

    # Pre-allocate histogram for each task; each column is exclusive to the task
    histograms = zeros(Int, max_tasks + 8, max_tasks + 1)       # Add padding to avoid false sharing

    # Run threaded region
    ord = Base.Order.ord(lt, by, rev, order)
    _sample_sort_parallel!(
        v, dest, ord,
        splitters, histograms,
        max_tasks;
        lt=lt, by=by, rev=rev, order=order,
    )

    v
end





# Utilities


# Create an integer linear space between start and stop on demand
struct IntLinSpace{T <: Integer}
    start::T
    stop::T
    length::T
end

function IntLinSpace(start::Integer, stop::Integer, length::Integer)
    start <= stop || throw(ArgumentError("`start` must be <= `stop`"))
    length >= 2 || throw(ArgumentError("`length` must be >= 2"))

    IntLinSpace{typeof(start)}(start, stop, length)
end

Base.IndexStyle(::IntLinSpace) = IndexLinear()
Base.length(ils::IntLinSpace) = ils.length

Base.firstindex(::IntLinSpace) = 1
Base.lastindex(ils::IntLinSpace) = ils.length

function Base.getindex(ils::IntLinSpace, i)
    @boundscheck 1 <= i <= ils.length || throw(BoundsError(ils, i))

    if i == 1
        ils.start
    elseif i == length
        ils.stop
    else
        ils.start + div((i - 1) * (ils.stop - ils.start), ils.length - 1, RoundUp)
    end
end




for _ in 1:1000
    global v = rand(Float32, 40)
    sample_sort!(v)
    @assert issorted(v)
end



v = zeros(Float32, 1_000_000_000)

# Collect a profile
Profile.clear()
@profile sample_sort!(v)
pprof()


@assert issorted(v)



t = @timed sample_sort!(v)


# @assert issorted(temp)
# println("sorted")


display(@benchmark sort!(v) setup=(v=rand(Float64, 1_000_000)))


display(@benchmark sample_sort!(v) setup=(v=rand(Float64, 1_000_000)))



