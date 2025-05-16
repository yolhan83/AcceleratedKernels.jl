
using StaticArrays
using SyncBarriers
using BenchmarkTools
import AcceleratedKernels as AK


using AllocCheck

using Random
Random.seed!(0)




@check_allocs ignore_throw=false function _sample_sort_histogram!(
    v::AbstractVector{T}, comp,
    splitters::Vector{T}, histograms::Matrix{Int},
    itask, irange,
) where T

    @inbounds begin

        # Compute the bucket histograms for this task
        for i in irange

            # Find the bucket for this element
            ibucket = 1 + @inline AK._searchsortedlast(splitters, v[i], 1, length(splitters), comp)

            # Increment the histogram for this task
            histograms[ibucket, itask] += 1
        end
    end

    nothing
end


@check_allocs ignore_throw=false function _sample_sort_move_buckets!(
    v, dest, comp,
    splitters, offsets,
    itask, max_tasks,
)
    @inbounds begin
        # Compute this task's output slice indices
        num_elements = length(v)
        dest_istart = offsets[itask] + 1
        if itask == max_tasks
            dest_istop = num_elements
        else
            dest_istop = offsets[itask + 1]
        end

        # Copy the elements into the destination buffer following splitters
        idest = dest_istart
        for i in 1:num_elements

            # Find the bucket for this element
            ibucket = AK._searchsortedlast(splitters, v[i], 1, max_tasks - 1, comp) + 1

            # Copy the element into the destination buffer
            if ibucket == itask
                dest[idest] = v[i]
                itemp += 1
            end
        end
    end

    nothing
end




# @check_allocs ignore_throw=false
function _sample_sort_parallel!(
    v, dest, comp,
    splitters, histograms,
    max_tasks,
)
    # Compute the histogram for each task
    AK.itask_partition(length(v), max_tasks, 1) do itask, irange
        _sample_sort_histogram!(
            v, comp,
            splitters, histograms,
            itask, irange,
        )
    end

    # # Debug
    # tp = AK.TaskPartitioner(length(v), max_tasks, 1)
    # for itask in 1:max_tasks
    #     _sample_sort_histogram!(
    #         v, comp,
    #         splitters, histograms,
    #         itask, tp[itask],
    #     )
    # end

    # # Sum up histograms and compute index offsets for each task
    # for itask in 2:max_tasks
    #     for j in 1:max_tasks
    #         histograms[j, 1] += histograms[j, itask]
    #     end
    # end
    # offsets = @view histograms[:, 1]
    # AK.accumulate!(
    #     +, offsets,
    #     init=0,
    #     inclusive=false,
    # )

    # # Move the elements into the destination buffer
    # @inbounds for itask in 1:max_tasks
    #     tasks[itask] = Threads.@spawn _sample_sort_move_buckets!(
    #         v, dest, comp,
    #         splitters, offsets,
    #         itask, max_tasks,
    #     )
    # end
    # @inbounds for itask in 1:max_tasks
    #     wait(tasks[itask])
    # end

    nothing
end



function sample_sort!(
    v;
    max_tasks=Threads.nthreads(),

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    temp=nothing
)

    oversampling_factor = 4
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

    # Construct comparator
    ord = Base.Order.ord(lt, by, rev, order)
    comp = (x, y) -> Base.Order.lt(ord, x, y)

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
    histograms = zeros(Int, max_tasks + 8, max_tasks)       # Add padding to avoid false sharing

    # Run threaded region
    _sample_sort_parallel!(
        v, dest, comp,
        splitters, histograms,
        max_tasks,
    )

    dest
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








v = rand(Float32, 100_000)

try
    temp = sample_sort!(v)
catch e
    display(e.errors[1])
    rethrow(e)
end


t = @timed sample_sort!(v)


# @assert issorted(temp)
# println("sorted")


display(@benchmark sort!(v) setup=(v=rand(Float64, 10_000_000)))
display(@benchmark sample_sort!(v) setup=(v=rand(Float64, 10_000_000)))
