
using StaticArrays
using SyncBarriers
using BenchmarkTools
import AcceleratedKernels as AK


using AllocCheck

using Random
Random.seed!(0)




# @check_allocs ignore_throw=false
function _sample_sort_histogram!(v, splitters, histograms, itask, irange)
    for i in irange
        ibucket = 1 + AK._searchsortedlast(splitters, v[i], 1, length(splitters), isless)
        histograms[ibucket, itask] += 1
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
            v,
            splitters, histograms,
            itask, irange,
        )
    end
    nothing
end



function sample_sort!(
    v;
    max_tasks=Threads.nthreads(),

    lt=isless,
    by=identity,
    rev::Union{Bool, Nothing}=nothing,
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


# display(@benchmark sort!(v) setup=(v=rand(Float64, 10_000_000)))
display(@benchmark sample_sort!(v) setup=(v=rand(Float64, 100_000)))
