using BenchmarkTools


# @check_allocs ignore_throw=false
function sample_sort_histogram!(
    v::AbstractVector{T},
    splitters::Vector{T},
    histograms::Matrix{Int},
    itask, irange,
) where T

    @inbounds begin

        # Compute the bucket histograms for this task
        for i in irange

            # Find the bucket for this element
            ibucket = 1 + searchsortedlast(splitters, v[i])

            # Increment the histogram for this task
            histograms[ibucket, itask] += 1
        end
    end

    nothing
end


function sample_sort_parallel!(v, splitters, histograms, max_tasks)
    # Compute the histogram for each task - i.e. the number of elements in each bucket
    tasks = Vector{Task}(undef, max_tasks)
    for itask in 1:max_tasks
        irange = div((itask - 1) * length(v), max_tasks) + 1 : div(itask * length(v), max_tasks)
        # @show irange
        tasks[itask] = Threads.@spawn sample_sort_histogram!(
            v,
            splitters, histograms,
            itask, irange,
        )
    end

    # Wait for all tasks to finish
    for itask in 1:max_tasks
        wait(tasks[itask])
    end

    nothing
end


function sample_sort!(
    v;
    max_tasks=Threads.nthreads(),
)
    splitters = Vector(range(0, 1, length=max_tasks + 1)[2:end-1])
    histograms = zeros(Int, max_tasks + 8, max_tasks)           # padding to avoid false sharing
    sample_sort_parallel!(v, splitters, histograms, max_tasks)
end


v = rand(1_000_000)

@benchmark sample_sort!(v)

