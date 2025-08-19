@inline function chan_merge(a::Tuple{Int64,T,T}, b::Tuple{Int64,T,T}) where {T<:Real}
    nA, mA, M2A = a
    nB, mB, M2B = b
    if nA == 0
        return b
    elseif nB == 0
        return a
    else
        nAB  = nA + nB                          
        δ    = mB - mA                          
        invn = inv(T(nAB))                
        mean = (nA*mA + nB*mB) * invn
        cross = (δ*δ) * (nA*nB * invn)
        return (nAB, mean, M2A + M2B + cross)
    end
end

function var(
    src::AbstractArray{T,N}, backend::Backend=get_backend(src);
    dims::Union{Nothing,Int}=nothing,
    corrected::Bool=true,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    block_size::Int=256,
    temp::Union{Nothing,AbstractArray}=nothing,  # ignored
    switch_below::Int=0,
) where {T<:Integer,N}

    init   = (0, 0f0, 0f0)
    mapper = x -> (1, Float32(x), 0f0)

    stats = mapreduce(
        mapper, chan_merge, src, backend;
        init=init, neutral=init,
        dims=dims,
        max_tasks=max_tasks, min_elems=min_elems, prefer_threads=prefer_threads,
        block_size=block_size,
        temp=nothing,
        switch_below=switch_below,
    )

    if dims === nothing
        n, _, M2 = stats
        return M2 / Float32(n - ifelse(corrected , 1 , 0))
    else
        # stats is an array of (Int32, Float32, Float32); finalize into Float32 array
        out = similar(stats, Float32)
        AcceleratedKernels.map!(
            s -> @inbounds(s[3] / Float32(s[1] - ifelse(corrected , 1 , 0))),
            out, stats, backend;
            max_tasks=max_tasks, min_elems=min_elems, block_size=block_size,
        )
        return out
    end
end


function var(
    src::AbstractArray{T,N}, backend::Backend=get_backend(src);
    dims::Union{Nothing,Int}=nothing,
    corrected::Bool=true,
    max_tasks::Int=Threads.nthreads(),
    min_elems::Int=1,
    prefer_threads::Bool=true,
    block_size::Int=256,
    temp::Union{Nothing,AbstractArray}=nothing,  # ignored
    switch_below::Int=0,
) where {T<:AbstractFloat,N}

    init   = (0, zero(T), zero(T))
    mapper = x -> (1, x, zero(typeof(x)))

    stats = mapreduce(
        mapper, chan_merge, src, backend;
        init=init, neutral=init,
        dims=dims,
        max_tasks=max_tasks, min_elems=min_elems, prefer_threads=prefer_threads,
        block_size=block_size,
        temp=nothing,
        switch_below=switch_below,
    )

    if dims === nothing
        n, _, M2 = stats
        return M2 / T(n - ifelse(corrected , 1 , 0))
    else
        # stats is an array of (Int32, Float32, Float32); finalize into Float32 array
        out = similar(stats, T)
        AcceleratedKernels.map!(
            s -> @inbounds(s[3] / (s[1] - ifelse(corrected , 1 , 0))),
            out, stats, backend;
            max_tasks=max_tasks, min_elems=min_elems, block_size=block_size,
        )
        return out
    end
end
