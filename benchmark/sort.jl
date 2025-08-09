group = addgroup!(SUITE, "sort")
n = 1_000_000
ntup = 5


for _test in [(UInt32, UInt32(1):UInt32(1_000_000)), NTuple{ntup, Int64}, Float32]

    T,randrange = if _test isa DataType
        _test, _test
    else
        _test
    end

    local _group = addgroup!(group, "$T")

    _group["base"] = @benchmarkable @sb(Base.sort!(v)) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck"] = @benchmarkable @sb(AK.sort!(v)) setup=(v = ArrayType(rand(rng, $randrange, n)))

    T == Float32 || continue

    _group["base_sin"] = @benchmarkable @sb(Base.sort!(v, by=sin)) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_sin"] = @benchmarkable @sb(AK.sort!(v, by=sin)) setup=(v = ArrayType(rand(rng, $randrange, n)))
end
