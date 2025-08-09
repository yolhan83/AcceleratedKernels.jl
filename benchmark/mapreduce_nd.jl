group = addgroup!(SUITE, "mapreduce_nd")

n1 = 3
n2 = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_dims=1"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=1"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_dims=2"] = @benchmarkable @sb(Base.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=2"] = @benchmarkable @sb(AK.reduce(+, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    T == Float32 || continue

    _group["base_dims=1_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=1_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=1)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))

    _group["base_dims=2_sin"] = @benchmarkable @sb(Base.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
    _group["acck_dims=2_sin"] = @benchmarkable @sb(AK.mapreduce(sin, +, v; init=$T(0), dims=2)) setup=(v = ArrayType(rand(rng, $randrange, n1, n2)))
end
