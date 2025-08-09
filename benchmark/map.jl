group = addgroup!(SUITE, "map")

n = 1_000_000
f(x) = typeof(x)(2) * x


for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_2x"] = @benchmarkable @sb(Base.map(f, v)) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_2x"] = @benchmarkable @sb(AK.map(f, v)) setup=(v = ArrayType(rand(rng, $randrange, n)))

    T == Float32 || continue

    _group["base_sin"] = @benchmarkable @sb(Base.map(sin, v)) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_sin"] = @benchmarkable @sb(AK.map(sin, v)) setup=(v = ArrayType(rand(rng, $randrange, n)))
end
