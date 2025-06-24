group = addgroup!(SUITE, "accumulate_1d")

acc_f(x, y) = sin(x) + cos(y)


GPUArrays.neutral_element(::typeof(acc_f), T) = T(0)

n = 1_000_000

for T in [UInt32, Int64, Float32]
    local _group = addgroup!(group, "$T")

    local randrange = T == Float32 ? T : T(1):T(100)

    _group["base_1d"] = @benchmarkable @sb(Base.accumulate(+, v; init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d"] = @benchmarkable @sb(AK.accumulate(+, v; init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))

    T == Float32 || continue

    _group["base_1d_sincos"] = @benchmarkable @sb(Base.accumulate(acc_f, v; init=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
    _group["acck_1d_sincos"] = @benchmarkable @sb(AK.accumulate(acc_f, v; init=$T(0), neutral=$T(0))) setup=(v = ArrayType(rand(rng, $randrange, n)))
end
