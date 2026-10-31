# Bayesian-bootstrap summary of the codec bake-off (units = SEG-second segments of the episode).
#
#   julia --project=julia julia/bb_summary.jl outputs/eval [baseline]    (reads seg_*.csv)
#
# Per variant: dWER as a ratio estimator Σwᵢeᵢ / Σwᵢnᵢ (Dirichlet weights over segments, shared
# across variants so paired differences are exact BB draws), and BB posteriors of the mean speaker
# cosine, PESQ-WB and log-mel L1 (closed form via BayesBoot). P(better) is the posterior
# probability that the variant beats `baseline` (default opus_12k) on that metric.
# Caveat: segments are adjacent pieces of one episode (same hosts), so this measures
# uncertainty about *this* episode, not about podcasts in general.

using BayesBoot, Printf, Random, Statistics

const NDRAW = 20_000

function load(dir)
    rows = Dict{String, Vector{NTuple{6, Float64}}}()
    files = filter(f -> startswith(f, "seg_") && endswith(f, ".csv"), readdir(dir))
    for file in files, (k, line) in enumerate(eachline(joinpath(dir, file)))
        k == 1 && continue
        v = split(line, ",")
        push!(get!(rows, v[1], NTuple{6, Float64}[]),
              (parse(Float64, v[2]), parse(Float64, v[4]), parse(Float64, v[5]),
               parse(Float64, v[6]), parse(Float64, v[7]), parse(Float64, v[8])))
    end
    Dict(k => sort(r) for (k, r) in rows)
end

function summ(x)
    x = filter(!isnan, x)
    m = bb_moments(x, 2)
    (m[1], sqrt(max(m[2] - m[1]^2, 0.0)), bb_quantile(x, 0.025), bb_quantile(x, 0.975))
end

function main(path, base = "opus_12k")
    R = load(path)
    nseg = length(first(values(R)))
    rng = Xoshiro(20261030)
    W = randexp(rng, NDRAW, nseg)
    W ./= sum(W; dims = 2)
    dwer(r) = (W * [x[3] for x in r]) ./ (W * [x[2] for x in r])
    wbase = haskey(R, base) ? dwer(R[base]) : nothing
    sbase = haskey(R, base) ? W * [x[4] for x in R[base]] : nothing
    order = sort(collect(keys(R)); by = k -> -median(dwer(R[k])))
    @printf("%d segments; baseline %s\n\n", nseg, base)
    println("| variant | dWER % (95% BB) | P(dWER < base) | spk cos ± sd | P(spk > base) | PESQ-WB ± sd | log-mel L1 ± sd |")
    println("|---|---|---|---|---|---|---|")
    for k in order
        r = R[k]
        d = dwer(r) .* 100
        pw = wbase === nothing ? NaN : mean(dwer(r) .< wbase)
        s = summ([x[4] for x in r])
        ps = sbase === nothing ? NaN : mean((W * [x[4] for x in r]) .> sbase)
        p = summ([x[5] for x in r])
        ml = summ([x[6] for x in r])
        @printf("| %s | %.2f [%.2f, %.2f] | %.2f | %.3f ± %.3f | %.2f | %.2f ± %.2f | %.3f ± %.3f |\n",
                k, mean(d), quantile(d, 0.025), quantile(d, 0.975), pw, s[1], s[2], ps, p[1], p[2], ml[1], ml[2])
    end
end

main(ARGS...)
