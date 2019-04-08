using CSV
using Dates
using DataFrames
using Distributed

const desired_nworkers = 30

if nworkers() != desired_nworkers
    wait(rmprocs(workers()))
    addprocs(desired_nworkers)
end

@info "Started $(nworkers())"
@info "Precompiling simulation code..."

include("debug.jl")

function main()
    @info "Running simulations..."
    data = test_parallel_sim(1:500)
    @info "Writing data..."
    result_dir = realpath("$(@__DIR__)/../results/")
    file_name = "sim_results-$(gethostname())-$(now()).csv"
    file = CSV.write(joinpath(result_dir, file_name), data)
    @info "All done! Wrote results to $file."
end

main()
