using CSV
using Dates
using DataFrames
using Distributed

const desired_nworkers = 1

if nworkers() != desired_nworkers
    wait(rmprocs(workers()))
    addprocs(desired_nworkers)
end

@info "Started $(nworkers())"
@info "Precompiling simulation code..."

include("debug.jl")

function main()
    @info "Running simulations..."
    data = test_parallel_sim(1:3, ignore_uncommited_changed=false)
    @info "Writing data..."
    result_dir = realpath("$(@__DIR__)/../results/")
    file_name = "sim_results-$(gethostname())-$(now()).csv"
    file = CSV.write(joinpath(result_dir, file_name), data)
    @info "All done! Wrote results to $file."
end

main()
