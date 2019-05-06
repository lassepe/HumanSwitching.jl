using CSV
using Dates
using DataFrames
using Distributed

const desired_nworkers = 10

if nworkers() != desired_nworkers
    wait(rmprocs(workers()))
    addprocs(desired_nworkers)
end

@info "Started $(nworkers())"
@info "Precompiling simulation code..."

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using HumanSwitching
end

function main()
    solver_keys = ["POMCPOW"]
    @info "Running simulations..."
    data = parallel_sim(1:1000, solver_keys; problem_instance_keys=["CornerGoalsNonTrivial"])
    @info "Writing data..."
    result_dir = realpath("$(@__DIR__)/../results/")
    file_name = "sim_results-$(gethostname())-$(now())-$(join(solver_keys, "_")).csv"
    file = CSV.write(joinpath(result_dir, file_name), data)
    @info "All done! Wrote results to $file."
end

main()
