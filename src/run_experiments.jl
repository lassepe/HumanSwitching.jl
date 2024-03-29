using CSV
using Dates
using DataFrames
using Distributed

const desired_nworkers = 20

const IN_SLURM = "SLURM_JOBID" in keys(ENV)
IN_SLURM && using ClusterManagers

if IN_SLURM
    pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    print("\n")
else
    if nworkers() != desired_nworkers
        wait(rmprocs(workers()))
        pids = addprocs(desired_nworkers)
    end
end


@info "Started $(nworkers()) workers..."
@info "Precompiling simulation code..."

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using HumanSwitching
end

function main()
    solver_keys = ["POMCPOW", "ProbObstacles"]
    @info "Running simulations..."
    data = parallel_sim(1:5000, solver_keys; problem_instance_keys=["CornerGoalsNonTrivial"], planner_hbm_keys=["HumanMultiGoalBoltzmann_half_goals"])
    @info "Writing data..."
    result_dir = realpath("$(@__DIR__)/../results/")
    file_name = "sim_results-$(gethostname())-$(now())-$(join(solver_keys, "_")).csv"
    file = CSV.write(joinpath(result_dir, file_name), data)
    @info "All done! Wrote results to $file."
end

main()

rmprocs(pids)
