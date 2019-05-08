using CSV
using Dates
using DataFrames
using Distributed
using Pkg
Pkg.instantiate()

const desired_nworkers = 30

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
    solver_keys = ["POMCPOW", "ProbObstacles", "GapChecking"]
    @info "Running simulations..."
    #data = parallel_sim(1:1000, solver_keys; problem_instance_keys=["CornerGoalsNonTrivial"])
    data = parallel_sim(101:200, [k for k in keys(solver_setup_span_map(keys_only=true))], simulation_hbm_keys=["HumanMultiGoalBoltzmann_all_goals"], planner_hbm_keys=["HumanConstVelBehavior"])
    @info "Writing data..."
    result_dir = realpath("$(@__DIR__)/../results/")
    file_name = "sim_results-$(gethostname())-$(now())-$(join(solver_keys, "_")).csv"
    file = CSV.write(joinpath(result_dir, file_name), data)
    @info "All done! Wrote results to $file."
end

main()

rmprocs(pids)
