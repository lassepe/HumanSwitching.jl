using Distributed
@everywhere begin
    using Pkg
    Pkg.activate(".")
    using ParticleFilters
    using POMDPs
    using POMDPPolicies
    using POMDPSimulators
    using POMDPGifs
    using BeliefUpdaters
    using POMCPOW
    using ARDESPOT
    using MCTS
    using HumanSwitching
    const HS = HumanSwitching
end

using Blink
using Revise
using Printf
using Compose
using Random
using ProgressMeter
using D3Trees

using Profile
using ProfileView
using Test
using BenchmarkTools

using DataFrames

# TODO: move this to a package / module

@everywhere begin
    using POMDPModelTools
    using CPUTime

    """
    TimedPolicy

    A thin planner/policy wrapper to add the time to the action info.
    """
    struct TimedPolicy{P <: Policy} <: Policy
        p::P
    end

    POMDPs.action(timed_policy::TimedPolicy, x) = action(timed_policy.p, x)

    function POMDPModelTools.action_info(timed_policy::TimedPolicy, x; kwargs...)
        start_us = CPUtime_us()
        action, info = action_info(timed_policy.p, x; kwargs...)
        info[:planning_cpu_time_us] = CPUtime_us() - start_us
        return action, info
    end
end

"""
current_commit_id

Determines the git commit id of the `HEAD` of this repo.
"""
current_commit_id() = chomp(read(`git rev-parse --verify HEAD`, String))

"""
reproduce_scenario

Reproduces the simulation environment for a given DataFrameRow
"""
function reproduce_scenario(scenario_data::DataFrameRow;
                            render_gif::Bool=false, ignore_commit_id::Bool=false)
    # verify that the correct commit was checked out (because behavior of code
    # might have changed)
    if !ignore_commit_id && current_commit_id() != scenario_data.git_commit_id
        return @error "Reproducing scenario with wrong commit ID!.
        If you are sure that this is still a good idea to do this, pass
        `ignore_commit_id=true` as kwarg to the call of `reproduce_scenario`."
    end

    sim = setup_test_scenario(scenario_data[:planner_key],
                               scenario_data[:i_run])

    # some sanity checks on the hist
    hist = simulate(sim)
    if discounted_reward(hist) != scenario_data.discounted_reward
        @warn "Reproduced reward differs from saved reward.
        Are you sure, no files changed since this was recorded?"
    else
        @info "Reproduced reward matches with saved data. Seems correct."
    end

    if validation_hash(hist) != scenario_data[:hist_validation_hash]
        @warn "Reproduced sim hist had differend hash.
        Are you sure, no files changed since this was recorded?"
    else
        @info "Reproduced `hist` hash matches with save data. Seems correct."
    end

    planner_model = sim.policy.problem
    hist = simulate(sim)

    return planner_model, hist
end

function setup_test_scenario(planner_key::String, i_run::Int)
    rng = MersenneTwister(i_run)
    scenario_rng = MersenneTwister(i_run + 1)

    # the model of the "true" human...
    simulation_hbm = HumanPIDBehavior(potential_targets=[HS.rand_pose(RoomRep(), scenario_rng) for i=1:10], goal_change_likelihood=0.01)
    # ...and the "true" world model (used for generating samples)
    ptnm_cov = [0.01, 0.01, 0.01]
    simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                     simulation_hbm,
                                                     HSGaussianNoisePTNM(pose_cov=ptnm_cov),
                                                     deepcopy(rng))

    # compose the corresponding planning model
    planning_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10), # using transition noise for observation weighting
                                      planner_hbm_map()[planner_key],   # using the hbm from the dict of models to explore
                                      HSIdentityPTNM(),                 # the planner assumes that transitions are exact (to reduce branching)
                                      simulation_model,                 # the simulation model is used to clone the shared properties (initial conditions etc)
                                      deepcopy(rng))

    n_particles = 2000
    # the blief updater is run with a stocahstic version of the world
    belief_updater = BasicParticleFilter(planning_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
    # the policy plannes without a model as it is always the same action
    solver = POMCPOWSolver(tree_queries=12000, max_depth=70, criterion=MaxUCB(80),
                           k_action=5, alpha_action=0.1,
                           k_observation=3, alpha_observation=0.15,
                           check_repeat_obs=true,
                           check_repeat_act=true,
                           estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))
    planner = solve(solver, planning_model)
    timed_planner = TimedPolicy(planner)

    # compose metadata
    git_commit_id = current_commit_id()
    md = Dict(:planner_key => planner_key,
              :i_run => i_run,
              :git_commit_id => git_commit_id)

    # compose the sim object for the `run_parallel` queue
    return Sim(simulation_model,
               timed_planner,
               belief_updater,
               initialstate_distribution(planning_model),
               initialstate(simulation_model, deepcopy(rng)),
               rng=deepcopy(rng),
               max_steps=500,
               metadata=md)
end

"""
planner_hbm

Maps a planner_key to a corresponding model instance. (to avoid storing the whole complex object)
"""
planner_hbm_map() = Dict{String, HumanBehaviorModel}(
                                                     "HumanBoltzmannModel1" => HumanBoltzmannModel()
                                                    )

@everywhere validation_hash(hist::SimHistory) = string(hash(collect(eachstep(hist, "s,a,sp,r,o"))))
"""
test_parallel_sim

Run experiments over `planner_hbms` for `runs`
"""
function test_parallel_sim(runs::UnitRange{Int}; planner_hbms=planner_hbm_map())
    # queue of simulation instances...
    sims::Array{Sim, 1} = []
    # filled with scenarios for different hbms and runs
    for (planner_key, planner_hbm) in planner_hbms, i_run in runs
        sim = push!(sims, setup_test_scenario(planner_key, i_run))
    end
    # Simulation is launched in parallel mode. In order for this to work, julia
    # musst be started as: `julia -p n`, where n is the number of
    # workers/processes
    data = run_parallel(sims) do sim::Sim, hist::SimHistory
        return [:n_steps => n_steps(hist),
                :discounted_reward => discounted_reward(hist),
                :hist_validation_hash => validation_hash(hist),
                :mean_planning_time => 1e-6*mean(ai[:planning_cpu_time_us] for ai in eachstep(hist, "ai"))]
    end
    return data
end

function visualize(planner_model, hist; filename::String="visualize_debug")
    makegif(planner_model, hist, filename=joinpath(@__DIR__, "../renderings/$filename.gif"),
            extra_initial=true, show_progress=true, render_kwargs=(sim_hist=hist, show_info=true))
end

function tree(model, hist, planner, step=1)
    beliefs = collect(eachstep(hist, "b"))
    b = beliefs[step]
    a, info = action_info(planner, b, tree_in_info=true)
    inbrowser(D3Tree(info[:tree], init_expand=1), "chromium")
end
