using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".")

using ParticleFilters
using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPGifs
using BeliefUpdaters
using POMCPOW
using ARDESPOT
using MCTS

using Blink
using Revise
using HumanSwitching
const HS = HumanSwitching
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
    if !ignore_commit_id && current_commit_id() !== scenario_data.git_commit_id
        return @error "Reproducing scenario with wrong commit ID!.
        If you are sure that this is still a good idea to do this, pass
        `ignore_commit_id=true` as kwarg to the call of `reproduce_scenario`."
    end

    return setup_test_scenario(scenario_data[:planner_key],
                               scenario_data[:i_run])
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

    # compose metadata
    git_commit_id = current_commit_id()
    md = Dict(:planner_key => planner_key,
              :i_run => i_run,
              :git_commit_id => git_commit_id)

    # compose the sim object for the `run_parallel` queue
    return Sim(simulation_model,
               planner,
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
    data = run_parallel(sims)
    return data
end

function visualize(model, sim_hist, planner)
    makegif(model, sim_hist, filename=joinpath(@__DIR__, "../renderings/visualize_debug.gif"),
            extra_initial=true, show_progress=true, render_kwargs=(sim_hist=sim_hist, show_info=true))
end

function tree(model, sim_hist, planner, step=1)
    beliefs = collect(eachstep(sim_hist, "b"))
    b = beliefs[step]
    a, info = action_info(planner, b, tree_in_info=true)
    inbrowser(D3Tree(info[:tree], init_expand=1), "chromium")
end
