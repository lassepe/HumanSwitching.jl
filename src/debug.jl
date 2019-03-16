using Distributed

const desired_nworkers = 1

if nworkers() != desired_nworkers
    wait(rmprocs(workers()))
    addprocs(desired_nworkers)
end
@show nworkers()

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Revise
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
    using Statistics
end

using DataFrames
using Blink
using Printf
using Compose
using Random
using ProgressMeter
using D3Trees

# TODO: move this to a package / module
@everywhere begin
    validation_hash(hist::SimHistory) = string(hash(collect(eachstep(hist, "s,a,sp,r,o"))))

    function final_state_type(m::HSModel, hist::SimHistory)
        final_state = last(collect(eachstep(hist, "sp")))
        if issuccess(m, final_state)
            return "success"
        elseif isfailure(m, final_state)
            return "failure"
        else
            return "nonterminal"
        end
    end
end

@everywhere begin
    using POMDPs
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
        CPUtic()
        action, info = action_info(timed_policy.p, x; kwargs...)
        info[:planning_cpu_time_us] = CPUtoq()
        return action, info
    end

    POMDPSimulators.problem(p::Policy) = p.problem
    POMDPSimulators.problem(p::TimedPolicy) = problem(p.p)
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
                            ignore_commit_id::Bool=false)
    # verify that the correct commit was checked out (because behavior of code
    # might have changed)
    if !ignore_commit_id && current_commit_id() != scenario_data.git_commit_id
        return @error "Reproducing scenario with wrong commit ID!.
        If you are sure that this is still a good idea to do this, pass
        `ignore_commit_id=true` as kwarg to the call of `reproduce_scenario`."
    end

    sim = setup_test_scenario(scenario_data[:hbm_key],
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

    planner_model = problem(sim.policy)
    println(discounted_reward(hist))

    return planner_model, hist, sim.policy
end

function setup_test_scenario(hbm_key::String, i_run::Int)
    rng = MersenneTwister(i_run)
    scenario_rng = MersenneTwister(i_run + 1)

    # TODO have a function
    # Input:
    # - human pose, robot pose, human target, robot target
    # - simulation hbm
    # - planning hbm
    #
    # Inside the function:
    # - construct simulation_model, belief_updater_model, planner_model
    # - simulation_model:
    #   - generate_hspomdp (as below) feeding poses (inits and targets)
    # - planning_model:
    #   - same as simulation but...
    #   - ...take hbm from hbm_key + polanner_hbm_map
    #   - ... ExactSensor and Identity PTNM
    #
    # simulation_model
    #   -

    room = RoomRep()
    human_init_pose = Pose(room.width/2, 1/10 * room.height, 0)
    robot_init_pose = Pose(room.width/2, 9/10 * room.height, 0)
    human_target_pose = robot_init_pose
    robot_target_pose = human_init_pose

    # the model of the "true" human...
    simulation_hbm = HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=10.0, beta_max=10.0, beta_resample_sigma=0.0)
    # ...and the "true" world model (used for generating samples)

    ptnm_cov = [0.01, 0.01, 0.01]
    simulation_model = generate_hspomdp(ExactPositionSensor(),
                                        simulation_hbm,
                                        HSGaussianNoisePTNM(pose_cov=ptnm_cov),
                                        deepcopy(rng),
                                        known_external_initstate=HSExternalState(human_init_pose,
                                                                                 robot_init_pose),
                                        robot_target=robot_target_pose)


    # compose the corresponding planning model
    belief_updater_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*9),
                                            planner_hbm_map()[hbm_key],
                                            HSIdentityPTNM(),
                                            simulation_model,
                                            deepcopy(rng))

    planning_model = generate_hspomdp(ExactPositionSensor(),                  # TODO: make this less verbose. Maybe have a funtion to sythetise these model tuples
                                      planner_hbm_map()[hbm_key],
                                      HSIdentityPTNM(),
                                      simulation_model,
                                      deepcopy(rng))

    n_particles = 2000
    # the blief updater is run with a stocahstic version of the world
    belief_updater = BasicParticleFilter(belief_updater_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
    # the policy plannes without a model as it is always the same action
    solver = POMCPOWSolver(tree_queries=12000, max_depth=70, criterion=MaxUCB(80),
                           k_action=5, alpha_action=0.1,
                           k_observation=5, alpha_observation=0.15,
                           check_repeat_obs=true,
                           check_repeat_act=true,
                           estimate_value=free_space_estimate, rng=deepcopy(rng))
    planner = solve(solver, planning_model)
    timed_planner = TimedPolicy(planner)

    # compose metadata
    git_commit_id = current_commit_id()
    md = Dict(:hbm_key => hbm_key,
              :i_run => i_run,
              :git_commit_id => git_commit_id)

    # compose the sim object for the `run_parallel` queue
    return Sim(simulation_model,
               timed_planner,
               belief_updater,
               initialstate_distribution(belief_updater_model),
               initialstate(simulation_model, deepcopy(rng)),
               rng=deepcopy(rng),
               max_steps=100,
               metadata=md)
end

"""
planner_hbm

Maps a hbm_key to a corresponding model instance. (to avoid storing the whole complex object)
"""
planner_hbm_map() = Dict{String, HumanBehaviorModel}(
                                                     "HumanBoltzmannModel1" =>
                                                     HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(Pose(RoomRep().width/2, 9/10 * RoomRep().height, 0)),
                                                                         beta_resample_sigma=0.0)
                                                    )
"""
test_parallel_sim

Run experiments over `planner_hbms` for `runs`
"""
function test_parallel_sim(runs::UnitRange{Int}; planner_hbms=planner_hbm_map())
    # queue of simulation instances...
    sims::Array{Sim, 1} = []
    # filled with scenarios for different hbms and runs
    for (hbm_key, planner_hbm) in planner_hbms, i_run in runs
        push!(sims, setup_test_scenario(hbm_key, i_run))
    end
    # Simulation is launched in parallel mode. In order for this to work, julia
    # musst be started as: `julia -p n`, where n is the number of
    # workers/processes
    data = run_parallel(sims) do sim::Sim, hist::SimHistory
        return [:n_steps => n_steps(hist),
                :discounted_reward => discounted_reward(hist),
                :hist_validation_hash => validation_hash(hist),
                :median_planning_time => median(ai[:planning_cpu_time_us] for ai in eachstep(hist, "ai")),
                :final_state_type => final_state_type(problem(sim), hist)]
    end
    return data
end

function visualize(planner_model, hist; filename::String="visualize_debug")
    makegif(planner_model, hist, filename=joinpath(@__DIR__, "../renderings/$filename.gif"),
            extra_initial=true, show_progress=true, render_kwargs=(sim_hist=hist, show_info=true))
end

function tree(model::POMDP, hist::SimHistory, policy::Policy, step=30)
    beliefs = collect(eachstep(hist, "b"))
    b = beliefs[step]
    a, info = action_info(policy, b, tree_in_info=true)
    inbrowser(D3Tree(info[:tree], init_expand=1), "chromium")
end

function debug(data, idx)
    viz = reproduce_scenario(data[idx, :])
    visualize(viz[1:2]...)
end
