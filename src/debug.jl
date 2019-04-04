using Distributed

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
    validation_hash(hist::SimHistory) = string(hash(collect((sp.external.robot_pose,
                                                             sp.external.human_pose)
                                                            for sp in eachstep(hist, "sp"))))

    function final_state_type(m::HSModel, hist::SimHistory)
        final_state = last(collect(eachstep(hist, "sp")))
        if HS.issuccess(m, final_state)
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
        info[:planner_cpu_time_us] = CPUtoq()
        return action, info
    end

    POMDPSimulators.problem(p::Policy) = p.problem
    POMDPSimulators.problem(p::TimedPolicy) = problem(p.p)
end

"""
Define three dictionaries that:
    1. Maps a key to a corresponding problem instance.
    2. Maps a key to a corresponding model instance for the planner.
    3. Maps a key to a corresponding true model instance for the simulator.
"""
# order (human_start_pose, robot_start_pose, human_target_pose, robot_target_pose)
const ProblemInstance = Tuple{Pose, Pose, Pose, Pose}
const PlannerHBMEntry = Tuple{HumanBehaviorModel, Float64}
const SimulationHBMEntry = Tuple{HumanBehaviorModel}

"""
current_commit_id

Determines the git commit id of the `HEAD` of this repo.
"""
current_commit_id() = chomp(read(`git rev-parse --verify HEAD`, String))
has_uncommited_changes() = !isempty(read(`git diff-index HEAD --`, String))

"""
reproduce_scenario

Reproduces the simulation environment for a given DataFrameRow
"""
function reproduce_scenario(scenario_data::DataFrameRow;
                            ignore_commit_id::Bool=false,
                            ignore_uncommited_changes::Bool=false)
    # verify that the correct commit was checked out (because behavior of code
    # might have changed)
    if !ignore_commit_id && current_commit_id() != scenario_data.git_commit_id
        throw("Reproducing scenario with wrong commit ID!.
        If you are sure that this is still a good idea to do this, pass
        `ignore_commit_id=true` as kwarg to the call of `reproduce_scenario`.")
    end

    if !ignore_uncommited_changes && has_uncommited_changes()
        throw("There are uncommited changes. The stored commit-id might not be meaning full.
        to ignore uncommited changes, set the corresponding kwarg.")
    end

    sim = setup_test_scenario(scenario_data[:pi_key], scenario_data[:simulation_hbm_key], scenario_data[:planner_hbm_key], scenario_data[:i_run])

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
    println("Discounted reward: $(discounted_reward(hist))")
    println("Validation hash: $(validation_hash(hist))")

    return planner_model, hist, sim.policy
end

function construct_models(rng::AbstractRNG, problem_instance::ProblemInstance,
                          simulation_hbm::HumanBehaviorModel, belief_updater_hbm::HumanBehaviorModel, planner_hbm::HumanBehaviorModel)
    """
    Function that constructs the simulation model, the belief updater model, and the planner model.

    Params:
        rng [AbstractRNG]: The random seed to be used for these models.
        human_start_pose [Pose]: The initial position of the human.
        robot_start_pose [Pose]: The initial position of the robot.
        human_target_pose [Pose]: The final target position of the human.
        robot_target_pose [Pose]: The final target position of the robot.
        simulation_hbm [HumanBehaviorModel]: The "true" human model used by the simulator.
        belief_updater_hbm [HumanBehaviorModel]: The human model used by the belief updater.
        planner_hbm [HumanBehaviorModel]: The human model used by the planner.

    Returns:
        simulation_model [HSModel]: The model of the world used by the simulator.
        belief_updater_model [HSModel]: The model of the world used by the belief updater.
        planner_model [HSModel]: The model of the world used by the planner.
    """

    (human_start_pose, robot_start_pose, human_target_pose, robot_target_pose) = problem_instance

    ptnm_cov = [0.01, 0.01, 0.01]
    simulation_model = generate_hspomdp(ExactPositionSensor(),
                                        simulation_hbm,
                                        HSGaussianNoisePTNM(pose_cov=ptnm_cov),
                                        deepcopy(rng),
                                        known_external_initstate=HSExternalState(human_start_pose, robot_start_pose),
                                        robot_target=robot_target_pose)

    belief_updater_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*9),
                                            belief_updater_hbm,
                                            HSIdentityPTNM(),
                                            simulation_model,
                                            deepcopy(rng))

    planner_model = generate_hspomdp(ExactPositionSensor(),
                                     planner_hbm,
                                     HSIdentityPTNM(),
                                     simulation_model,
                                     deepcopy(rng))

    return simulation_model, belief_updater_model, planner_model
end

function belief_updater_from_planner_model(planner_hbm::HumanBoltzmannModel, epsilon::Float64)
    return HumanBoltzmannModel(reward_model=planner_hbm.reward_model, betas=planner_hbm.betas, epsilon=epsilon)
end

function belief_updater_from_planner_model(planner_hbm::HumanConstVelBehavior, vel_resample_sigma::Float64)
    return HumanConstVelBehavior(vel_max=planner_hbm.vel_max, vel_resample_sigma=vel_resample_sigma)
end

function setup_test_scenario(pi_key::String, simulation_hbm_key::String, planner_hbm_key::String, i_run::Int)
    rng = MersenneTwister(i_run)

    # Load in the given instance keys.
    problem_instance = problem_instance_map()[pi_key]
    (planner_hbm, epsilon) = planner_hbm_map(problem_instance)[planner_hbm_key]
    (simulation_hbm,) = simulation_hbm_map(problem_instance, i_run)[simulation_hbm_key]

    # Construct belief updater.
    belief_updater_hbm = belief_updater_from_planner_model(planner_hbm, epsilon)

    # Construct models.
    simulation_model, belief_updater_model, planner_model = construct_models(rng, problem_instance, simulation_hbm,
                                                                             belief_updater_hbm, planner_hbm)

    n_particles = 2000
    # the belief updater is run with a stochastic version of the world
    belief_updater = BasicParticleFilter(belief_updater_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
    # the policy plannes without a model as it is always the same action
    solver = POMCPOWSolver(tree_queries=12000, max_depth=70, criterion=MaxUCB(80),
                           k_observation=5, alpha_observation=1.0/30.0,
                           enable_action_pw=false,
                           check_repeat_obs=!(planner_hbm isa HumanConstVelBehavior),
                           check_repeat_act=true,
                           estimate_value=free_space_estimate, rng=deepcopy(rng))
    planner = solve(solver, planner_model)
    timed_planner = TimedPolicy(planner)

    # compose metadata
    git_commit_id = current_commit_id()
    md = Dict(:pi_key => pi_key,
              :simulation_hbm_key => simulation_hbm_key,
              :planner_hbm_key => planner_hbm_key,
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

function problem_instance_map()
    room = RoomRep()
    return Dict{String, ProblemInstance}(
        "DiagonalAcross" => (Pose(1/10 * room.width, 1/10 * room.height, 0), Pose(8/10 * room.width, 4/10 * room.height, 0),
                             Pose(9/10 * room.width, 9/10 * room.height, 0), Pose(1/10 * room.width, 9/10 * room.height, 0)),
        "FrontalCollision" => (Pose(1/2 * room.width, 1/10 * room.height, 0), Pose(1/2 * room.width, 9/10 * room.height, 0),
                               Pose(1/2 * room.width, 9/10 * room.height, 0), Pose(1/2 * room.width, 1/10 * room.height, 0))
       )
end

function planner_hbm_map(problem_instance::ProblemInstance)
    human_target_pose = problem_instance[3]
    return Dict{String, PlannerHBMEntry}(
        "HumanConstVelBehavior" => (HumanConstVelBehavior(vel_max=1, vel_resample_sigma=0.0), 0.05),
        "HumanBoltzmannModel_PI/12" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose),
                                                            aspace=HS.gen_human_aspace(pi/12)), 0.01),
        "HumanBoltzmannModel_PI/8" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose),
                                                           aspace=HS.gen_human_aspace(pi/8)), 0.01),
        "HumanBoltzmannModel_PI/4" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose),
                                                          aspace=HS.gen_human_aspace(pi/4)), 0.01)
       )
end

function simulation_hbm_map(problem_instance::ProblemInstance, i_run::Int)
    human_start_pose = problem_instance[1]
    human_target_pose = problem_instance[3]
    simulation_rng = MersenneTwister(i_run + 1)
    return Dict{String, SimulationHBMEntry}(
        "HumanBoltzmannModel0.1" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=0.1, beta_max=0.1),),
        "HumanBoltzmannModel1.0" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=1.0, beta_max=1.0),),
        "HumanBoltzmannModel5.0" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=5.0, beta_max=5.0),),
        "HumanBoltzmannModel10.0" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=10.0, beta_max=10.0),),
        "HumanBoltzmannModel15.0" => (HumanBoltzmannModel(reward_model=HumanSingleTargetRewardModel(human_target_pose), beta_min=15.0, beta_max=15.0),),
        "WayPoints_n5_sig1.0" => (HumanPIDBehavior(target_sequence=noisy_waypoints(human_start_pose, human_target_pose, 5, simulation_rng, 1.0)),),
                                          )
end

function noisy_waypoints(start_p::Pose, end_p::Pose, n_waypoints::Int, rng::AbstractRNG, sigma::Float64)
    waypoints = []
    for i = 1:n_waypoints
        direct_waypoint::Pose = start_p + (end_p - start_p) * i/(n_waypoints + 1)
        noisy_waypoint = Pose(direct_waypoint.x + randn(rng) * sigma,
                              direct_waypoint.y + randn(rng) * sigma,
                              direct_waypoint.phi)
        push!(waypoints, noisy_waypoint)
    end
    push!(waypoints, end_p)
end

"""
Fuction that runs experiments [runs] times.
"""
function test_parallel_sim(runs::UnitRange{Int}; ignore_uncommited_changes::Bool=false)
    if !ignore_uncommited_changes && has_uncommited_changes()
        throw("There are uncommited changes. The stored commit-id might not be meaning full.
        to ignore uncommited changes, set the corresponding kwarg.")
    end

    # Create the problem instance maps:
    problem_instances = problem_instance_map()

    # Queue of simulation instances to be filled with scenarios for different hbms and runs:
    sims::Array{Sim, 1} = []

    for (pi_key, pi_entry) in problem_instances, i_run in runs
        planner_hbms = planner_hbm_map(pi_entry)
        simulation_hbms = simulation_hbm_map(pi_entry, i_run)
        for simulation_hbm_key in keys(simulation_hbms)
            for planner_hbm_key in keys(planner_hbms)
                push!(sims, setup_test_scenario(pi_key, simulation_hbm_key, planner_hbm_key, i_run))
            end
        end
    end
    # Simulation is launched in parallel mode. In order for this to work, julia
    # musst be started as: `julia -p n`, where n is the number of
    # workers/processes
    data = run_parallel(sims) do sim::Sim, hist::SimHistory
        return [:n_steps => n_steps(hist),
                :discounted_reward => discounted_reward(hist),
                :hist_validation_hash => validation_hash(hist),
                :median_planner_time => median(ai[:planner_cpu_time_us] for ai in eachstep(hist, "ai")),
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
    inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")
end

function MCTS.node_tag(o::HSExternalState)
    """
    h: $(o.human_pose)
    r: $(o.robot_pose)
    """
end

function debug(data, idx; kwargs...)
    viz = reproduce_scenario(data[idx, :]; kwargs...)
    visualize(viz[1:2]...)
end
