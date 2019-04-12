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
using Parameters

"""
Define three dictionaries that:
    1. Maps a key to a corresponding problem instance.
    2. Maps a key to a corresponding model instance for the planner.
    3. Maps a key to a corresponding true model instance for the simulator.
"""
# order (human_start_pos, robot_start_pos, human_goal_pos, robot_goal_pos)
const SimulationHBMEntry = HumanBehaviorModel

@with_kw struct ProblemInstance
    human_start_pos::Union{Pos, Nothing} = nothing
    robot_start_pos::Union{Pos, Nothing} = nothing
    robot_goal_pos::Union{Pos, Nothing} = nothing
    room::Room = Room()
    force_nontrivial::Bool = false
    human_goals::Union{Function, Nothing} = nothing
end

@with_kw struct PlannerSetup{HBM<:HumanBehaviorModel}
    hbm::HBM
    n_particles::Int
    epsilon::Float64
end

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

    sim = setup_test_scenario(scenario_data[:pi_key], scenario_data[:simulation_hbm_key],
                              scenario_data[:planner_hbm_key], scenario_data[:solver_setup_key],
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
        problem_instance [ProblemInstance]: defines the setup of the problem (e.g. initial conditionsa and goals for all agents)
        simulation_hbm [HumanBehaviorModel]: The "true" human model used by the simulator.
        belief_updater_hbm [HumanBehaviorModel]: The human model used by the belief updater.
        planner_hbm [HumanBehaviorModel]: The human model used by the planner.

    Returns:
        simulation_model [HSModel]: The model of the world used by the simulator.
        belief_updater_model [HSModel]: The model of the world used by the belief updater.
        planner_model [HSModel]: The model of the world used by the planner.
    """

    ptnm_cov = [0.01, 0.01]
    if problem_instance.force_nontrivial
        if isnothing(problem_instance.robot_start_pos) && isnothing(problem_instance.robot_goal_pos) && isnothing(problem_instance.human_start_pos)
            simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                             simulation_hbm,
                                                             HSGaussianNoisePTNM(pos_cov=ptnm_cov),
                                                             deepcopy(rng))
        else
            throw("Illegal problem instance!")
        end
    else
        simulation_model = generate_hspomdp(ExactPositionSensor(),
                                            simulation_hbm,
                                            HSGaussianNoisePTNM(pos_cov=ptnm_cov),
                                            deepcopy(rng),
                                            known_external_initstate=HSExternalState(problem_instance.human_start_pos, problem_instance.robot_start_pos),
                                            robot_goal=problem_instance.robot_goal_pos)
    end


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

function belief_updater_from_planner_model(planner_setup::PlannerSetup{<:HumanBoltzmannModel})
    # clone the model but set the new epsilon
    return HumanBoltzmannModel(beta_min=planner_setup.hbm.beta_min,
                               beta_max=planner_setup.hbm.beta_max,
                               betas=planner_setup.hbm.betas,
                               epsilon=planner_setup.epsilon,
                               reward_model=planner_setup.hbm.reward_model,
                               aspace=planner_setup.hbm.aspace)
end

function belief_updater_from_planner_model(planner_setup::PlannerSetup{<:HumanConstVelBehavior})
    # clone the model but set the new epsilon
    return HumanConstVelBehavior(vel_max=planner_setup.hbm.vel_max,
                                 vel_resample_sigma=planner_setup.epsilon)
end

function belief_updater_from_planner_model(planner_setup::PlannerSetup{<:HumanMultiGoalBoltzmann})
    # clone the model but set the new epsilon
    return HumanMultiGoalBoltzmann(beta_min=planner_setup.hbm.beta_min,
                                   beta_max=planner_setup.hbm.beta_max,
                                   goals=planner_setup.hbm.goals,
                                   next_goal_generator=planner_setup.hbm.next_goal_generator,
                                   initial_goal_generator=planner_setup.hbm.initial_goal_generator,
                                   vel_max=planner_setup.hbm.vel_max,
                                   goal_resample_sigma=planner_setup.hbm.goal_resample_sigma,
                                   beta_resample_sigma=planner_setup.epsilon,
                                   aspace=planner_setup.hbm.aspace)
end

function setup_test_scenario(pi_key::String, simulation_hbm_key::String, planner_hbm_key::String, solver_setup_key::String, i_run::Int)
    rng = MersenneTwister(i_run)

    # Load in the given instance keys.
    problem_instance = problem_instance_map()[pi_key]
    planner_setup = planner_hbm_map(problem_instance)[planner_hbm_key]
    simulation_hbm = simulation_hbm_map(problem_instance, i_run)[simulation_hbm_key]

    # Construct belief updater.
    belief_updater_hbm = belief_updater_from_planner_model(planner_setup)

    # Construct models.
    simulation_model, belief_updater_model, planner_model = construct_models(rng, problem_instance, simulation_hbm,
                                                                             belief_updater_hbm, planner_setup.hbm)

    # the belief updater is run with a stochastic version of the world
    belief_updater = BasicParticleFilter(belief_updater_model, SharedExternalStateResampler(planner_setup.n_particles), planner_setup.n_particles, deepcopy(rng))
    solver = solver_setup_map(planner_setup, planner_model, rng)[solver_setup_key]
    planner = solve(solver, planner_model)

    # compose metadata
    git_commit_id = (has_uncommited_changes() ? "dirty::" : "") * current_commit_id()
    md = Dict(:pi_key => pi_key,
              :simulation_hbm_key => simulation_hbm_key,
              :planner_hbm_key => planner_hbm_key,
              :solver_setup_key => solver_setup_key,
              :i_run => i_run,
              :git_commit_id => git_commit_id)

    # compose the sim object for the `run_parallel` queue
    return Sim(simulation_model,
               TimedPolicy(planner),
               TimedUpdater(belief_updater),
               initialstate_distribution(belief_updater_model),
               initialstate(simulation_model, deepcopy(rng)),
               rng=deepcopy(rng),
               max_steps=200,
               metadata=md)
end

function problem_instance_map()
    room = Room()
    return Dict{String, ProblemInstance}(
    # "DiagonalAcross" => ProblemInstance(human_start_pos=Pos(1/10 * room.width, 1/10 * room.height),
    #                                     robot_start_pos=Pos(8/10 * room.width, 4/10 * room.height),
    #                                     robot_goal_pos=Pos(1/10 * room.width, 9/10 * room.height),
    #                                     room=room),
    # "FrontalCollision" => ProblemInstance(human_start_pos=Pos(1/2 * room.width, 1/10 * room.height),
    #                                       robot_start_pos=Pos(1/2 * room.width, 9/10 * room.height),
    #                                       robot_goal_pos=Pos(1/2 * room.width, 1/10 * room.height),
    #                                       room=room),
    #"RandomNontrivial" => ProblemInstance(force_nontrivial=true,
    #                                      room=room,
    #                                      human_goals=symmetric_goals),
    #"DiningHallNontrivial" => ProblemInstance(force_nontrivial=true,
    #                                          room=room,
    #                                          human_goals=dining_hall_goals)
    "CornerGoalsNonTrivial" => ProblemInstance(force_nontrivial=true,
                                             room=room,
                                             human_goals=corner_positions)
   )
end

abs_pos(rel_pos::Pos, room::Room) = Pos(rel_pos.x*room.width, rel_pos.y*room.height)

symmetric_goals(room::Room) = vcat(corner_positions(room),
                                    [abs_pos(p, room) for p in [Pos(0.5, 0.5),
                                                                Pos(0.5, 0.1), Pos(0.5, 0.9),
                                                                Pos(0.1, 0.5), Pos(0.9, 0.5)]])

dining_hall_goals(room::Room) = vcat([abs_pos(p, room) for p in [Pos(0.2, 0.7), Pos(0.5, 0.7), Pos(0.8, 0.7),
                                                                 Pos(0.2, 0.3), Pos(0.5, 0.3), Pos(0.8, 0.3)]])

function planner_hbm_map(problem_instance::ProblemInstance)
    return Dict{String, PlannerSetup}(
        # "HumanConstVelBehavior" => (HumanConstVelBehavior(vel_max=1, vel_resample_sigma=0.0), 0.05),
        # "HumanBoltzmannModel_PI/8" => (HumanBoltzmannModel(reward_model=HumanSingleGoalRewardModel(human_goal_pos),
        "HumanMultiGoalBoltzmann_all_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room),
                                                                                        beta_min=0.1, beta_max=50,
                                                                                        goal_resample_sigma=0.1,
                                                                                        beta_resample_sigma=0.0),
                                                            epsilon=0.02,
                                                            n_particles=6000),
        "HumanMultiGoalBoltzmann_3_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room)[1:3],
                                                                                        beta_min=0.1, beta_max=50,
                                                                                        goal_resample_sigma=0.1,
                                                                                        beta_resample_sigma=0.0),
                                                            epsilon=0.02,
                                                            n_particles=4500),
        "HumanConstVelBehavior" => PlannerSetup(hbm=HumanConstVelBehavior(vel_max=1, vel_resample_sigma=0.0),
                                                epsilon=0.1,
                                                n_particles=1000),
        # "HumanMultiGoalBoltzmann_5_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room)[5:9],
        #                                                                               beta_min=0.1, beta_max=50,
        #                                                                               goal_resample_sigma=0.1,
        #                                                                               beta_resample_sigma=0.0),
        #                                                   epsilon=0.02,
        #                                                   n_particles=3000),
        # "HumanMultiGoalBoltzmann_4_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room)[6:9],
        #                                                                               beta_min=0.1, beta_max=50,
        #                                                                               goal_resample_sigma=0.1,
        #                                                                               beta_resample_sigma=0.0),
        #                                                   epsilon=0.02,
        #                                                   n_particles=2000),
       )
end

function solver_setup_map(planner_setup::PlannerSetup, planner_model::HSModel, rng::MersenneTwister)
    return Dict{String, Solver}(
                                # TODO: DESPOT needs value estimate at end to reduce rollout length!
                                "DESPOT" => begin
                                    default_policy = StraightToGoal(planner_model)
                                    # alternative lower bound: DefaultPolicyLB(default_policy)
                                    bounds = IndependentBounds(DefaultPolicyLB(default_policy), free_space_estimate, check_terminal=true)

                                    solver = DESPOTSolver(K=200, D=60, max_trials=10, T_max=Inf, lambda=0.01,
                                                          bounds=bounds, rng=deepcopy(rng), tree_in_info=true)
                                end,
                                "POMCPOW" => begin
                                    # TODO: use separate setting for tree quries
                                    solver = POMCPOWSolver(tree_queries=floor(planner_setup.n_particles*2.5), max_depth=70, criterion=MaxUCB(80),
                                                           k_observation=5, alpha_observation=1.0/30.0,
                                                           enable_action_pw=false,
                                                           check_repeat_obs=!(planner_setup.hbm isa HumanConstVelBehavior),
                                                           check_repeat_act=true,
                                                           estimate_value=free_space_estimate, rng=deepcopy(rng))
                                end
                               )
end

function simulation_hbm_map(problem_instance::ProblemInstance, i_run::Int)
    simulation_rng = MersenneTwister(i_run + 1)
    return Dict{String, SimulationHBMEntry}(
        "HumanMultiGoalBoltzmann_all_goals" => HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room),
                                                                       beta_min=50, beta_max=50,
                                                                       goal_resample_sigma=0.1,
                                                                       beta_resample_sigma=0.0)
       )
end

function noisy_waypoints(start_p::Pos, end_p::Pos, n_waypoints::Int, rng::AbstractRNG, sigma::Float64)
    waypoints = []
    for i = 1:n_waypoints
        direct_waypoint::Pos = start_p + (end_p - start_p) * i/(n_waypoints + 1)
        noisy_waypoint = Pos(direct_waypoint.x + randn(rng) * sigma,
                             direct_waypoint.y + randn(rng) * sigma)
        push!(waypoints, noisy_waypoint)
    end
    push!(waypoints, end_p)
end

"""
Fuction that runs experiments [runs] times.
"""
function test_parallel_sim(runs::UnitRange{Int}, solver_setup_key::String="POMCPOW"; ignore_uncommited_changes::Bool=false)
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
        for simulation_hbm_key in keys(simulation_hbms), planner_hbm_key in keys(planner_hbms)
                push!(sims, setup_test_scenario(pi_key, simulation_hbm_key, planner_hbm_key, solver_setup_key, i_run))
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
                :median_updater_time => median(ui[:updater_cpu_time_us] for ui in eachstep(hist, "ui")),
                :final_state_type => final_state_type(problem(sim), hist),
                :free_space_estimate => free_space_estimate(mdp(problem(sim)), first(collect(s for s in eachstep(hist, "s"))))]
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
    h: $(o.human_pos)
    r: $(o.robot_pos)
    """
end

function debug(data, idx; kwargs...)
    viz = reproduce_scenario(data[idx, :]; kwargs...)
    visualize(viz[1:2]..., filename="$idx")
end

function debug(data; kwargs...)
    for idx in 1:nrow(data)
        debug(data, idx; kwargs...)
    end
end
