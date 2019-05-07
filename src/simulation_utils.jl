POMDPSimulators.problem(p::Policy) = p.problem
POMDPSimulators.problem(p::DESPOTPlanner) = p.pomdp
POMDPSimulators.problem(p::TimedPolicy) = problem(p.p)
POMDPSimulators.problem(p::ProbObstaclePolicy) = p.pomdp

validation_hash(hist::SimHistory) = string(hash(collect((sp.external.robot_pos,
                                                         sp.external.human_pos)
                                                        for sp in eachstep(hist, "sp"))))

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
    human_obstacles::Function = no_obstacles
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
has_uncommited_changes() = LibGit2.isdirty(LibGit2.GitRepo(base_dir()))

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

    i_run = scenario_data[:i_run]
    pi_key = scenario_data[:pi_key]
    simulation_hbm_key = scenario_data[:simulation_hbm_key]
    simulation_model = setup_simulation_model(MersenneTwister(i_run), problem_instance_map()[pi_key], simulation_hbm_key, i_run)
    sim = setup_test_scenario(pi_key, simulation_model, simulation_hbm_key, scenario_data[:planner_hbm_key], scenario_data[:solver_setup_key], i_run)

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

function setup_simulation_model(rng::AbstractRNG, problem_instance::ProblemInstance, simulation_hbm_key::String, i_run::Int)

    simulation_hbm = simulation_hbm_map(problem_instance, i_run)[simulation_hbm_key]

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
        simulation_model = HSPOMDP(ExactPositionSensor(),
                                   gen_hsmdp(deepcopy(rng),
                                             human_behavior_model=simulation_hbm,
                                             physical_transition_noise_model=HSGaussianNoisePTNM(pos_cov=ptnm_cov),
                                             known_external_initstate=HSExternalState(problem_instance.human_start_pos, problem_instance.robot_start_pos),
                                             robot_goal=problem_instance.robot_goal_pos)
                                  )
    end

    return simulation_model
end

function construct_models(rng::AbstractRNG, simulation_model::HSModel, belief_updater_hbm::HumanBehaviorModel, planner_hbm::HumanBehaviorModel)
    belief_updater_model = generate_from_template(simulation_model, deepcopy(rng),
                                                  sensor=NoisyPositionSensor(physical_transition_noise_model(simulation_model).pos_cov*9),
                                                  human_behavior_model=belief_updater_hbm,
                                                  physical_transition_noise_model=HSIdentityPTNM())

    planner_model = generate_from_template(simulation_model, deepcopy(rng),
                                           sensor=ExactPositionSensor(),
                                           human_behavior_model=planner_hbm,
                                           physical_transition_noise_model=HSIdentityPTNM())

    return belief_updater_model, planner_model
end

function belief_updater_from_planner_model(planner_setup::PlannerSetup{<:HumanConstVelBehavior})
    # clone the model but set the new epsilon
    return HumanConstVelBehavior(speed_max=planner_setup.hbm.speed_max,
                                 vel_resample_sigma=planner_setup.epsilon)
end

function belief_updater_from_planner_model(planner_setup::PlannerSetup{<:HumanMultiGoalBoltzmann})
    # clone the model but set the new epsilon
    return HumanMultiGoalBoltzmann(beta_min=planner_setup.hbm.beta_min,
                                   beta_max=planner_setup.hbm.beta_max,
                                   goals=planner_setup.hbm.goals,
                                   next_goal_generator=planner_setup.hbm.next_goal_generator,
                                   initial_goal_generator=planner_setup.hbm.initial_goal_generator,
                                   speed_max=planner_setup.hbm.speed_max,
                                   goal_resample_sigma=planner_setup.hbm.goal_resample_sigma,
                                   beta_resample_sigma=planner_setup.epsilon,
                                   aspace=planner_setup.hbm.aspace)
end

function setup_test_scenario(pi_key::String, simulation_model::HSModel, simulation_hbm_key::String, planner_hbm_key::String, solver_setup_key::String, i_run::Int)
    rng = MersenneTwister(i_run)

    # Load in the given instance keys.
    problem_instance = problem_instance_map()[pi_key]
    planner_setup = planner_hbm_map(problem_instance)[planner_hbm_key]

    # Construct belief updater.
    belief_updater_hbm = belief_updater_from_planner_model(planner_setup)

    # Construct models.
    belief_updater_model, planner_model = construct_models(deepcopy(rng), simulation_model, belief_updater_hbm, planner_setup.hbm)
    # the belief updater is run with a stochastic version of the world
    belief_updater = BasicParticleFilter(belief_updater_model, SharedExternalStateResampler(planner_setup.n_particles), planner_setup.n_particles, deepcopy(rng))
    solver = solver_setup_map(planner_setup, planner_model, deepcopy(rng))[solver_setup_key]
    planner = solver isa Policy ? solver : solve(solver, planner_model)

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
    # "RandomNontrivial" => ProblemInstance(force_nontrivial=true,
    #                                       room=room,
    #                                       human_goals=symmetric_goals),
    # "DiningHallNontrivial" => ProblemInstance(force_nontrivial=true,
    #                                           room=room,
    #                                           human_goals=dining_hall_goals),
    "CornerGoalsNonTrivial" => ProblemInstance(force_nontrivial=true,
                                               room=room,
                                               human_goals=corner_positions,
                                               human_obstacles=outer_obstacles)
   )
end

abs_pos(rel_pos::Pos, room::Room) = Pos(rel_pos.x*room.width, rel_pos.y*room.height)

symmetric_goals(room::Room) = vcat(corner_positions(room),
                                    [abs_pos(p, room) for p in [Pos(0.5, 0.5),
                                                                Pos(0.5, 0.1), Pos(0.5, 0.9),
                                                                Pos(0.1, 0.5), Pos(0.9, 0.5)]])

dining_hall_goals(room::Room) = vcat([abs_pos(p, room) for p in [Pos(0.2, 0.7), Pos(0.5, 0.7), Pos(0.8, 0.7),
                                                                 Pos(0.2, 0.3), Pos(0.5, 0.3), Pos(0.8, 0.3)]])

no_obstacles(room::Room) = Circle[]
outer_obstacles(room::Room) = vcat([Circle(abs_pos(p, room), 1.2) for p in [Pos(0.2, 0.5), Pos(0.8, 0.5),
                                                                            Pos(0.5, 0.2), Pos(0.5, 0.8)]])

function planner_hbm_map(problem_instance::ProblemInstance)
    return Dict{String, PlannerSetup}(
        "HumanMultiGoalBoltzmann_all_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room),
                                                                                        beta_min=0.1, beta_max=50,
                                                                                        goal_resample_sigma=0.05,
                                                                                        beta_resample_sigma=0.0),
                                                            epsilon=0.02,
                                                            n_particles=8000),
        "HumanMultiGoalBoltzmann_half_goals" => PlannerSetup(hbm=HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room)[1:cld(length(problem_instance.human_goals(problem_instance.room)), 2)],
                                                                                        beta_min=0.1, beta_max=50,
                                                                                        goal_resample_sigma=0.05,
                                                                                        beta_resample_sigma=0.0),
                                                            epsilon=0.02,
                                                            n_particles=5000),
        "HumanConstVelBehavior" => PlannerSetup(hbm=HumanConstVelBehavior(vel_resample_sigma=0.0),
                                                epsilon=0.1,
                                                n_particles=2000)
       )
end

function solver_setup_map(planner_setup::PlannerSetup, planner_model::HSModel, rng::MersenneTwister)
    return Dict{String, Union{Solver, Policy}}(
                                               "POMCP" => begin
                                                   POMCPSolver(max_depth=20, c=1.0, tree_queries=1000, estimate_value=free_space_estimate,
                                                       max_time=Inf, rng=deepcopy(rng), tree_in_info=true)
                                               end,
                                               # TODO: DESPOT needs value estimate at end to reduce rollout length!
                                               "DESPOT" => begin
                                                   default_policy = StraightToGoal(planner_model)
                                                   bounds = IndependentBounds(DefaultPolicyLB(default_policy), free_space_estimate, check_terminal=true)

                                                   DESPOTSolver(K=cld(planner_setup.n_particles, 10), D=70, max_trials=20, T_max=Inf, lambda=0.00001,
                                                                bounds=bounds, rng=deepcopy(rng), tree_in_info=true)
                                               end,
                                               "POMCPOW" => begin
                                                   # TODO: use separate setting for tree quries
                                                   POMCPOWSolver(tree_queries=floor(planner_setup.n_particles*2.5), max_depth=70, criterion=MaxUCB(500),
                                                                 k_observation=5, alpha_observation=1.0/30.0,
                                                                 enable_action_pw=false,
                                                                 check_repeat_obs=!(planner_setup.hbm isa HumanConstVelBehavior),
                                                                 check_repeat_act=true,
                                                                 estimate_value=free_space_estimate, rng=deepcopy(rng))
                                               end,
                                               "ProbObstacles" => begin
                                                   n_particles = 1000
                                                   human_predictor = PredictModel{HSHumanState}((hs::HSHumanState, rng::AbstractRNG) -> begin
                                                                                                    human_pos, hbs = hs
                                                                                                    human_transition(hbs, human_behavior_model(planner_model), planner_model, human_pos, rng)
                                                                                                end)
                                                   pbp = ParticleBeliefPropagator(human_predictor, n_particles, deepcopy(rng))
                                                   ProbObstacleSolver(belief_propagator=pbp)
                                               end,
					                           "GapChecking" => begin
                                                   n_particles = 1000
                                                   human_predictor = PredictModel{HSHumanState}((hs::HSHumanState, rng::AbstractRNG) -> begin
                                                                                                    human_pos, hbs = hs
                                                                                                    human_transition(hbs, human_behavior_model(planner_model), planner_model, human_pos, rng)
                                                                                                end)
                                                   pbp = ParticleBeliefPropagator(human_predictor, n_particles, deepcopy(rng))
                                                   prob_obstacle_policy = solve(ProbObstacleSolver(belief_propagator=pbp), planner_model)
                                                   GapCheckingPolicy(smarter_policy=prob_obstacle_policy, problem=planner_model)
                                               end,
					                           "StraightToGoal" => StraightToGoal(planner_model)
                                              )
end

function simulation_hbm_map(problem_instance::ProblemInstance, i_run::Int)
    simulation_rng = MersenneTwister(i_run + 1)
    return Dict{String, SimulationHBMEntry}(
        "HumanMultiGoalBoltzmann_all_goals" => HumanMultiGoalBoltzmann(goals=problem_instance.human_goals(problem_instance.room),
                                                                       beta_min=50, beta_max=50,
                                                                       goal_resample_sigma=0.05,
                                                                       beta_resample_sigma=0.0),
        "HumanDeterministicPlanner" => HumanDeterministicPlanner(goals=problem_instance.human_goals(problem_instance.room),
                                                                 obstacles=problem_instance.human_obstacles(problem_instance.room))
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

Note: If keyset is not specified, all entries of the corresponding maps will be used.
"""
function parallel_sim(runs::UnitRange{Int}, solver_setup_keys::Array{String};
                      problem_instance_keys::Union{Array{String}, Nothing} = nothing,
                      planner_hbm_keys::Union{Array{String}, Nothing} = nothing,
                      simulation_hbm_keys::Union{Array{String}, Nothing} = nothing,
                      ignore_uncommited_changes::Bool=false)

    if !ignore_uncommited_changes && has_uncommited_changes()
        throw("There are uncommited changes. The stored commit-id might not be meaning full.
              to ignore uncommited changes, set the corresponding kwarg.")
    end

    # if no keys are specified, we use all of them from the map
    if isnothing(problem_instance_keys)
        problem_instance_keys = [k for k in keys(problem_instance_map())]
    else
        # check whether all keys are valid
        @assert all(in.(problem_instance_keys, (keys(problem_instance_map()),)))
    end

    # Queue of simulation instances to be filled with scenarios for different hbms and runs:
    sims = []

    @info "Generating scenarios..."
    @time @sync for i_run in runs, pi_key in problem_instance_keys
        pi_entry = problem_instance_map()[pi_key]
        # check whether all keys are valid
        if isnothing(planner_hbm_keys)
            planner_hbm_keys = [k for k in keys(planner_hbm_map(pi_entry))]
        else
            @assert all(in.(planner_hbm_keys, (keys(planner_hbm_map(pi_entry)),)))
        end
        if isnothing(simulation_hbm_keys)
            simulation_hbm_keys = [k for k in keys(simulation_hbm_map(pi_entry, i_run))]
        else
            @assert all(in.(simulation_hbm_keys, (keys(simulation_hbm_map(pi_entry, i_run)),)))
        end
        for simulation_hbm_key in simulation_hbm_keys
            # computing simultion_models might take a little longer (due to
            # adversarial scenario generation). Thus, we perform this in an
            # asynchronous fashion distributed over multiple workers.
            @async append!(sims, @fetch begin
                               simulation_model = setup_simulation_model(MersenneTwister(i_run), pi_entry, simulation_hbm_key, i_run)
                               [setup_test_scenario(pi_key, simulation_model, simulation_hbm_key, planner_hbm_key, solver_setup_key, i_run) for planner_hbm_key in planner_hbm_keys, solver_setup_key in solver_setup_keys]
                           end)
        end
    end
    # Simulation is launched in parallel mode. In order for this to work, julia
    # must be started as: `julia -p n`, where n is the number of
    # workers/processes
    data = run(sims) do sim::Sim, hist::SimHistory
        return [:n_steps => n_steps(hist),
                :discounted_reward => discounted_reward(hist),
                :hist_validation_hash => validation_hash(hist),
                :median_updater_time => median(ui[:updater_time] for ui in eachstep(hist, "ui")),
                :median_prediction_time => median(ai[:prediction_time] for ai in eachstep(hist, "ai")),
                :median_planning_time => median(ai[:planning_time] for ai in eachstep(hist, "ai")),
                :final_state_type => final_state_type(problem(sim), hist),
                :free_space_estimate => free_space_estimate(mdp(problem(sim)), first(collect(s for s in eachstep(hist, "s"))))]
    end
    return data
end

function visualize(planner_model, hist, policy; filename::String="visualize_debug")
    makegif(planner_model, hist, filename=joinpath(from_base_dir("renderings"), "$filename.gif"),
            extra_initial=true, show_progress=true,
	    render_kwargs=(po=policy, sim_hist=hist, show_info=true),
	    fps=Base.convert(Int, cld(1, dt)))
end

function visualize_plan(planner_model, hist::SimHistory, policy::Policy, step::Int;
		        fps::Int=Base.convert(Int, cld(1, dt)),
			filename::String="plan")
    policy = unwrap(policy)
    beliefs = collect(eachstep(hist, "b"))
    a, info = action_info(policy, beliefs[step])
    e = beliefs[step] |> particles |> first |> external
    hp = human_pos(e)
    rp = robot_pos(e)

    plan = get_plan(policy, beliefs[step])

    frames = Frames(MIME("image/png"), fps=fps)
    for planning_step in plan
	push!(frames, render_plan(policy, planner_model, planning_step, hp, rp))
    end
    simplified_policy_name = first(split(string(typeof(policy)), "{"))
    filename = "$simplified_policy_name-$filename-$(lpad(step, 3, "0"))"
    savedir = joinpath(from_base_dir("renderings"), "$filename.gif")
    @show write(savedir, frames)
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
    visualize(viz..., filename="$(lpad(idx, 3, "0"))")
end

function debug(data; kwargs...)
    for idx in 1:nrow(data)
        debug(data, idx; kwargs...)
    end
end

function debug_with_plan(data, idx; kwargs...)
    model, hist, policy = reproduce_scenario(data[idx, :]; kwargs...)
    visualize(model, hist, policy, filename="$(lpad(idx, 3, "0"))")
    for step in 1:length(hist)
        visualize_plan(model, hist, policy, step)
    end
end
