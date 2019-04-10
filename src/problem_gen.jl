function generate_hspomdp(sensor::HSSensor, human_behavior_model::HumanBehaviorModel,
                          physical_transition_noise_model::HSPhysicalTransitionNoiseModel, rng::AbstractRNG;
                          room::Room=Room(),
                          aspace=HSActionSpace(),
                          reward_model::HSRewardModel=HSRewardModel(),
                          robot_goal::Pos=rand_pos(room, rng),
                          agent_min_distance::Float64=0.4,
                          goal_reached_distance::Float64=0.2,
                          known_external_initstate::HSExternalState=external(rand_external_state(room, rng)))

    mdp = HSMDP(;room=room,
                physical_transition_noise_model=physical_transition_noise_model,
                aspace=aspace,
                reward_model=reward_model,
                human_behavior_model=human_behavior_model,
                robot_goal=robot_goal,
                agent_min_distance=agent_min_distance,
                goal_reached_distance=goal_reached_distance,
                known_external_initstate=known_external_initstate)

    # if we generated our own init state then we also return it for external use
    # (e.g. to setup an equivalent problem for the solver and simulator)
    return HSPOMDP(sensor, mdp)
end

function generate_hspomdp(sensor::HSSensor, human_behavior_model::HumanBehaviorModel,
                          physical_transition_noise_model::HSPhysicalTransitionNoiseModel, template_model::HSModel, rng::AbstractRNG)
    return generate_hspomdp(sensor, human_behavior_model, physical_transition_noise_model, rng;
                            room=room(template_model),
                            aspace=mdp(template_model).aspace,
                            reward_model=reward_model(template_model),
                            robot_goal=robot_goal(template_model),
                            agent_min_distance=agent_min_distance(template_model),
                            goal_reached_distance=goal_reached_distance(template_model),
                            known_external_initstate=mdp(template_model).known_external_initstate)
end

function generate_non_trivial_scenario(sensor::HSSensor, human_behavior_model::HumanBehaviorModel,
                                       physical_transition_noise_model::HSPhysicalTransitionNoiseModel, rng::AbstractRNG;
                                       kwargs...)
    if get(kwargs, :known_external_initstate, nothing) !== nothing
        @error "Non-trivial scenarios can't be generated from fixed external init states."
    elseif get(kwargs, :robot_goal, nothing) !== nothing
        @error "Non-trivial scenarios can't be generated from fixed robot goals."
    end

    simulator_rng = deepcopy(rng)

    while true
        # sample a new, partially observable setup
        po_model = generate_hspomdp(sensor, human_behavior_model, physical_transition_noise_model, rng; kwargs...)
        # check if the trivial policy (go straight to goal, ignoring human) works well on the full
        # observable problem
        fo_model = mdp(po_model)

        trivial_policy = FunctionPolicy(s->reduce((a1, a2) ->
                                                  dist_to_pos(apply_robot_action(robot_pos(s), a1), robot_goal(fo_model), p=2)
                                                  < dist_to_pos(apply_robot_action(robot_pos(s), a2), robot_goal(fo_model), p=2) ?
                                                  a1 : a2,
                                                  HSActionSpace()))


        simulator = HistoryRecorder(max_steps=100, show_progress=false, rng=deepcopy(simulator_rng))
        sim_hist = simulate(simulator, fo_model, trivial_policy)

        state_history = collect(eachstep(sim_hist, "sp"))
        if length(state_history) == 0
            continue
        end
        last_state = last(state_history)

        if has_collision(fo_model, last_state)
            return po_model
        end
    end
end
