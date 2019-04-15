function generate_from_template(template_model::HSModel, rng::AbstractRNG;
                                sensor::HSSensor, human_behavior_model::HumanBehaviorModel, physical_transition_noise_model::HSPhysicalTransitionNoiseModel)

    # copy all fields but modify human_behavior_model and physical_transition_noise_model
    return HSPOMDP(sensor, construct_with(mdp(template_model),
                                          :human_behavior_model=>human_behavior_model,
                                          :physical_transition_noise_model=>physical_transition_noise_model, type_hint=HSMDP))
end

function generate_non_trivial_scenario(sensor::HSSensor, human_behavior_model::HumanBehaviorModel,
                                       physical_transition_noise_model::HSPhysicalTransitionNoiseModel, rng::AbstractRNG; kwargs...)
    if get(kwargs, :known_external_initstate, nothing) !== nothing
        @error "Non-trivial scenarios can't be generated from fixed external init states."
    elseif get(kwargs, :robot_goal, nothing) !== nothing
        @error "Non-trivial scenarios can't be generated from fixed robot goals."
    end

    simulator_rng = deepcopy(rng)

    while true
        # sample a new, partially observable setup
        po_model = HSPOMDP(sensor, gen_hsmdp(rng, human_behavior_model=human_behavior_model, physical_transition_noise_model=physical_transition_noise_model; kwargs...))

        # check if the trivial policy (go straight to goal, ignoring human) works well on the full
        # observable problem
        fo_model = mdp(po_model)
        trivial_policy = StraightToGoal(fo_model)

        simulator = HistoryRecorder(max_steps=100, show_progress=false, rng=deepcopy(simulator_rng))
        sim_hist = simulate(simulator, fo_model, trivial_policy, initialstate(fo_model, deepcopy(simulator_rng)))

        state_history = collect(eachstep(sim_hist, "sp"))
        if length(state_history) == 0 || dist_to_wall(robot_goal(fo_model), room(fo_model)) < 2*agent_min_distance(fo_model)
            continue
        end
        last_state = last(state_history)

        if has_collision(fo_model, last_state)
            return po_model
        end
    end
end
