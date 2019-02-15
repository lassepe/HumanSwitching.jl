function generate_hspomdp(sensor::HSSensor, post_transition_transform::HSPostTransitionTransform, rng::AbstractRNG;
                          room::RoomRep=RoomRep(),
                          aspace=HSActionSpace(),
                          reward_model::HSRewardModel=HSRewardModel(),
                          agent_min_distance::Float64=1.0,
                          known_external_initstate::Union{HSExternalState, Nothing}=nothing)

  generate_own_init_state = (known_external_initstate === nothing)

  # if no explicit fixed stated for this problem was provided, we generate it
  if generate_own_init_state
    known_external_initstate = external(rand_state(room, rng))
  end

  mdp = HSMDP(;room=room,
              post_transition_transform=post_transition_transform,
              aspace=aspace,
              reward_model=reward_model,
              agent_min_distance=agent_min_distance,
              known_external_initstate=known_external_initstate)

  # if we generated our own init state then we also return it for external use
  # (e.g. to setup an equivalent problem for the solver and simulator)
  return generate_own_init_state ? (HSPOMDP(sensor, mdp), known_external_initstate) : HSPOMDP(sensor, mdp)
end

function generate_non_trivial_scenario(sensor::HSSensor, post_transition_transform::HSPostTransitionTransform, rng::AbstractRNG; kwargs...)
  trivial_policy = FunctionPolicy(s->reduce((a1, a2) ->
                                            dist_to_pose(apply_action(robot_pose(s), a1), robot_target(s))
                                            < dist_to_pose(apply_action(robot_pose(s), a2), robot_target(s)) ?
                                            a1 : a2,
                                            HSActionSpace()))

  if get(kwargs, :known_external_initstate, nothing) !== nothing
    @error "Non-trivial scenarios can't be generated from fixed external init states."
  end

  simulator_rng = deepcopy(rng)

  while true
    # sample a new, partially observable setup
    po_model, external_init_state = generate_hspomdp(sensor, post_transition_transform, rng; kwargs...)
    # check if the trivial policy (go straight to goal, ignoring human) works well on the full
    # observable problem
    fo_model = mdp(po_model)

    simulator = HistoryRecorder(max_steps=100, show_progress=false, rng=deepcopy(simulator_rng))
    sim_hist = simulate(simulator, fo_model, trivial_policy)

    state_history = collect(eachstep(sim_hist, "sp"))
    if length(state_history) == 0
      continue
    end
    last_state = last(state_history)

    if has_collision(fo_model, last_state)
      return po_model, external_init_state
    else
      println("Was Trivial - Sampling Again!")
    end
  end

end
