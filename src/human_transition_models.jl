"""
Defnining some human transition models. (dynamics according to which human
move)
"""

function human_transition(hbs::HumanConstVelBState, hbm::HumanConstVelBehavior, m::HSModel,
                          p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanConstVelBState}
  human_pose_p = free_evolution(hbs, p)
  hbs_p = HumanConstVelBState(rand(rng, Normal(hbs.velocity, hbm.vel_sigma)))
  return human_pose_p, hbs_p
end

function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel,
                          p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanPIDBState}
  human_pose_p = free_evolution(hbs, p)

  hbs_p = (dist_to_pose(human_pose_p, human_target(hbs)) < agent_min_distance(m)
           || rand(rng) < hbm.goal_change_likelihood ?
           rand_hbs(rng, hbm) : hbs)

  return human_pose_p, hbs_p
end

# TODO: all of this could be typed more strongly to improve type stability. Avoid jusing abstract classes!
function human_transition(hbs::HumanBehaviorState, hbm::HumanUniformModelMix, m::HSModel,
                          p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanBehaviorState}
  # select the corresponding sub model
  hbm_sub = select_submodel(hbm, hbs)
  # propagate state according to this model
  human_pose_p, hbs_sub_prime = human_transition(hbs, hbm_sub, m, p, rng)
  # small likelihood of randomly changing behavior state
  hbs_p = rand(rng) < hbm.bstate_change_likelihood ? rand_hbs(rng, hbm) : hbs

  return human_pose_p, hbs_p
end

function human_transition(hbs::HumanBoltzmannBState, hbm::HumanBoltzmannModel, m::HSModel,
                          p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanBoltzmannBState}
  d = get_action_distribution(hbm, hbs, p)
  sampled_action = hbm.aspace[rand(rng, d)]
  # TODO: also rand beta
  return apply_human_action(p, sampled_action), hbs
end
