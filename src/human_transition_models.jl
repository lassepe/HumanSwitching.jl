"""
Defnining some human transition models. (dynamics according to which human
move)
"""

# default: invariant hbs and free evolution of the external state according to the hbs
function human_transition(hbs::HumanPIDBState, hbm::HumanBehaviorModel, m::HSModel, p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanBehaviorState}
  @warn "Using Fallback" maxlog=1
  return free_evolution(hbs, p), hbs
end


function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel, p::Pose, rng::AbstractRNG)::Tuple{Pose, HumanPIDBState}
  human_pose_p = free_evolution(hbs, p)

  hbs_p = (dist_to_pose(human_pose_p, human_target(hbs)) < agent_min_distance(m) || rand(rng) < hbm.goal_change_likelyhood ?
           rand_hbs(rng, human_behavior_model(m)) : hbs)

  return human_pose_p, hbs_p
end
#
# function human_transition(hbs::HumanConstVelBState, p::Pose, m::HSModel, rng::AbstractRNG)::Tuple{Pose, HumanConstVelBState}
# end
