"""
PControlledHumanTransition

A human with constant velocity, only controlling it's orientation
"""
struct PControlledHumanTransition <: HSTransitionModel end

# human controlled by simple P-controller
function human_p_transition(s::HSState)::Tuple{Pose, Pose}
  human_velocity = min(0.3, human_dist_to_target(s)) #m/s
  vec2target = human_vec_to_target(s)
  target_direction = normalize(vec2target)
  current_walk_direction = @SVector [cos(human_pose(s).phi), sin(human_pose(s).phi)]
  walk_direction = (target_direction + current_walk_direction)/2
  # new position:
  human_pose_p::Pose = human_pose(s)
  if !any(isnan(i) for i in target_direction)
    xy_p = human_pose(s)[1:2] + walk_direction * human_velocity
    phi_p = atan(walk_direction[2], walk_direction[1])
    human_pose_p = [xy_p..., phi_p]
  end

  return human_pose_p, human_target(s)
end

# helper funciton to access the deterministic P controlled human transition
function POMDPs.generate_s(m::HSMDP{PControlledHumanTransition, <:Any}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # assembling the new state
  human_pose_p, human_target_p = human_p_transition(s)
  if human_reached_target(s)
    human_target_p = rand(rng, corner_poses(room(m)))
  end

  # a deterministic robot transition model
  robot_pose_p = apply_action(robot_pose(s), a)
  robot_target_p = robot_target(s)

  HSState(human_pose_p, human_target_p, robot_pose_p, robot_target_p)
end

"""
PControlledHumanAWGNTransition

Same as above but with added white gaussian noise.
"""
@with_kw struct PControlledHumanAWGNTransition <: HSTransitionModel
  pose_cov::Array{Float64, 1} = [0.15, 0.15, 0.01] # the diagonal of the transition noise covariance matrix
end

function POMDPs.generate_s(m::HSMDP{PControlledHumanAWGNTransition, <:Any}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # first get the deterministic version
  human_pose_p::Pose, human_target_p::Pose = human_p_transition(s)
  # add AWGN to the pose and have small likelyhood of chaning the target
  do_resample = rand(rng) > 0.99 || human_reached_target(s)
  human_target_p = do_resample ? rand(rng, corner_poses(room(m))) : human_target_p
  # a deterministic robot transition model

  # TODO: Just proof of concept! MOVE to a proper place!
  # TODO: The robot should have it's own transition statistics
  #
  human_pose_p += rand(rng, MvNormal([0, 0, 0], transition_model(m).pose_cov))
  transition_noise = rand(rng, MvNormal([0, 0, 0], transition_model(m).pose_cov))
  robot_pose_p::Pose = apply_action(robot_pose(s), a) + [transition_noise[1:2]..., 0]
  robot_target_p::Pose = robot_target(s)

  HSState(human_pose_p, human_target_p,
          robot_pose_p, robot_target_p)
end

