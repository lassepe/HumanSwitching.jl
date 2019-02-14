@with_kw struct HSRewardModel
  discount_factor::Float64 = 0.8
  living_penalty::Float64 = -2
  control_cost::Float64 = -0.1
  collision_penalty::Float64 = -100.0
  move_to_goal_reward::Float64 = 0.1
  target_reached_reward::Float64 = 100.0
  left_room_penalty::Float64 = -100.0
end

"""
reward

The reward function for this problem.

NOTE: Nothing intereseting here until the agent is also moving
"""
function POMDPs.reward(m::HSModel, s::HSState, a::HSAction, sp::HSState)::Float64
  rm = reward_model(m)
  step_reward::Float64 = 0

  # encourage finishing in finite time
  step_reward += rm.living_penalty
  # control_cost
  step_reward += rm.control_cost * a.d
  # avoid collision
  if has_collision(m, sp)
    step_reward += rm.collision_penalty
  end
  # make rewards less sparse by rewarding going towards the goal
  step_reward += rm.move_to_goal_reward * (robot_dist_to_target(s) - robot_dist_to_target(sp))
  # reward for reaching the goal
  if robot_reached_target(sp)
    step_reward += rm.target_reached_reward
  end

  if !isinroom(robot_pose(sp), room(m))
    step_reward += rm.left_room_penalty
  end

  step_reward
end
