"""
reward

The reward function for this problem.
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
    step_reward += rm.move_to_goal_reward * (robot_dist_to_goal(m, s) - robot_dist_to_goal(m, sp))
    # reward for reaching the goal
    if robot_reached_goal(m, sp)
        step_reward += rm.goal_reached_reward
    end

    if !isinroom(robot_pos(sp), room(m))
        step_reward += rm.left_room_penalty
    end

    # being close to humans is asymptotically bad
    if dist_to_pos(robot_pos(sp), human_pos(sp); p=2) < 2 * agent_min_distance(m)
        step_reward += rm.dist_to_human_penalty
    end

    step_reward
end
