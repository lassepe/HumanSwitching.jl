"""
HumanPIDBehavior
"""
@with_kw struct HumanPIDBState <: HumanBehaviorState
    goal_index::Int = 1
    vel_max::Float64 = 0.4
end

goal_index(hbs::HumanPIDBState) = hbs.goal_index

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
    goal_sequence::Array{Pos, 1}
end

HumanPIDBehavior(r::Room) = HumanPIDBehavior(goal_sequence=corner_positions(r))

bstate_type(::HumanBehaviorModel)::Type = HumanPIDBState

rand_hbs(rng::AbstractRNG, hbm::HumanPIDBehavior) = HumanPIDBState(goal_index=1)

human_goal(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = hbm.goal_sequence[goal_index(hbs)]
next_goal_index(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = min(length(hbm.goal_sequence), goal_index(hbs)+1)

function free_evolution(hbm::HumanPIDBehavior, hbs::HumanPIDBState, p::Pos)
    human_velocity = min(hbs.vel_max, dist_to_pos(p, human_goal(hbm, hbs))) #m/s
    vec2goal = vec_from_to(p, human_goal(hbm, hbs))
    walk_direction = normalize(vec2goal)
    # new position:
    human_pos_p::Pos = p
    if !any(isnan(i) for i in walk_direction)
        xy_p = p[1:2] + walk_direction * human_velocity
        human_pos_p = xy_p
    end

    return human_pos_p
end

function human_transition(hbs::HumanPIDBState, hbm::HumanPIDBehavior, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbm, hbs, p)

    hbs_p = (dist_to_pos(human_pos_p, human_goal(hbm, hbs)) < goal_reached_distance(m) ?
             hbs=HumanPIDBState(goal_index=next_goal_index(hbm, hbs)) : hbs)

    return human_pos_p, hbs_p
end

