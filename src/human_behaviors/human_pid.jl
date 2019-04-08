"""
HumanPIDBehavior
"""
@with_kw struct HumanPIDBState <: HumanBehaviorState
    target_index::Int = 1
    vel_max::Float64 = 0.4
end

target_index(hbs::HumanPIDBState) = hbs.target_index

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
    target_sequence::Array{Pos, 1}
end

HumanPIDBehavior(r::RoomRep) = HumanPIDBehavior(target_sequence=corner_positions(r))

bstate_type(::HumanBehaviorModel)::Type = HumanPIDBState

rand_hbs(rng::AbstractRNG, hbm::HumanPIDBehavior) = HumanPIDBState(target_index=1)

human_target(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = hbm.target_sequence[target_index(hbs)]
next_target_index(hbm::HumanPIDBehavior, hbs::HumanPIDBState) = min(length(hbm.target_sequence), target_index(hbs)+1)

function free_evolution(hbm::HumanPIDBehavior, hbs::HumanPIDBState, p::Pos)
    human_velocity = min(hbs.vel_max, dist_to_pos(p, human_target(hbm, hbs))) #m/s
    vec2target = vec_from_to(p, human_target(hbm, hbs))
    walk_direction = normalize(vec2target)
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

    hbs_p = (dist_to_pos(human_pos_p, human_target(hbm, hbs)) < agent_min_distance(m) ?
             hbs=HumanPIDBState(target_index=next_target_index(hbm, hbs)) : hbs)

    return human_pos_p, hbs_p
end

