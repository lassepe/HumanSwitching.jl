"""
# HumanBehaviorModel
"""

# basic models don't have further submodels
select_submodel(hbm::HumanBehaviorModel, hbs_type::Type{<:HumanBehaviorState}) = hbm
select_submodel(hbm::HumanBehaviorModel, hbs::HumanBehaviorState)::HumanBehaviorModel = select_submodel(hbm, typeof(hbs))

"""
    rand_hbs(rng::AbstractRNG, hbm::HumanBehaviorModel)

Generates a random human behavior state for the model based on some internal
prior.
"""
function rand_hbs end
"""
    speed_max(hbm::HumanBehaviorModel)

Returns the maximum speed the human can go according to this model. Here we
provide a sane default.
"""
speed_max(hbm::HumanBehaviorModel) = hbm.speed_max
"""
    human_transition(hbs::HumanBehaviorState, hbm::HumanBehaviorModel, m::HSModel, p::Pos, rng::AbstractRNG)

Implements the human transition. That is it returns a tuple over the next human
position and the next human behavior state (human_pos_p, hbs_p) given the last
position, the last behavior state and the mdoel.
"""
function human_transition end

"""
# Human Behavior utils
"""
@with_kw struct HumanAction <: FieldVector{2, Float64}
    d::Float64 = 0 # distance
    phi::Float64 = 0 # direction
end

function gen_human_aspace(;dist::Float64=0.2, phi_step::Float64=pi/8)
    direction_actions = [i for i in -pi:phi_step:(pi-phi_step)]
    SVector{length(direction_actions)+1, HumanAction}([zero(HumanAction),(HumanAction(dist, direction) for direction in direction_actions)...])
end

apply_human_action(p::Pos, a::HumanAction)::Pos = Pos(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d)

function uniform_goal_generator(goals::Array{Pos, 1}, rng::AbstractRNG)
    return rand(rng, goals)::Pos
end

function uniform_goal_generator(current_goal::Pos, goals::Array{Pos, 1}, rng::AbstractRNG)
    return rand(rng) < 0.0 ? current_goal : uniform_goal_generator(goals, rng)::Pos
end
