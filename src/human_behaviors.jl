"""
# state representations of the human internal state
"""
@with_kw struct HumanConstVelBState <: HumanBehaviorState
    velocity::Float64
end

function free_evolution(hbs::HumanConstVelBState, p::Pose)::Pose
    dp = Pose(cos(p.phi), sin(p.phi), 0) * hbs.velocity
    return Pose(p + dp)
end

@with_kw struct HumanPIDBState <: HumanBehaviorState
    human_target::Pose
    max_speed::Float64 = 0.5
end

human_target(hm::HumanPIDBState) = hm.human_target

function free_evolution(hbs::HumanPIDBState, p::Pose)::Pose
    human_velocity = min(hbs.max_speed, dist_to_pose(p, human_target(hbs))) #m/s
    vec2target = vec_from_to(p, human_target(hbs))
    target_direction = normalize(vec2target)
    current_walk_direction = @SVector [cos(p.phi), sin(p.phi)]
    walk_direction = (target_direction + current_walk_direction)/2
    # new position:
    human_pose_p::Pose = p
    if !any(isnan(i) for i in target_direction)
        xy_p = p[1:2] + walk_direction * human_velocity
        phi_p = atan(walk_direction[2], walk_direction[1])
        human_pose_p = [xy_p..., phi_p]
    end

    return human_pose_p
end

@with_kw struct HumanBoltzmannBState{RMT, NA, TA} <: HumanBehaviorState
    beta::Float64
    reward_model::RMT
    aspace::SVector{NA, TA}
end

function free_evolution(hbs::HumanBoltzmannBState, p::Pose, rng::AbstractRNG)::Pose
    d = get_action_distribution(hbs, p)
    sampled_action = rand(rng, d)
    # TODO: also rand beta
    return apply_human_action(p, sampled_action)
end

"""
# HumanBehaviorModel

Each describe
- from which distribution HumanBehaviorState's are sampled
- how HumanBehaviorState's evolve (see `human_transition_models.jl`)
"""
# basic models don't have further submodels
select_submodel(hbm::HumanBehaviorModel, hbs::Type{<:HumanBehaviorState}) = hbm
select_submodel(hbm::HumanBehaviorModel, hbs::HumanBehaviorState)::HumanBehaviorModel = select_submodel(hbm, typeof(hbs))

@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
    min_max_vel::Array{Float64, 1} = [0.0, 1.0]
    vel_sigma::Float64 = 0.01
end

bstate_type(::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState from the min_max_vel range
function rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior)::HumanConstVelBState
    return HumanConstVelBState(rand(rng, Uniform(hbm.min_max_vel...)))
end

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
    potential_targets::Array{Pose, 1}
    goal_change_likelihood::Float64 = 0.01
end

HumanPIDBehavior(room::RoomRep; kwargs...) = HumanPIDBehavior(potential_targets=corner_poses(room); kwargs...)

bstate_type(::HumanBehaviorModel)::Type = HumanPIDBState

function rand_hbs(rng::AbstractRNG, hbm::HumanPIDBehavior)::HumanPIDBState
    return HumanPIDBState(human_target=rand(rng, hbm.potential_targets))
end

function target_index(hbm::HumanPIDBehavior, p::Pose)
    idx = findfirst(x->x==p, vec(hbm.potential_targets))
    if idx === nothing
        @warn "Lookup of unknown target!" maxlog=1
    end
    return idx
end

abstract type HumanRewardModel end

@with_kw struct HumanBoltzmannModel{RMT, NA, TA} <: HumanBehaviorModel
    min_max_beta::Array{Float64, 1} = [0, 10]
    beta_rasample_sigma::Float64 = 1.0
    reward_model::RMT= HumanSingleTargetRewardModel()
    aspace::SVector{NA, TA} = gen_human_aspace()
end

bstate_type(::HumanBoltzmannModel)::Type = HumanBoltzmannBState

function rand_hbs(rng::AbstractRNG, hbm::HumanBoltzmannModel)
    # TODO: Reward model parameters should be random as well, if one want's to estimate them
    return HumanBoltzmannBState(beta=rand(rng, Uniform(hbm.min_max_beta...)),
                                reward_model=hbm.reward_model,
                                aspace=hbm.aspace)
end

@with_kw struct HumanSingleTargetRewardModel
    human_target::Pose = Pose(7.5, 7.5, 0)
end

@with_kw struct HumanBoltzmannAction <: FieldVector{2, Float64}
    d::Float64 = 0 # distance
    phi::Float64 = 0 # direction
end

function gen_human_aspace()
    dist_actions = @SVector[0.3, 0.6]
    direction_actions = @SVector[i for i in -pi:pi/4:(pi-pi/4)]
    SVector{length(dist_actions)*length(direction_actions)+1, HumanBoltzmannAction}([zero(HumanBoltzmannAction),(HumanBoltzmannAction(d, direction) for d in dist_actions, direction in direction_actions)...])
end

apply_human_action(p::Pose, a::HumanBoltzmannAction)::Pose = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

function compute_qval(p::Pose, a::HumanBoltzmannAction, reward_model::HumanSingleTargetRewardModel)::Float64
    # TODO: reason about whether this should be the 2 or 1 norm!
    return -norm(a.d) - dist_to_pose(apply_human_action(p, a), reward_model.human_target; p=2)
end

function get_action_distribution(hbs::HumanBoltzmannBState, p::Pose)
    qvals = (compute_qval(p, a, hbs.reward_model) for a in hbs.aspace)
    action_props = (exp(hbs.beta * q) for q in qvals)
    return SparseCat(hbs.aspace, action_props)
end

struct HumanUniformModelMix{T} <: HumanBehaviorModel
    submodels::Array{T, 1}
    bstate_change_likelihood::Float64
    bstate_type::Type
end

function HumanUniformModelMix(models...; bstate_change_likelihood::Float64)
    submodels = [models...]
    return HumanUniformModelMix{Union{typeof.(models)...}}(submodels,
                                                           bstate_change_likelihood,
                                                           Union{Iterators.flatten([[bstate_type(sm)] for sm in submodels])...})
end

bstate_type(hbm::HumanUniformModelMix) = hbm.bstate_types

function select_submodel(hbm::HumanUniformModelMix{T}, t::Type{<:HumanBehaviorState})::T where T
    candidate_submodels = filter(x->(t <: bstate_type(x)), hbm.submodels)
    @assert(length(candidate_submodels) == 1)
    return first(candidate_submodels)
end

rand_hbs(rng::AbstractRNG, hbm::HumanUniformModelMix)::HumanBehaviorState = rand_hbs(rng::AbstractRNG, rand(rng, hbm.submodels))
