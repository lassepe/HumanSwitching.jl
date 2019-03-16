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

struct HumanBoltzmannBState <: HumanBehaviorState
    beta::Float64
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
    vel_min::Float64 = 0.0
    vel_max::Float64 = 1.0
    vel_sigma::Float64 = 0.01
end

bstate_type(::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState from the min_max_vel range
function rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior)::HumanConstVelBState
    return HumanConstVelBState(rand(rng, Uniform(hbm.vel_min,
                                                 hbm.vel_max)))
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

struct HumanBoltzmannModel{RMT, NA, TA} <: HumanBehaviorModel
    beta_min::Float64
    beta_max::Float64
    beta_resample_sigma::Float64
    reward_model::RMT
    epsilon::Float64

    aspace::SVector{NA, TA}
    _aprob_mem::MVector{NA, Float64}
end

function HumanBoltzmannModel(;beta_min=0.0, beta_max=15.0, beta_resample_sigma=0.3,
                             reward_model=HumanSingleTargetRewardModel(), epsilon=0.02,
                             aspace=gen_human_aspace())
    if beta_min == beta_max
        @assert iszero(beta_resample_sigma)
    end
    return HumanBoltzmannModel(beta_min, beta_max, beta_resample_sigma, reward_model, epsilon, aspace,
                              @MVector(zeros(length(aspace))))
end

bstate_type(::HumanBoltzmannModel)::Type = HumanBoltzmannBState

function rand_hbs(rng::AbstractRNG, hbm::HumanBoltzmannModel)
    return HumanBoltzmannBState(hbm.beta_min == hbm.beta_max ?
                                hbm.beta_max : rand(rng, Truncated(Exponential(1),hbm.beta_min, hbm.beta_max)))
end

@with_kw struct HumanSingleTargetRewardModel
    human_target::Pose = Pose(5, 5, 0)
end

@with_kw struct HumanBoltzmannAction <: FieldVector{2, Float64}
    d::Float64 = 0 # distance
    phi::Float64 = 0 # direction
end

function gen_human_aspace(phi_step::Float64=pi/12)
    dist = 0.5
    direction_actions = [i for i in -pi:phi_step:(pi-phi_step)]
    SVector{length(direction_actions)+1, HumanBoltzmannAction}([zero(HumanBoltzmannAction),(HumanBoltzmannAction(dist, direction) for direction in direction_actions)...])
end

apply_human_action(p::Pose, a::HumanBoltzmannAction)::Pose = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

function free_evolution(hbm::HumanBoltzmannModel, hbs::HumanBoltzmannBState, p::Pose, rng::AbstractRNG)
    d = get_action_distribution(hbm, hbs, p)
    sampled_action = hbm.aspace[rand(rng, d)]
    p_p = apply_human_action(p, sampled_action)
end

function compute_qval(p::Pose, a::HumanBoltzmannAction, reward_model::HumanSingleTargetRewardModel)
    # TODO: reason about whether this should be the 2 or 1 norm!
    return -a.d - dist_to_pose(apply_human_action(p, a), reward_model.human_target; p=2)
end

function get_action_distribution(hbm::HumanBoltzmannModel, hbs::HumanBoltzmannBState, p::Pose)
    for (i, a) in enumerate(hbm.aspace)
        hbm._aprob_mem[i] = exp(hbs.beta * compute_qval(p, a, hbm.reward_model))
    end
    return Categorical(Array(normalize!(hbm._aprob_mem, 1)))
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
