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

@with_kw struct HumanBoltzmannBState{RMT, AT} <: HumanBehaviorState
  beta::Float64
  reward_model::RMT
  aspace::Array{AT}
end

function free_evolution(hbs::HumanBoltzmannBState, p::Pose, rng::AbstractRNG)::Pose
  d = get_action_distribution(hbs, p)
  sampled_action = hbs.aspace[rand(rng, d)]
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
select_submodel(hbm::HumanBehaviorModel, hbs::HumanBehaviorState)::HumanBehaviorModel = hbm

@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
  min_max_vel::Array{Float64} = [0.0, 1.0]
  vel_sigma::Float64 = 0.01
end

bstate_type(hbm::HumanConstVelBehavior)::Type = HumanConstVelBState

# this model randomely generates HumanConstVelBState from the min_max_vel range
function rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior)::HumanConstVelBState
  return HumanConstVelBState(rand(rng, Uniform(hbm.min_max_vel...)))
end

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
  potential_targets::Array{Pose}
  goal_change_likelihood::Float64 = 0.01
end

HumanPIDBehavior(room::RoomRep; kwargs...) = HumanPIDBehavior(potential_targets=corner_poses(room); kwargs...)

bstate_type(hbm::HumanBehaviorModel)::Type = HumanPIDBState

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

@with_kw struct HumanBoltzmannModel{RMT, AT} <: HumanBehaviorModel
  min_max_beta::Array{Float64} = [0, 50]
  reward_model::RMT= HumanSingleTargetRewardModel()
  aspace::Array{AT} = gen_human_aspace()
end

bstate_type(hbm::HumanBoltzmannModel)::Type = HumanBoltzmannBState

function rand_hbs(rng::AbstractRNG, hbm::HumanBoltzmannModel)::HumanBoltzmannBState
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

function gen_human_aspace()::Array{HumanBoltzmannAction, 1}
  dist_actions::Array{Float64}= [0.3, 0.6]
  direction_resolution::Float64 = pi/4
  direction_actions = (-pi:direction_resolution:(pi-direction_resolution))

  return vec([zero(HumanBoltzmannAction),
              (HumanBoltzmannAction(d, direction) for d in dist_actions, direction in direction_actions)...])
end

apply_human_action(p::Pose, a::HumanBoltzmannAction)::Pose = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

function compute_qval(p::Pose, a::HumanBoltzmannAction, reward_model::HumanSingleTargetRewardModel)::Float64
  # TODO: reason about whether this should be the 2 or 1 norm!
  return -norm(a.d) - dist_to_pose(apply_human_action(p, a), reward_model.human_target; p=2)
end

function get_action_distribution(hbs::HumanBoltzmannBState, p::Pose)::Categorical
  qvals::Array{Float64} = [compute_qval(p, a, hbs.reward_model) for a in hbs.aspace]
  action_props::Array{Float64} = normalize([exp(hbs.beta * q) for q in qvals], 1)
  return Categorical(action_props)
end

@with_kw struct HumanUniformModelMix <: HumanBehaviorModel
  submodels::Array{HumanBehaviorModel}
  bstate_change_likelihood::Float64
end

bstate_type(hbm::HumanUniformModelMix)::Type = Union{Iterators.flatten([[bstate_type(sm)] for sm in hbm.submodels])...}

function select_submodel(hbm::HumanUniformModelMix, hbs::HumanBehaviorState)::HumanBehaviorModel
  candidate_submodels = filter(x->(hbs isa bstate_type(x)), hbm.submodels)
  @assert(length(candidate_submodels) == 1)
  return first(candidate_submodels)
end

rand_hbs(rng::AbstractRNG, hbm::HumanUniformModelMix)::HumanBehaviorState = rand_hbs(rng::AbstractRNG, rand(rng, hbm.submodels))
