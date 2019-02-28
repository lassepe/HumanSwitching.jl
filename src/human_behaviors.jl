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

struct HumanBoltzmannBState
  beta::Float64
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

@with_kw struct HumanBoltzmannModel
  # TODO: It might be a good idea to initialize with a rather low beta?
  min_max_beta::Array{Float64} = [0, 100]
  human_action_space::Array{Pose} = gen_human_aspace()
end

function gen_human_aspace()::Array{Pose, 1}
  # parameters, TODO: move!
  dphi_max::Float64 = pi/4
  n_phi_steps::Int = 3
  dphi_actions::Array{Float64} = range(-dphi_max, stop=dphi_max, length=n_phi_steps)

  # parameters, TODO: move!
  dist_actions::Array{Float64} = [0.6]
  direction_actions::Array{Float64} = [-pi/4, 0.0, pi/4]
  dxy_actions = vec([[0,0], ([dist, direction] for dist in dist_actions, direction in direction_actions)...])

  return vec([Pose(dxy..., dphi) for dxy in dxy_actions, dphi in dphi_actions])
end
bstate_type(hbm::HumanBoltzmannModel)::Type = HumanBoltzmannBState

function rand_hbs(rng::AbstractRNG, hbm::HumanBoltzmannModel)::HumanBoltzmannBState
  return HumanBoltzmannBState(rand(rng, Uniform(min_max_beta...)))
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

function rand_hbs(rng::AbstractRNG, hbm::HumanUniformModelMix)::HumanBehaviorState
  return rand_hbs(rng::AbstractRNG, rand(rng, hbm.submodels))
end
