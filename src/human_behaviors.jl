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

"""
# HumanBehaviorModel

Each describe
  - from which distribution HumanBehaviorState's are sampled
  - how HumanBehaviorState's evolve (see `human_transition_models.jl`)
"""

@with_kw struct HumanConstVelBehavior <: HumanBehaviorModel
  min_max_vel::Array{Float64} = [0.0, 1.0]
end

# this model randomely generates HumanConstVelBState from the min_max_vel range
function rand_hbs(rng::AbstractRNG, hbm::HumanConstVelBehavior)::HumanConstVelBState
  return HumanConstVelBState(rand(rng, Uniform(hbm.min_max_vel...)))
end

@with_kw struct HumanPIDBehavior <: HumanBehaviorModel
  potential_targets::Array{Pose}
  goal_change_likelyhood::Float64 = 0.01
end

function rand_hbs(rng::AbstractRNG, hbm::HumanPIDBehavior)::HumanPIDBState
  return HumanPIDBState(human_target=rand(rng, hbm.potential_targets))
end

target_index(hbm::HumanPIDBehavior, p::Pose) = findfirst(x->x==p, vec(hbm.potential_targets))
