"""
# Utility Types
"""
# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 15
  height::Float64 = 15
end

# the physical representation of an agent
@with_kw struct Pose <: FieldVector{3, Float64}
  x::Float64 = 0   # horizontal position
  y::Float64 = 0   # vertical position
  phi::Float64 = 0 # the orientation of the human
end

"""
# Sensor Models
For each type an observation model is defined

Details: `generate_o` and `observation`
"""
abstract type HSSensor end
struct ExactPositionSensor <: HSSensor end

@with_kw struct NoisyPositionSensor <: HSSensor
  # the diagonal of the measurement covariance matrix
  measurement_cov::Array{Float64, 1} = [0.1,
                                        0.1,
                                        0.01]
end

"""
# Robot reward model
Describes the rewards the robot cares about
"""

@with_kw struct HSRewardModel
  discount_factor::Float64 = 0.97
  living_penalty::Float64 = -1
  collision_penalty::Float64 = -50
  left_room_penalty::Float64 = -50
  target_reached_reward::Float64 = 40.0
  dist_to_human_penalty::Float64 = -5
  move_to_goal_reward::Float64 = 0
  control_cost::Float64 = 0
end

"""
# Post Transition Transformations
For each type a post-processing step for the particle filter is defined (e.g adding noise)

Details: `physical_transition_noise_model.jl`
"""
abstract type HSPhysicalTransitionNoiseModel end

struct HSIdentityPTNM <: HSPhysicalTransitionNoiseModel end

@with_kw struct HSGaussianNoisePTNM <: HSPhysicalTransitionNoiseModel
  pose_cov::Array{Float64} = [0.15, 0.15, 0.01] # the diagonal of the transition noise covariance matrix
end

"""
# Human Behavior Models
For each behavior a `human_transition` is defined

Details:
- behavior tree see `human_behavior_tree.jl`
- transition models see `human_transition_models.jl`
"""
abstract type HumanBehaviorState end
abstract type HumanBehaviorModel end
function rand_hbs end

"""
# State Representation
"""
@with_kw struct HSExternalState
  human_pose::Pose
  robot_pose::Pose
end

HSExternalState(v::AbstractVector{Float64}) = HSExternalState(v[1:3], v[4:6])
convert(::Type{V}, o::HSExternalState) where V <: AbstractVector = V([human_pose(o)..., robot_pose(o)...])

@with_kw struct HSState
  external::HSExternalState
  hbs::HumanBehaviorState
end

external(s::HSState) = s.external
external(s::HSExternalState) = s
hbs(s::HSState) = s.hbs
hbs(m::HumanBehaviorState) = m
internal(s::HSState) = hbs(s::HSState)
compose_state(e::HSExternalState, i::HumanBehaviorState) = HSState(external=e, hbs=i)

human_pose(s::Union{HSState, HSExternalState}) = external(s).human_pose
robot_pose(s::Union{HSState, HSExternalState}) = external(s).robot_pose

"""
# Action (Space) representation
"""
@with_kw struct HSAction <: FieldVector{2, Float64}
  d::Float64 = 0   # distance to travel
  phi::Float64 = 0 # angle of direction in which the distance is traveled
end

# defining the default action space
function HSActionSpace()
  dist_actions = (0.3)
  phi_resolution = (pi/2)
  phi_actions = (-pi:phi_resolution:(pi-phi_resolution))

  return vec([zero(HSAction()), (HSAction(d, phi) for d in dist_actions, phi in phi_actions)...])
end

apply_action(p::Pose, a::HSAction) = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

"""
# POMDP and MDP Representation
- implementing the POMDPs.jl interface
"""
@with_kw struct HSMDP{AS} <: MDP{HSState, HSAction}
  room::RoomRep = RoomRep()
  aspace::AS = HSActionSpace()
  reward_model::HSRewardModel = HSRewardModel()
  human_behavior_model::HumanBehaviorModel = HumanPIDBehavior(room)
  physical_transition_noise_model::HSPhysicalTransitionNoiseModel = HSIdentityPTNM()
  robot_target::Pose = rand_pose(room, Random.GLOBAL_RNG, forced_orientation=0.0)
  agent_min_distance::Float64 = 1.0
  known_external_initstate::Union{HSExternalState, Nothing} = nothing
end

@with_kw struct HSPOMDP{TS, O, M} <: POMDP{HSState, HSAction, O}
  sensor::TS = ExactPositionSensor()
  mdp::M = HSMDP()
end

function HSPOMDP(sensor::HSSensor, mdp::HSMDP)
  HSPOMDP{typeof(sensor), HSExternalState, typeof(mdp)}(sensor, mdp)
end

const HSModel = Union{HSMDP, HSPOMDP}

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
human_behavior_model(m::HSModel) = mdp(m).human_behavior_model
robot_target(m::HSModel) = mdp(m).robot_target
reward_model(m::HSModel) = mdp(m).reward_model
physical_transition_noise_model(m::HSModel) = mdp(m).physical_transition_noise_model
room(m::HSModel) = mdp(m).room
agent_min_distance(m::HSModel) = mdp(m).agent_min_distance

"""
# Implementation of main POMDP Interface
"""
POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)
POMDPs.discount(m::HSModel) = reward_model(m).discount_factor

# this simple forwards to the different transition models
function POMDPs.generate_s(m::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  @assert (a in actions(m))

  human_pose_intent, hbs_p = human_transition(hbs(s), human_behavior_model(m), m, human_pose(s), rng)
  robot_pose_intent = apply_action(robot_pose(s), a)
  external_intent::HSExternalState = HSExternalState(human_pose_intent, robot_pose_intent)
  # the intended transition is augmented with the physical transition noise
  external_p = apply_physical_transition_noise(physical_transition_noise_model(m), external_intent, rng)

  sp::HSState = HSState(external=external_p,
                        hbs=hbs_p)
end

# In this version the observation is a **deterministic** extraction of the observable part of the state
POMDPs.generate_o(m::HSPOMDP{ExactPositionSensor, HSExternalState, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSExternalState = external(sp)
POMDPs.generate_o(m::HSPOMDP{NoisyPositionSensor, HSExternalState, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSExternalState = HSExternalState(rand(rng, POMDPs.observation(m, sp)))

# TODO: This is a bit ugly. There should be away to directly define a distribution type on a FieldVector
# at least one should get away with less type conversion
function POMDPs.observation(m::HSPOMDP{NoisyPositionSensor, HSExternalState, <:Any}, s::HSState)
  # TODO: do this properly
  return MvNormal(Array{Float64, 1}([human_pose(s)..., robot_pose(s)...]),
                  Array{Float64, 1}([m.sensor.measurement_cov..., m.sensor.measurement_cov...]))
end
Distributions.pdf(distribution::MvNormal, sample::HSExternalState) = pdf(distribution, convert(Vector{Float64}, sample))

function POMDPs.isterminal(m::HSModel, s::HSState)
  robot_reached_target(m, s) || !isinroom(robot_pose(s), room(m)) || has_collision(m, s)
end

function POMDPs.initialstate(m::HSModel, rng::AbstractRNG)::HSState
  return rand_state(m, rng; known_external_state=mdp(m).known_external_initstate)
end

struct HSInitialDistribution{ModelType<:HSModel}
  m::ModelType
end

POMDPs.rand(rng::AbstractRNG, d::HSInitialDistribution) = initialstate(d.m, rng)
POMDPs.initialstate_distribution(m::HSModel) = HSInitialDistribution(m)
