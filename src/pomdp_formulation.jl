# the physical representation of a room
@with_kw struct RoomRep
  width::Float64 = 15
  height::Float64 = 15
end

corner_poses(r::RoomRep) = [Pose(x, y, 0) for x in [0.1r.width, 0.9r.width], y in [0.1r.height, 0.9r.height]]

# the physical representation of an agent
@with_kw struct Pose <: FieldVector{3, Float64}
  x::Float64 = 0   # horizontal position
  y::Float64 = 0   # vertical position
  phi::Float64 = 0 # the orientation of the human
end

# defining some sensor/observation models
abstract type HSSensor end
"""
A sensor that gives the exact position (simply extracting the corresponding
portion from the state representation)
"""
struct ExactPositionSensor <: HSSensor end
"""
A sensor that gives a noisy reading to the agents positions
"""
@with_kw struct NoisyPositionSensor <: HSSensor
  # the diagonal of the measurement covariance matrix
  measurement_cov::Array{Float64, 1} = [0.1,
                                        0.1,
                                        0.01]
end
POMDPs.obstype(::ExactPositionSensor) = HSObservation
POMDPs.obstype(::NoisyPositionSensor) = HSObservation

# defining some transition models
abstract type HSTransitionModel end
"""
PIDHumanTransition

a deterministic human transition model where the human follows a simiple
proportional controller towards a fixed target
"""
struct PControlledHumanTransition <: HSTransitionModel end
@with_kw struct PControlledHumanAWGNTransition <: HSTransitionModel
  " the diagonal of the transition AWGN covariance matrix"
  pose_cov::Array{Float64, 1} = [0.15, 0.15, 0.01]
end

"""
The state representation of the whole HumanSwitching POMDP

NOTE:
- For now this features only the agent state of a the human (composition and
orientation) as well as its target.
- For now the human is assumed to walk in a straight line towards the unobserved goal
"""
@with_kw struct HSState
  human_pose::Pose
  human_target::Pose
  robot_pose::Pose
  robot_target::Pose
end

function Base.isequal(a::HSState, b::HSState)
  isequal(a.human_pose, b.human_pose) && isequal(a.human_target, b.human_target) &&
  isequal(a.robot_pose, b.robot_pose) && isequal(a.robot_target, b.robot_target)
end

"""
The action representation in the HumanSwitching POMDP

NOTE:
- for now this is not too interesting as the agent can not take any actions
"""

# TODO: robotActions - for now the quad agent only moves in x and y. (never
# changes orientation)
@with_kw struct HSAction <: FieldVector{2, Float64}
  d::Float64 = 0   # distance to travel
  phi::Float64 = 0 # angle of direction in which the distance is traveled
end

"""
The action space for this problem
"""
struct HSActionSpace
  actions::AbstractVector{HSAction}
end
# defining the default action space
dist_actions = (0.25, 0.5)
phi_resolution = (pi/4)
phi_actions = (-pi:phi_resolution:(pi-phi_resolution))
HSActionSpace() = vec([zero(HSAction()), (HSAction(d, phi) for d in dist_actions, phi in phi_actions)...])
apply_action(p::Pose, a::HSAction) = Pose(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d, p.phi)

@with_kw struct HSRewardModel
  living_penalty::Float64 = -2
  control_cost::Float64 = -0.1
  collision_penalty::Float64 = -100.0
  move_to_goal_reward::Float64 = 0.1
  target_reached_reward::Float64 = 100.0
  left_room_penalty::Float64 = -100.0
end

"""
The MDP representation of the fully observable HumanSwitching problem. This MDP
formulation is used to derive the POMDP formulation for the partially
observable problem.

Parameters:

- `TM` the type of the transition model
- `AS` the type of the action space
"""
@with_kw struct HSMDP{TM, AS} <: MDP{HSState, HSAction}
  room::RoomRep = RoomRep()
  transition_model::TM = PControlledHumanTransition()
  aspace::AS = HSActionSpace()
  reward_model = HSRewardModel()
  agent_min_distance::Float64 = 1.0
  known_external_initstate::Union{HSState, Nothing} = nothing
end

"""
The POMDP representation (thus partially observable) formulation of the
HumanSwitching problem.

Fields:

- `sensor::TS` a struct specifying the sensor used
- `mdp::HSDMP` a MDP version of this problem

Parameters:

- `TS` the type of the sensor
- `O` the type of of observations the sensor model returns
- `M` the type of the underlying MDP
"""
@with_kw struct HSPOMDP{TS, O, M} <: POMDP{HSState, HSAction, O}
  sensor::TS = ExactPositionSensor()
  mdp::M = HSMDP()
end

function HSPOMDP(sensor::HSSensor, mdp::HSMDP)
  HSPOMDP{typeof(sensor), obstype(sensor), typeof(mdp)}(sensor, mdp)
end

function generate_hspomdp(sensor::HSSensor, transition_model::HSTransitionModel, rng::AbstractRNG;
                          room::RoomRep=RoomRep(),
                          aspace=HSActionSpace(),
                          reward_model::HSRewardModel=HSRewardModel(),
                          agent_min_distance::Float64=1.0,
                          known_external_initstate::Union{HSState, Nothing}=nothing)

  generate_own_init_state = (known_external_initstate === nothing)

  # if no explicit fixed stated for this problem was provided, we generate it
  if generate_own_init_state
    known_external_initstate = rand_state(room, rng)
  end

  mdp = HSMDP(;room=room,
              transition_model=transition_model,
              aspace=aspace,
              reward_model=reward_model,
              agent_min_distance=agent_min_distance,
              known_external_initstate=known_external_initstate)

  return generate_own_init_state ? (HSPOMDP(sensor, mdp), known_external_initstate) : HSPOMDP(sensor, mdp)
end

const HSModel = Union{HSMDP, HSPOMDP}
POMDPs.discount(HSModel) = 0.8  # TODO: maybe move to struct field

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
transition_model(m::HSModel) = mdp(m).transition_model
reward_model(m::HSModel) = mdp(m).reward_model
room(m::HSModel) = mdp(m).room
agent_min_distance(m::HSModel) = mdp(m).agent_min_distance

POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)

# Implementing the main API of the generative interface
"""
generate_s

Generates the next state given the last state and the taken action
"""
# this simple forwards to the different transition models
function POMDPs.generate_s(m::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  @assert (a in actions(m))
  generate_s(mdp(m), s, a, rng)
end

# human controlled by simple P-controller
function human_p_transition(s::HSState)::Tuple{Pose, Pose}
  human_velocity = min(0.3, human_dist_to_target(s)) #m/s
  vec2target = human_vec_to_target(s)
  target_direction = normalize(vec2target)
  current_walk_direction = @SVector [cos(s.human_pose.phi), sin(s.human_pose.phi)]
  walk_direction = (target_direction + current_walk_direction)/2
  # new position:
  human_pose_p::Pose = s.human_pose
  if !any(isnan(i) for i in target_direction)
    xy_p = s.human_pose[1:2] + walk_direction * human_velocity
    phi_p = atan(walk_direction[2], walk_direction[1])
    human_pose_p = [xy_p..., phi_p]
  end

  return human_pose_p, s.human_target
end

# helper funciton to access the deterministic P controlled human transition
function POMDPs.generate_s(m::HSMDP{PControlledHumanTransition, <:Any}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # assembling the new state
  human_pose_p, human_target_p = human_p_transition(s)
  if human_reached_target(s)
    human_target_p = rand(rng, corner_poses(room(m)))
  end

  # a deterministic robot transition model
  robot_pose_p = apply_action(s.robot_pose, a)
  robot_target_p = s.robot_target

  HSState(human_pose_p, human_target_p, robot_pose_p, robot_target_p)
end

# same as above but with AWGN
function POMDPs.generate_s(m::HSMDP{PControlledHumanAWGNTransition, <:Any}, s::HSState, a::HSAction, rng::AbstractRNG)::HSState
  # first get the deterministic version
  human_pose_p, human_target_p = human_p_transition(s)
  # add AWGN to the pose and have small likelyhood of chaning the target
  do_resample = rand(rng) > 0.99 || human_reached_target(s)
  human_target_p = do_resample ? rand(rng, corner_poses(room(m))) : human_target_p
  # a deterministic robot transition model

  # TODO: Just proof of concept! MOVE to a proper place!
  # TODO: The robot should have it's own transition statistics
  transition_noise = rand(rng, MvNormal([0, 0, 0], transition_model(m).pose_cov))
  robot_pose_p = apply_action(s.robot_pose, a) + [transition_noise[1:2]..., 0]
  robot_target_p = s.robot_target

  HSState(human_pose_p + rand(rng, MvNormal([0, 0, 0], transition_model(m).pose_cov)), human_target_p,
          robot_pose_p, robot_target_p)
end

@with_kw struct HSObservation <:FieldVector{5, Float64}
  h_x::Float64 = 0
  h_y::Float64 = 0
  h_phi::Float64 = 0
  r_x::Float64 = 0
  r_y::Float64 = 0
end

# some convenient constructor
HSObservation(s::HSState) = HSObservation(s.human_pose..., s.robot_pose[1:2]...)

"""
generate_o

Generates an observation for an observed transition
"""

# In this version the observation is a **deterministic** extraction of the observable part of the state
POMDPs.generate_o(m::HSPOMDP{ExactPositionSensor, HSObservation, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSObservation = HSObservation(sp)
POMDPs.generate_o(m::HSPOMDP{NoisyPositionSensor, HSObservation, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG)::HSObservation = HSObservation(rand(rng, POMDPs.observation(m, sp)))

# TODO: This is a bit ugly. There should be away to directly define a distribution type on a FieldVector
function POMDPs.observation(m::HSPOMDP{NoisyPositionSensor, HSObservation, <:Any}, s::HSState)
  # TODO: do this properly
  return MvNormal(Array{Float64, 1}([s.human_pose..., s.robot_pose[1:2]...]),
                  Array{Float64, 1}([m.sensor.measurement_cov..., m.sensor.measurement_cov[1:2]...]))
end

"""
isterminal

checks if the state is a terminal state
"""
function POMDPs.isterminal(m::HSModel, s::HSState)
  robot_reached_target(s) || !isinroom(s.robot_pose, room(m)) || has_collision(m, s)
end

"""
initialstate

Sample an initial state and a target state for each agent.
"""
function POMDPs.initialstate(m::HSModel, rng::AbstractRNG)::HSState
  return rand_state(room(m), rng; known_external_state=mdp(m).known_external_initstate)
end

struct HSInitialDistribution{ModelType<:HSModel}
  m::ModelType
end

"""
rand

Defines how to sample from a HSInitialDistribution
"""
POMDPs.rand(rng::AbstractRNG, d::HSInitialDistribution) = initialstate(d.m, rng)

"""
initialstate_distribution

defines the initial state distribution on this
problem set which can be passed to rand
"""
POMDPs.initialstate_distribution(m::HSModel) = HSInitialDistribution(m)

"""
reward

The reward function for this problem.

NOTE: Nothing intereseting here until the agent is also moving
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
  step_reward += rm.move_to_goal_reward * (robot_dist_to_target(s) - robot_dist_to_target(sp))
  # reward for reaching the goal
  if robot_reached_target(sp)
    step_reward += rm.target_reached_reward
  end

  if !isinroom(sp.robot_pose, room(m))
    step_reward += rm.left_room_penalty
  end

  step_reward
end
