"""
# Utility Types
"""

const dt = 0.15

# the physical representation of a room
@with_kw struct Room
    width::Float64 = 8
    height::Float64 = 8
end

@with_kw struct Pos <: FieldVector{2, Float64}
    x::Float64 = 0   # horizontal position
    y::Float64 = 0   # vertical position
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
                                          0.1]
end

"""
# Robot reward model
Describes the rewards the robot cares about
"""

@with_kw struct HSRewardModel
    discount_factor::Float64 = 0.99
    living_penalty::Float64 = -1
    collision_penalty::Float64 = -1000
    left_room_penalty::Float64 = -1000
    goal_reached_reward::Float64 = 0
    dist_to_human_penalty::Float64 = -20
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
    pos_cov::Array{Float64} = [0.15, 0.15] # the diagonal of the transition noise covariance matrix
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
    human_pos::Pos
    robot_pos::Pos
end

HSExternalState(v::T) where T<:AbstractVector{Float64} = HSExternalState(v[1:2], v[3:4])

function convert(::Type{V}, e::HSExternalState) where V <: AbstractVector
    hp = human_pos(e)
    rp = robot_pos(e)
    return V([hp[1], hp[2], rp[1], rp[2]])
end

@with_kw struct HSState{HBS<:HumanBehaviorState}
    external::HSExternalState
    hbs::HBS
end

external(s::HSState) = s.external
external(s::HSExternalState) = s
hbs(s::HSState{HBS}) where HBS = s.hbs
hbs(m::HumanBehaviorState) = m
internal(s::HSState) = hbs(s::HSState)
compose_state(e::HSExternalState, i::HumanBehaviorState) = HSState(external=e, hbs=i)

human_pos(s::Union{HSState, HSExternalState}) = external(s).human_pos
robot_pos(s::Union{HSState, HSExternalState}) = external(s).robot_pos

"""
# Action (Space) representation
"""
@with_kw struct HSAction <: FieldVector{2, Float64}
    d::Float64 = 0   # distance to travel
    phi::Float64 = 0 # angle of direction in which the distance is traveled
end

# defining the default action space
function HSActionSpace()
    dist_actions = 1.5 * dt
    direction_actions = (-pi:pi/2:(pi-pi/2))

    return vec([zero(HSAction),
                (HSAction(d, phi) for d in dist_actions, phi in direction_actions)...])
end

apply_robot_action(p::Pos, a::HSAction) = Pos(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d)

"""
# POMDP and MDP Representation
- implementing the POMDPs.jl interface
"""
@with_kw struct HSMDP{AS, HBM<:HumanBehaviorModel, PTNM<:HSPhysicalTransitionNoiseModel} <: MDP{HSState, HSAction}
    room::Room = Room()
    aspace::AS = HSActionSpace()
    reward_model::HSRewardModel = HSRewardModel()
    human_behavior_model::HBM = HumanPIDBehavior(room)
    physical_transition_noise_model::PTNM = HSIdentityPTNM()
    robot_goal::Pos = rand_pos(room, Random.GLOBAL_RNG)
    agent_min_distance::Float64 = 0.5
    goal_reached_distance::Float64 = 0.35
    known_external_initstate::Union{HSExternalState, Nothing} = nothing
end

@with_kw struct HSPOMDP{TS, O, M} <: POMDP{HSState, HSAction, O}
    sensor::TS = ExactPositionSensor()
    mdp::M = HSMDP()
end

HSPOMDP(sensor::HSSensor, mdp::HSMDP) = HSPOMDP{typeof(sensor), HSExternalState, typeof(mdp)}(sensor, mdp)

const HSModel = Union{HSMDP, HSPOMDP}

mdp(m::HSMDP) = m
mdp(m::HSPOMDP) = m.mdp
human_behavior_model(m::HSModel) = mdp(m).human_behavior_model
robot_goal(m::HSModel) = mdp(m).robot_goal
reward_model(m::HSModel) = mdp(m).reward_model
physical_transition_noise_model(m::HSModel) = mdp(m).physical_transition_noise_model
room(m::HSModel) = mdp(m).room
agent_min_distance(m::HSModel) = mdp(m).agent_min_distance
goal_reached_distance(m::HSModel) = mdp(m).goal_reached_distance

"""
# Implementation of main POMDP Interface
"""
POMDPs.actions(m::HSModel) = mdp(m).aspace
POMDPs.n_actions(m::HSModel) = length(mdp(m).aspace)
POMDPs.discount(m::HSModel) = reward_model(m).discount_factor

# this simple forwards to the different transition models
function POMDPs.generate_s(m::HSModel, s::HSState, a::HSAction, rng::AbstractRNG)
    @assert (a in actions(m))

    human_pos_intent, hbs_p = human_transition(hbs(s), human_behavior_model(m), m, human_pos(s), rng)
    robot_pos_intent = apply_robot_action(robot_pos(s), a)
    external_intent::HSExternalState = HSExternalState(human_pos_intent, robot_pos_intent)
    # the intended transition is augmented with the physical transition noise
    external_p = apply_physical_transition_noise(physical_transition_noise_model(m), external_intent, rng)

    sp::HSState = HSState(external=external_p, hbs=hbs_p)
end

# In this version the observation is a **deterministic** extraction of the observable part of the state
POMDPs.generate_o(m::HSPOMDP{ExactPositionSensor, HSExternalState, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG) = external(sp)
POMDPs.generate_o(m::HSPOMDP{NoisyPositionSensor, HSExternalState, <:Any},
                  s::HSState, a::HSAction, sp::HSState, rng::AbstractRNG) = HSExternalState(rand(rng, POMDPs.observation(m, sp)))

POMDPs.observation(m::HSPOMDP{ExactPositionSensor, HSExternalState, <:Any}, s::HSState) = Deterministic(s)

# TODO: This is a bit ugly. There should be away to directly define a distribution type on a FieldVector
# at least one should get away with less type conversion
function POMDPs.observation(m::HSPOMDP{NoisyPositionSensor, HSExternalState, <:Any}, s::HSState)
    # TODO: do this properly
    hp = human_pos(s)
    rp = robot_pos(s)
    mc  = m.sensor.measurement_cov
    return MvNormal([hp[1], hp[2], rp[1], rp[2]],
                    [mc[1], mc[2], mc[1], mc[2]])
end
Distributions.pdf(distribution::MvNormal, sample::HSExternalState) = pdf(distribution, convert(Vector{Float64}, sample))

function POMDPs.isterminal(m::HSModel, s::HSState)
    robot_reached_goal(m, s) || !isinroom(robot_pos(s), room(m)) || has_collision(m, s)
end

function POMDPs.initialstate(m::HSModel, rng::AbstractRNG)::HSState
    return rand_state(m, rng; known_external_state=mdp(m).known_external_initstate)
end

struct HSInitialDistribution{ModelType<:HSModel}
    m::ModelType
end

POMDPs.rand(rng::AbstractRNG, d::HSInitialDistribution) = initialstate(d.m, rng)
POMDPs.initialstate_distribution(m::HSModel) = HSInitialDistribution(m)
