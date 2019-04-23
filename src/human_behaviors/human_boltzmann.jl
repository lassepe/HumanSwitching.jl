struct HumanBoltzmannBState <: HumanBehaviorState
    beta::Float64
end

struct HumanBoltzmannToGoalBState <: HumanBehaviorState
    beta::Float64
    goal::Pos
end

@with_kw struct HumanBoltzmannAction <: FieldVector{2, Float64}
    d::Float64 = 0 # distance
    phi::Float64 = 0 # direction
end

function gen_human_aspace(;dist::Float64=0.2, phi_step::Float64=pi/8)
    direction_actions = [i for i in -pi:phi_step:(pi-phi_step)]
    SVector{length(direction_actions)+1, HumanBoltzmannAction}([zero(HumanBoltzmannAction),(HumanBoltzmannAction(dist, direction) for direction in direction_actions)...])
end

apply_human_action(p::Pos, a::HumanBoltzmannAction)::Pos = Pos(p.x + cos(a.phi)*a.d, p.y + sin(a.phi)*a.d)

"""
HumanMultiGoalBoltzmann
"""
@with_kw struct HumanMultiGoalBoltzmann{NA, TA} <: HumanBehaviorModel
    beta_min::Float64
    beta_max::Float64
    goals::Array{Pos, 1} = corner_positions(Room())
    next_goal_generator::Function = uniform_goal_generator
    initial_goal_generator::Function = uniform_goal_generator
    vel_max::Float64 = 1.4
    goal_resample_sigma::Float64 = 0.01
    beta_resample_sigma::Float64 = 0.01

    aspace::SVector{NA, TA} = gen_human_aspace(dist=dt*vel_max)
    _aprob_mem::MVector{NA, Float64} = @MVector(zeros(length(aspace)))
end

vel_max(hbm::HumanMultiGoalBoltzmann) = hbm.vel_max

bstate_type(hbm::HumanMultiGoalBoltzmann) = HumanBoltzmannToGoalBState

function rand_beta(rng::AbstractRNG, hbm::HumanMultiGoalBoltzmann)
    hbm.beta_min == hbm.beta_max ?
    hbm.beta_max : rand(rng, Truncated(Exponential(5), hbm.beta_min, hbm.beta_max))
end

function rand_hbs(rng::AbstractRNG, hbm::HumanMultiGoalBoltzmann)
    return HumanBoltzmannToGoalBState(rand_beta(rng, hbm),
                                      hbm.initial_goal_generator(hbm.goals, rng))
end

function uniform_goal_generator(goals::Array{Pos, 1}, rng::AbstractRNG)
    return rand(rng, goals)::Pos
end
function uniform_goal_generator(current_goal::Pos, goals::Array{Pos, 1}, rng::AbstractRNG)
    return rand(rng) < 0.0 ? current_goal : uniform_goal_generator(goals, rng)::Pos
end


function compute_qval(p::Pos, hbs::HumanBoltzmannToGoalBState, a::HumanBoltzmannAction)
    return -dist_to_pos(apply_human_action(p, a), hbs.goal; p=2)
end

function get_action_distribution(hbm::HumanMultiGoalBoltzmann, hbs::HumanBoltzmannToGoalBState, p::Pos)
    for (i, a) in enumerate(hbm.aspace)
        hbm._aprob_mem[i] = exp(hbs.beta * compute_qval(p, hbs, a))
    end
    return Categorical(Array(normalize!(hbm._aprob_mem, 1)))
end

function free_evolution(hbm::HumanMultiGoalBoltzmann, hbs::HumanBoltzmannToGoalBState, p::Pos, rng::AbstractRNG)
    d = get_action_distribution(hbm, hbs, p)
    sampled_action = hbm.aspace[rand(rng, d)]
    p_p = apply_human_action(p, sampled_action)
end

function human_transition(hbs::HumanBoltzmannToGoalBState, hbm::HumanMultiGoalBoltzmann, m::HSModel,
                          p::Pos, rng::AbstractRNG)
    human_pos_p = free_evolution(hbm, hbs, p, rng)
    # with a small likelyhood, resample a new beta
    beta_p = rand(rng) < hbm.beta_resample_sigma ? rand_beta(rng, hbm) : hbs.beta
    # if close to goal, sample next goal according from generative model
    # representing P(g_{k+1} | g_{k})
    goal_p = ((rand(rng) < hbm.goal_resample_sigma || dist_to_pos(human_pos_p, hbs.goal) < goal_reached_distance(m)) ?
              hbm.next_goal_generator(hbs.goal, hbm.goals, rng) : hbs.goal)

    hbs_p = HumanBoltzmannToGoalBState(beta_p, goal_p)
    return human_pos_p, hbs_p
end
