using Pkg
using Revise
using HumanSwitching
const HS = HumanSwitching

using Random
using ParticleFilters
using POMDPs
using POMDPModelTools

using CPUTime
using Parameters

HS.decoupled(s::HSState) = (human_pos(s), hbs(s))::HSHumanState

function test_prob_obstacle()
    rng = MersenneTwister(6)
    pomdp = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng;
                                                     human_behavior_model=HumanMultiGoalBoltzmann(;beta_min=20, beta_max=20)))

    human_predictor = PredictModel{HSHumanState}((hs::HSHumanState, rng::AbstractRNG) -> begin
                                                     human_pos, hbs = hs
                                                     return HS.human_transition(hbs, human_behavior_model(pomdp), pomdp, human_pos, rng)
                                                 end)

    n_particles = 50
    pbp = ParticleBeliefPropagator(human_predictor, n_particles, rng)

    solver = ProbObstacleSolver(belief_propagator=pbp)

    policy = solve(solver, pomdp)
    # println("=========")
    # dump(policy)
    b0 = initialstate_distribution(pomdp)
    CPUtic();
    a, info = action_info(policy, b0)

    display(CPUtoq())

    visualize_plan(policy, info)
    return info
end
