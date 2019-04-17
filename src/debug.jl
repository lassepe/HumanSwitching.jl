using Pkg
using Revise
using HumanSwitching
const HS = HumanSwitching

using Random
using ParticleFilters
using POMDPs

using CPUTime

HS.decoupled(s::HSState) = (human_pos(s), hbs(s))::HSHumanState

function test_prob_obstacle()
    rng = MersenneTwister(1)
    pomdp = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng))

    #println("=========")
    #s = initialstate(pomdp, rng)
    #dump(s)
    #println("=========")
    #sd = decoupled(s)
    #dump(sd)
    #println("=========")
    #human_pos, hbs = sd
    #println("A")
    #dump(human_pos)
    #println("B")
    #dump(hbs)

    # human_pos_intent, hbs_p = human_transition(hbs(s), human_behavior_model(m), m, human_pos(s), rng)

    human_predictor = PredictModel{HSHumanState}((hs::HSHumanState, rng::AbstractRNG) -> begin
                                                     human_pos, hbs = hs
                                                     a = HS.human_transition(hbs, human_behavior_model(pomdp), pomdp, human_pos, rng)
                                                     return a
                                                 end)

    n_particles = 3
    pbp = ParticleBeliefPropagator(human_predictor, n_particles, rng)
    solver = ProbObstacleSolver(belief_propagator=pbp)

    policy = solve(solver, pomdp)
    # println("=========")
    # dump(policy)

    b0 = initialstate_distribution(pomdp)
    CPUtic();
    a = action(policy, b0)
    display(CPUtoq())
end
