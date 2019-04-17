using Pkg
using Revise
using HumanSwitching

using Random
using ParticleFilters

decoupled(s::HSState) = (human_pos(s), hbs(s))::HSHumanState

function test_prob_obstacle()
    rng = MersenneTwister(1)
    pomdp = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng))

    s = initialstate(pomdp, rng)
    dump(s)
    sd = decoupled(s)
    dump(sd)

    # human_predictor = PredictModel{HSHumanState}
    # bp = BeliefPropagator()

    # solver = ProbObstacleSolver()
end
