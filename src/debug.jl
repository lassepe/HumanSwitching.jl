using Pkg
using Revise
using HumanSwitching
const HS = HumanSwitching

using Random
using ParticleFilters
using POMDPs
using POMDPModelTools

using CPUTime

HS.decoupled(s::HSState) = (human_pos(s), hbs(s))::HSHumanState

function test_prob_obstacle()
    rng = MersenneTwister(1)
    pomdp = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng;
                                                     human_behavior_model=HumanMultiGoalBoltzmann(;beta_min=20, beta_max=20)))

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

    n_particles = 100
    pbp = ParticleBeliefPropagator(human_predictor, n_particles, rng)
    solver = ProbObstacleSolver(belief_propagator=pbp)

    policy = solve(solver, pomdp)
    # println("=========")
    # dump(policy)

    b0 = initialstate_distribution(pomdp)
    CPUtic();
    a, info = action_info(policy, b0)

    display(CPUtoq())

    visualize_plan(info[:m], info[:belief_predictions])
    return info
end

using Reel

function visualize_plan(m::Union{MDP, POMDP}, belief_predictions::Vector{ParticleCollection}; fps::Int=7, filename::String="$(@__DIR__)/../renderings/debug_prob_obstacle_plan.gif")
    frames = Frames(MIME("image/png"), fps=fps)
    for bp in belief_predictions
        # TODO: maybe rather implement render for another data structure, not
        # NamedTuple but ProbObstaclePlanViz or something
        step = (bp=bp,)
        push!(frames, render(m, step))
    end
    @show write(filename, frames)
end
