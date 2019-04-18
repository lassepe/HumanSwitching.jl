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
    rng = MersenneTwister(3)
    pomdp = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng;
                                                     human_behavior_model=HumanMultiGoalBoltzmann(;beta_min=20, beta_max=20)))

    human_predictor = PredictModel{HSHumanState}((hs::HSHumanState, rng::AbstractRNG) -> begin
                                                     human_pos, hbs = hs
                                                     a = HS.human_transition(hbs, human_behavior_model(pomdp), pomdp, human_pos, rng)
                                                     return a
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

using Reel

# TODO: remove
# render_plan(m::HSModel, planning_step::NamedTuple) = ProbObstaclePlanViz(m, planning_step)
function visualize_plan(po::ProbObstaclePolicy, info::NamedTuple; fps::Int=Base.convert(Int, cld(1, HS.dt)), filename::String="$(@__DIR__)/../renderings/debug_prob_obstacle_plan.gif")
    frames = Frames(MIME("image/png"), fps=fps)

    # TODO: think about a better check?
    # @assert length(info.belief_predictions) == length(info.action_sequence)
    # dump(info.action_sequence)
    # dump(info.belief_predictions)
    for i in 1:length(info.action_sequence)
        planning_step = (human_pos=info.human_pos,
                         robot_pos=info.robot_pos,
                         bp=info.belief_predictions[i],
                         robot_prediction=info.state_sequence[i].rp)
        push!(frames, render_plan(po.pomdp, planning_step))
    end
    @show write(filename, frames)
end
