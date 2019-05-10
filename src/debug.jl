using Revise
using Compose
using HumanSwitching
const HS=HumanSwitching

# maybe fix Require/Revise interaction bug
Revise.track(HumanSwitching, "src/analyze_results.jl")

function compare_models()
    data_switching = parallel_sim(1:3, "GapChecking"; ignore_uncommited_changes=true)
    data_probObstacle = parallel_sim(1:3, "ProbObstacles"; ignore_uncommited_changes=true)

    println("Discounted reward diff: switching - probObstacles")
    display(data_switching.discounted_reward .- data_probObstacle.discounted_reward)

    return (data_switching, data_probObstacle)
end

using POMDPs, POMDPSimulators, POMDPModelTools
using Profile
using BenchmarkTools

function profile_prob_obstacles()
    sim = HS.setup_test_scenario("CornerGoalsNonTrivial",
                                 "HumanMultiGoalBoltzmann_all_goals",
                                 "HumanMultiGoalBoltzmann_all_goals",
                                 "ProbObstacles",
                                 1)

    hist = simulate(sim)

    b = first(collect(eachstep(hist, "b")))
    policy = sim.policy

    # run once for compilation
    a, info = action_info(policy, b)
    display(@benchmark action_info($policy, $b))

    # run the actual profiling
    Profile.clear()
    @profile begin
        a, info = action_info(policy, b)
    end
end
