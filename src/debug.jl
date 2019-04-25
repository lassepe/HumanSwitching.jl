using Revise
using HumanSwitching
using StaticArrays
using LinearAlgebra

function compare_models()
    data_switching = parallel_sim(1:3, "GapChecking"; ignore_uncommited_changes=true)
    data_probObstacle = parallel_sim(1:3, "ProbObstacles"; ignore_uncommited_changes=true)

    println("Discounted reward diff: switching - probObstacles")
    display(data_switching.discounted_reward .- data_probObstacle.discounted_reward)

    return (data_switching, data_probObstacle)
end
