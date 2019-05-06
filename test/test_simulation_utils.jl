@testset "Adversarial Sampling" begin
    test_data = parallel_sim(1:50, ["StraightToGoal"];
                             problem_instance_keys=["CornerGoalsNonTrivial"],
                             ignore_uncommited_changes=true)
    # any non-trivial scenario should not be solved succesfully with the straight to goal policy
    @test all(test_data.final_state_type .== "failure")
end
