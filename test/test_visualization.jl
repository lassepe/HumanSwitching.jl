@testset "POMDP visualization" begin
    mdp = gen_hsmdp(MersenneTwister(42), physical_transition_noise_model=HSIdentityPTNM())
    pomdp = HSPOMDP(sensor=NoisyPositionSensor([0.1,0.1]), mdp=mdp)
    rng = MersenneTwister(42)
    belief_updater = NothingUpdater()
    policy = FunctionPolicy(x->HSAction())
    # this only checks wether the calls work without error (@test_nowarn doesn't
    # work due to progressbar I assume)
    simulator = HistoryRecorder(max_steps=100, show_progress=false, rng=rng)
    sim_hist = simulate(simulator, pomdp, policy, belief_updater)

    @test_nowarn makegif(pomdp, sim_hist, filename=joinpath(@__DIR__, "test_renderings", "makegif_test.gif"),
                         extra_initial=true, show_progress=false, render_kwargs=(sim_hist=sim_hist, show_info=true))
end;

