@testset "POMDP interface" begin
    # checking whether we can actually succesfully construct all those types
    rng = MersenneTwister(42)
    hbm = HumanPIDBehavior(Room())
    hs_pomdp_exact_o = HSPOMDP(ExactPositionSensor(), gen_hsmdp(rng, human_behavior_model=hbm, physical_transition_noise_model=HSIdentityPTNM()))
    hs_pomdp_noisy_o = HSPOMDP(NoisyPositionSensor(), gen_hsmdp(rng, human_behavior_model=hbm, physical_transition_noise_model=HSIdentityPTNM()))

    s = initialstate(hs_pomdp_exact_o, rng)
    a = HS.HSAction()
    # Transition model, simply checking whether the call is successfull
    sp = HS.generate_s(hs_pomdp_exact_o, s, a, rng)

    # Obsevation model:
    # the deterministic observation model
    o = HS.generate_o(hs_pomdp_exact_o, s, a, sp, rng)
    @test human_pos(sp) == human_pos(o)
    @test robot_pos(sp) == robot_pos(o)

    # the noisy obsevation model
    test_obs_data = collect(human_pos(HS.generate_o(hs_pomdp_noisy_o, s, a, sp, rng)) for i in 1:5)
    dist = norm(mean(test_obs_data) - human_pos(sp))
    @test 0 <= dist <= 0.1

    # Initial state generation
    test_inits_data = [HS.initialstate(hs_pomdp_exact_o, rng) for i in 1:10000]
    r = HS.room(hs_pomdp_exact_o)
    @test all(HS.isinroom(human_pos(td), r) && HS.isinroom(robot_pos(td), r) for td in test_inits_data)
end;

@testset "POMDP Actions" begin
    # check whether the action space contains exactly one zero action
    aspace = HSActionSpace(1.0)[2:end]
    @test count(iszero(a) for a in aspace) === 1

    # check whether the action space is symmetric
    isapproxin(container, external_it) = any(isapprox(it, external_it) for it in container)
    reachable_states = (apply_robot_action(zero(Pos), a) for a in aspace)
    @test all(isapproxin(reachable_states, -s) for s in reachable_states)
end


# this test set checks whether everything is implemented to be pseudo-random.
# Meaning that with the same rng we should get the same result!
@testset "POMDP deterministic checks" begin
    mdp = gen_hsmdp(MersenneTwister(42), physical_transition_noise_model=HSGaussianNoisePTNM())
    pomdp = HSPOMDP(sensor=NoisyPositionSensor(), mdp=mdp)
    a = HS.HSAction()

    # ORDER of everything matters (rng1 and rng2 must undergo the same
    # "transitions")
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)

    # Initial states:
    s1 = initialstate(pomdp, rng1)
    s2 = initialstate(pomdp, rng2)
    @test isequal(s1, s2)

    # Transitions
    sp1 = generate_s(pomdp, s1, a, rng1)
    sp2 = generate_s(pomdp, s2, a, rng2)
    @test isequal(sp1, sp2)

    # Observation
    o1 = generate_o(pomdp, s1, a, sp1, rng1)
    o2 = generate_o(pomdp, s2, a, sp2, rng2)
    @test isequal(o1, o2)
end;

