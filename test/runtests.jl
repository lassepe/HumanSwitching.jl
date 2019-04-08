using Test
using Suppressor

using HumanSwitching
const HS = HumanSwitching

using Random
using LinearAlgebra
using Statistics

using BeliefUpdaters
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPGifs

macro testblock(ex)
    quote
        try
            $(esc(eval(ex)))
            true
        catch err
            isa(err, ErrorException) ? false : rethrow(err)
        end
    end
end

@testset "normalized_angle_diff" begin
    @test isapprox(HS.normalized_angle_diff(pi/2), pi/2)
    @test isapprox(HS.normalized_angle_diff(-pi/2), -pi/2)
    @test isapprox(HS.normalized_angle_diff(2pi), 0)
    @test isapprox(HS.normalized_angle_diff(1.5pi), -0.5pi)
    @test isapprox(HS.normalized_angle_diff(Float64(pi)), pi)
end;

@testset "POMDP interface" begin
    # checking whether we can actually succesfully construct all those types
    rng = MersenneTwister(42)
    hbm = HumanPIDBehavior(RoomRep())
    hs_pomdp_exact_o = generate_hspomdp(ExactPositionSensor(), hbm, HSIdentityPTNM(), rng)
    hs_pomdp_noisy_o = generate_hspomdp(NoisyPositionSensor(), hbm, HSIdentityPTNM(), rng)

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

# this test set checks whether everything is implemented to be pseudo-random.
# Meaning that with the same rng we should get the same result!
@testset "POMDP deterministic checks" begin
    mdp = HSMDP(physical_transition_noise_model=HSGaussianNoisePTNM())
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

@testset "POMDP visualization" begin
    mdp = HSMDP(physical_transition_noise_model=HSIdentityPTNM())
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

@testset "POMDP Actions" begin
    # check whether the action space contains exactly one zero action
    @test count(iszero(a) for a in HSActionSpace()) === 1

    # check whether the action space is symmetric
    isapproxin(container, external_it) = any(isapprox(it, external_it) for it in container)
    reachable_states = (apply_robot_action(zero(Pos), a) for a in HSActionSpace())
    @test all(isapproxin(reachable_states, -s) for s in reachable_states)
end

@testset "Estimate Value Policies" begin
    # check whether the free space estimate is actually optimistic
    n_samples = 10000
    rng = MersenneTwister(1337)
    mdp = HSMDP(physical_transition_noise_model=HSIdentityPTNM())

    for i in 1:n_samples
        # sample some random state
        s = initialstate(mdp, rng)

        # take a random action
        a = rand(rng, actions(mdp))
        # propagate the state according to this action
        sp, r = generate_sr(mdp, s, a, rng)

        # compute the estimated reward as the difference in value
        v_s = free_space_estimate(mdp, s)
        v_sp = free_space_estimate(mdp, sp)
        r_est = isterminal(mdp, sp) ? v_s : v_sp - v_s * discount(mdp)

        if isterminal(mdp, s)
            @test iszero(v_s)
            continue
        else
            # the free space estimate must be optimistic at every step to be
            # admissable and consistent in a heuristic sense
            if r_est < r - eps(Float32)
                println("Failed for:")
                println("s:")
                dump(s)
                println("sp:")
                dump(sp)
                println("a:")
                dump(a)
                println("v_s:")
                dump(v_s)
                println("v_sp:")
                dump(v_sp)
            end
            @test r_est >= r - eps(Float32)
        end
    end
end

@testset "Type Inference tests" begin
    # external state
    rng = MersenneTwister(1)
    e = @inferred HSExternalState(Pos(), Pos())

    # Constant Velocity
    @test @testblock quote
        hbm = @inferred HumanConstVelBehavior()
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    # PID
    @test @testblock quote
        hbm = @inferred HumanPIDBehavior(RoomRep())
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    # Boltzmann
    @test @testblock quote
        hbm = HumanBoltzmannModel()
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    # multi goal human
    @test @testblock quote
        hbm = HumanMultiGoalBoltzmann(beta_min=1, beta_max=20)
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    # Uniform Mix
    # TODO: Stabilize type
    @test_broken @testblock quote
        hbm = HumanUniformModelMix(HumanPIDBehavior(RoomRep()),
                                   HumanBoltzmannModel(),
                                   bstate_change_likelihood=0.1)
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    @test @testblock quote
        ptnm_cov = [0.01, 0.01]
        hbm = HumanBoltzmannModel()
        hbs = HS.rand_hbs(rng, hbm)
        s = HSState(external=e, hbs=hbs)
        planning_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10),
                                          hbm,
                                          HSIdentityPTNM(),
                                          deepcopy(rng))

        @inferred HS.rand_state(planning_model, rng, known_external_state=mdp(planning_model).known_external_initstate)
        @inferred HS.rand_state(planning_model, rng)
        @inferred mdp(planning_model.mdp)
        @inferred initialstate(planning_model, rng)

        @inferred HS.human_transition(hbs, hbm, planning_model, Pos(), rng)
        a = rand(rng, HSActionSpace())
        sp = @inferred HS.generate_s(planning_model, s, a, rng)
        o = @inferred generate_o(planning_model, s, a, sp, rng)

        d = @inferred observation(planning_model, s, a, sp)
        w = @inferred obs_weight(planning_model, s, a, sp, o)
    end
end
