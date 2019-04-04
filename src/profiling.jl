using Profile
using ProfileView

using BenchmarkTools

function profile_testrun()
    @time test_parallel_sim(4:4);
    Profile.init(n=10^7)
    Profile.clear()
    Profile.clear_malloc_data()
    @profile test_parallel_sim(4:4);
    ProfileView.view()
end

function profile_detailed()
    ptnm_cov = [0.01, 0.01, 0.01]

    hbm = HumanBoltzmannModel()
    profile_hbm(hbm)
end

function profile_hbm(hbm)
    rng = MersenneTwister(1)
    ptnm_cov = [0.01, 0.01, 0.01]
    model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10),
                             hbm,
                             HSIdentityPTNM(),
                             deepcopy(rng))

    @info string(typeof(hbm))
    @info "initialstate"
    b = @benchmark initialstate($model, $rng)
    display(b)

    s = initialstate(model, rng)
    @info "generate_s profiling"
    if hbm isa HumanUniformModelMix
        for submodel in hbm.submodels
            hbs = HS.rand_hbs(rng, submodel)
            println(typeof(hbs))
            s = HSState(external=external(s), hbs=hbs)
            b = @benchmark generate_s($model, $s, rand($rng, $HSActionSpace()), $rng)
        end
    else
        b = @benchmark generate_s($model, $s, rand($rng, $HSActionSpace()), $rng)
    end
    display(b)

    as = HSActionSpace()
    Profile.clear()
    Profile.clear_malloc_data()
    function f(s, model, as, rng)
        for i in 1:10000
            s = generate_s(model, s, rand(rng, as), rng)
        end
        return s
    end
    @profile f(s, model, as, rng)
    ProfileView.view()
end

function profile_rollout(run::Int)
    rng = MersenneTwister(run)
    ptnm_cov = [0.01, 0.01, 0.01]
    hbm = HumanPIDBehavior(potential_targets=[Pos(5, 5, 0)], goal_change_likelihood=0.01)
    model = generate_non_trivial_scenario(ExactPositionSensor(),
                                          hbm,
                                          HSGaussianNoisePTNM(pos_cov=ptnm_cov),
                                          deepcopy(rng))
    n_particles = 2000
    belief_updater = BasicParticleFilter(model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))


    K = 10
    rng = MersenneTwister(14)
    rs = MemorizingSource(K, 50)
    Random.seed!(rs, 10)
    b_0 = initialstate_distribution(model)
    scenarios = [i=>rand(rng, b_0) for i in 1:K]
    b = ScenarioBelief(scenarios, rs, 0, false)

    @info "\n\n POMDPs.action"
    rollout_policy = StraightToTarget(model)
    # @code_warntype POMDPs.action(rollout_policy, b)
    bench = @benchmark POMDPs.action($rollout_policy, $b)
    display(bench)

    @info "\n\n generate_sor"
    s = rand(rng, b_0)
    a = POMDPs.action(rollout_policy, b)
    # @code_warntype generate_sor(model, s, a, rng)
    bench = @benchmark generate_sor($model, $s, $a, $rng)
    display(bench)
end
