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
        hbm = @inferred HumanPIDBehavior(Room())
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
        hbm = HumanUniformModelMix(HumanPIDBehavior(Room()),
                                   HumanConstVelBehavior(),
                                   bstate_change_likelihood=0.1)
        hbs = @inferred HS.rand_hbs(rng, hbm)
        s = @inferred HSState(external=e, hbs=hbs)
    end

    @test @testblock quote
        ptnm_cov = [0.01, 0.01]
        hbm = HumanMultiGoalBoltzmann(beta_min=1, beta_max=20)
        hbs = HS.rand_hbs(rng, hbm)
        s = HSState(external=e, hbs=hbs)
        planning_model = HSPOMDP(NoisyPositionSensor(ptnm_cov*10),
                                 gen_hsmdp(rng,
                                           human_behavior_model=hbm,
                                           physical_transition_noise_model=HSIdentityPTNM()))

        @inferred HS.rand_state(planning_model, rng, known_external_state=mdp(planning_model).known_external_initstate)
        @inferred HS.rand_state(planning_model, rng)
        @inferred mdp(planning_model.mdp)
        @inferred initialstate(planning_model, rng)

        @inferred HS.human_transition(hbs, hbm, planning_model, Pos(), rng)
        a = rand(rng, HSActionSpace(1.0)[2:end])
        sp = @inferred HS.generate_s(planning_model, s, a, rng)
        o = @inferred generate_o(planning_model, s, a, sp, rng)

        d = @inferred observation(planning_model, s, a, sp)
        w = @inferred obs_weight(planning_model, s, a, sp, o)
    end
end
