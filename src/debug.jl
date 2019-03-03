using Pkg

if !haskey(Pkg.installed(), "HumanSwitching")
    # load the environment if not yet done
    jenv = joinpath(dirname(@__FILE__()), "../.")
    Pkg.activate(jenv)
    @info("Activated Environment")
end

using ParticleFilters
using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPGifs
using BeliefUpdaters
using POMCPOW
using MCTS

using Blink
using Revise
using HumanSwitching
const HS = HumanSwitching
using Printf
using Compose
using Random
using ProgressMeter
using D3Trees

using Profile
using ProfileView
using Test

include("estimate_value_policies.jl")

function test_custom_particle_filter(runs)
    for i_run in runs
        rng = MersenneTwister(i_run)
        # setup models

        # the simulation is fully running on PID human model
        ptnm_cov = [0.01, 0.01, 0.01]
        simulation_hbm = HumanPIDBehavior(RoomRep(); goal_change_likelihood=0.01)
        simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                         simulation_hbm,
                                                         HSGaussianNoisePTNM(pose_cov=ptnm_cov),
                                                         deepcopy(rng))

        # the planner uses a mix of all models
        planning_hbm = HumanBoltzmannModel(min_max_beta=[0, 10])

        # planning_hbm = HumanConstVelBehavior()
        planning_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10),
                                          planning_hbm,
                                          HSIdentityPTNM(),
                                          simulation_model,
                                          deepcopy(rng))

        n_particles = 2000
        # the blief updater is run with a stocahstic version of the world
        belief_updater = BasicParticleFilter(planning_model, SharedExternalStateResampler(n_particles), n_particles, deepcopy(rng))
        # the policy plannes without a model as it is always the same action
        solver = POMCPOWSolver(tree_queries=6000, max_depth=70, criterion=MaxUCB(80),
                               k_action=4, alpha_action=0.1,
                               k_observation=1, alpha_observation=0,
                               estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))
        planner = solve(solver, planning_model)

        # the simulator uses the exact dynamics (not known to the belief_updater)
        simulator = HistoryRecorder(max_steps=100, show_progress=true, rng=deepcopy(rng))
        sim_hist = simulate(simulator, simulation_model, planner, belief_updater, initialstate_distribution(planning_model), initialstate(simulation_model, rng))

        # a, info = action_info(planner, initialstate_distribution(model), tree_in_info=true)
        # inchrome(D3Tree(info[:tree], init_expand=3))

        println(AgentPerformance(simulation_model, sim_hist))
        # makegif(simulation_model, sim_hist, filename=joinpath(@__DIR__, "../renderings/visualize_debug.gif"), extra_initial=true, show_progress=true)
        return planning_model, sim_hist
    end
end

function visualize(belief_updater_model, sim_hist)
    makegif(belief_updater_model, sim_hist, filename=joinpath(@__DIR__, "../renderings/visualize_debug.gif"), extra_initial=true, show_progress=true)
end

function profile_testrun()
    @time test_custom_particle_filter(4);
    Profile.init(n=10^7)
    Profile.clear()
    Profile.clear_malloc_data()
    @profile test_custom_particle_filter(4);
    ProfileView.view()
end

function profile_detailed()
    rng = MersenneTwister(1)
    ptnm_cov = [0.01, 0.01, 0.01]
    hbm = HumanUniformModelMix(submodels=[HumanPIDBehavior(RoomRep(),
                                                           goal_change_likelihood=0.01),
                                          HumanBoltzmannModel(min_max_beta=[0, 10])],
                               bstate_change_likelihood=0.1)

    planning_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10),
                                      hbm,
                                      HSIdentityPTNM(),
                                      deepcopy(rng))


    @info "Initial State Profiling"
    Profile.clear()
    Profile.clear_malloc_data()
    @time for i in 1:100000
        initialstate(planning_model, rng)
    end

    s = initialstate(planning_model, rng)
    @info "generate_s profiling"
    Profile.clear()
    Profile.clear_malloc_data()
    @time for i in 1:100000
        s = generate_s(planning_model, s, rand(rng, HSActionSpace()), rng)
    end

    # @info "code_warntype: human_behavior_model"
    # @code_warntype human_behavior_model(planning_model)
    # @code_warntype initialstate(planning_model, rng)
end

function profile_type_stability()
    # external state
    rng = MersenneTwister(1)
    e = @inferred HSExternalState(Pose(), Pose())

    # Constant Velocity
    hbm = @inferred HumanConstVelBehavior()
    hbs = @inferred HS.rand_hbs(rng, hbm)
    s = @inferred HSState(external=e, hbs=hbs)

    # PID
    hbm = @inferred HumanPIDBehavior(RoomRep(), goal_change_likelihood=0.1)
    hbs = @inferred HS.rand_hbs(rng, hbm)
    s = @inferred HSState(external=e, hbs=hbs)

    # Boltzmann
    hbm = @inferred HumanBoltzmannModel()
    hbs = @inferred HS.rand_hbs(rng, hbm)
    s = @inferred HSState(external=e, hbs=hbs)

    # Uniform Mix
    hbm = @inferred HumanUniformModelMix(HumanPIDBehavior(RoomRep(), goal_change_likelihood=0.01),
                                         HumanBoltzmannModel(min_max_beta=[0, 10]),
                                         bstate_change_likelihood=0.1)
    # TODO: Stabilize type
    hbs = HS.rand_hbs(rng, hbm)
    s = @inferred HSState(external=e, hbs=hbs)

    ptnm_cov = [0.01, 0.01, 0.01]
    hbm = @inferred HumanBoltzmannModel()
    planning_model = generate_hspomdp(NoisyPositionSensor(ptnm_cov*10),
                                      hbm,
                                      HSIdentityPTNM(),
                                      deepcopy(rng))

    @inferred HS.rand_state(planning_model, rng, known_external_state=mdp(planning_model).known_external_initstate)
    @inferred HS.rand_state(planning_model, rng)
    @inferred mdp(planning_model.mdp)
    @inferred initialstate(planning_model, rng)

    @inferred HS.human_transition(hbs, hbm, planning_model, Pose(), rng)
    a = rand(rng, HSActionSpace())
    sp = @inferred HS.generate_s(planning_model, s, a, rng)
    o = @inferred generate_o(planning_model, s, a, sp, rng)

    d = @inferred observation(planning_model, s, a, sp)
    w = @inferred obs_weight(planning_model, s, a, sp, o)
end
