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
using BenchmarkTools

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
    ptnm_cov = [0.01, 0.01, 0.01]

    hbm = HumanBoltzmannModel(min_max_beta=[0, 10])
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
    @profile for i in 1:10000
        s = generate_s(model, s, rand(rng, as), rng)
    end
    ProfileView.view()
end
