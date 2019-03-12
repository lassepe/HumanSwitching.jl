using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".")

using ParticleFilters
using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPGifs
using BeliefUpdaters
using POMCPOW
using ARDESPOT
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

function test_pomdp_run(runs; render_gif::Bool=false)
    for i_run in runs
        rng = MersenneTwister(i_run)
        scenario_rng = MersenneTwister(i_run+1)
        # setup models

        # the simulation is fully running on PID human model
        ptnm_cov = [0.01, 0.01, 0.01]
        simulation_hbm = HumanPIDBehavior(potential_targets=[HS.rand_pose(RoomRep(), scenario_rng) for i=1:10], goal_change_likelihood=0.01)
        simulation_model = generate_non_trivial_scenario(ExactPositionSensor(),
                                                         simulation_hbm,
                                                         HSGaussianNoisePTNM(pose_cov=ptnm_cov),
                                                         deepcopy(rng))

        # the planner uses a mix of all models
        planning_hbm = HumanBoltzmannModel()

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
        solver = POMCPOWSolver(tree_queries=12000, max_depth=70, criterion=MaxUCB(80),
                               k_action=5, alpha_action=0.1,
                               k_observation=3, alpha_observation=0.15,
                               check_repeat_obs=true,
                               check_repeat_act=true,
                               estimate_value=free_space_estimate, default_action=zero(HSAction), rng=deepcopy(rng))

        planner = solve(solver, planning_model)

        # the simulator uses the exact dynamics (not known to the belief_updater)
        simulator = HistoryRecorder(max_steps=100, show_progress=true, rng=deepcopy(rng))
        sim_hist = simulate(simulator, simulation_model, planner, belief_updater, initialstate_distribution(planning_model), initialstate(simulation_model, rng))

        println(AgentPerformance(simulation_model, sim_hist))
        if render_gif
            makegif(planning_model, sim_hist, filename=joinpath(@__DIR__, "../renderings/$i_run-out.gif"),
                    extra_initial=true, show_progress=true, render_kwargs=(sim_hist=sim_hist, show_info=true))
        else
            return planning_model, sim_hist, planner
        end
    end
end

function visualize(model, sim_hist, planner)
    makegif(model, sim_hist, filename=joinpath(@__DIR__, "../renderings/visualize_debug.gif"),
            extra_initial=true, show_progress=true, render_kwargs=(sim_hist=sim_hist, show_info=true))
end

function tree(model, sim_hist, planner, step=1)
    beliefs = collect(eachstep(sim_hist, "b"))
    b = beliefs[step]
    a, info = action_info(planner, b, tree_in_info=true)
    inbrowser(D3Tree(info[:tree], init_expand=1), "chromium")
end
