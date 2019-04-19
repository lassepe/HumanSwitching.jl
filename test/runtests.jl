using Test
using Suppressor

using HumanSwitching
const HS = HumanSwitching

using Random
using LinearAlgebra
using Statistics

using BeliefUpdaters
using ParticleFilters
using POMDPs
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPGifs
using Parameters

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

@testset "POMDP Actions" begin
    # check whether the action space contains exactly one zero action
    aspace = HSActionSpace(1.0)[2:end]
    @test count(iszero(a) for a in aspace) === 1

    # check whether the action space is symmetric
    isapproxin(container, external_it) = any(isapprox(it, external_it) for it in container)
    reachable_states = (apply_robot_action(zero(Pos), a) for a in aspace)
    @test all(isapproxin(reachable_states, -s) for s in reachable_states)
end

@testset "Estimate Value Policies" begin
    # check whether the free space estimate is actually optimistic
    n_samples = 10000
    rng = MersenneTwister(1337)
    mdp = gen_hsmdp(rng, physical_transition_noise_model=HSIdentityPTNM())

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

@testset "Adversarial Sampling" begin
    test_data = parallel_sim(1:50, "StraightToGoal";
                             problem_instance_keys=["CornerGoalsNonTrivial"],
                             ignore_uncommited_changes=true)
    # any non-trivial scenario should not be solved succesfully with the straight to goal policy
    @test all(test_data.final_state_type .== "failure")
end


# defining a simple test search problem
struct GridPosition
    x_idx::Int
    y_idx::Int
end

struct GridAction
    dx::Int
    dy::Int
end

apply_action(s::GridPosition, a::GridAction) = GridPosition(s.x_idx + a.dx, s.y_idx + a.dy)

@with_kw struct GridNavigationProblem <: SearchProblem{GridPosition}
    grid_dimensions::Tuple{Int, Int} = (10, 10)
    obstacles::Vector{GridPosition} = []
    aspace::Vector{GridAction} = [GridAction(0, 0),
                                  GridAction(1, 0), GridAction(-1, 0),
                                  GridAction(0, 1), GridAction(0, -1)]
end

HS.start_state(p::GridNavigationProblem) = GridPosition(1, 1)
HS.is_goal_state(p::GridNavigationProblem, s::GridPosition) = s == GridPosition(p.grid_dimensions...)
on_grid(p::GridNavigationProblem, s::GridPosition) = (1 <= s.x_idx <= p.grid_dimensions[1]) && (1 <= s.y_idx <= p.grid_dimensions[2])

function HS.successors(p::GridNavigationProblem, s::GridPosition)
    successors::Vector{Tuple{GridPosition, GridAction, Int}} = []
    sizehint!(successors, length(p.aspace))
    for a in p.aspace
        sp = apply_action(s, a)
        if !on_grid(p, sp) || sp in p.obstacles
            continue
        end
        # we are solving a minimum time problem, the step cost is always 1
        push!(successors, (sp, a, 1))
    end
    return successors
end

@testset "SearchTest" begin
    # tuples of (dimensions, obstacles, optimal_nsteps, solvable)
    test_setups::Vector{Tuple{Tuple{Int, Int}, Vector, Int, Bool}} =
    [
     # 1x1 grid, no action needed to reach the goal
     ((1, 1), [], 0, true),
     # empty 10x10 grid, solvable in 18 steps
     ((10, 10), [], 18, true),
     # 10x10 grid with wall in the middle that has a gap, solvable in 18 steps
     ((10, 10), [GridPosition(5, y) for y in 1:9], 18, true),
     # 10x10 grid with wall all the way, not solvable
     ((10, 10), [GridPosition(5, y) for y in 1:10], -1, false),
     # 5x5 grid with two walls, gap at top and bottom, solvable in 16 steps
     ((5, 5), [[GridPosition(2, y) for y in 1:4]...,
               [GridPosition(4, y) for y in 2:5]...], 16, true)
    ]

    for (test_dims, test_obstacles, optimal_nsteps, solvable) in test_setups
        p = GridNavigationProblem(grid_dimensions=test_dims,
                                  obstacles=test_obstacles)
        # using the manhattan distance as a heuristic
        h = (s::GridPosition) -> abs(s.x_idx - p.grid_dimensions[1]) + abs(s.y_idx - p.grid_dimensions[2])
        if solvable
            aseq, sseq = astart_search(p, h)
            @test length(aseq) == optimal_nsteps
        else
            @test_throws ErrorException astart_search(p, h)
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
        hbm = @inferred HumanPIDBehavior(Room())
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
        hbm = HumanUniformModelMix(HumanPIDBehavior(Room()),
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
