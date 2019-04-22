# TODO: run some straight to goal runs and make sure that the simulated reward is always below the free_space_estimate
# - with custom reward model where there is no penalty for collision with humans or close to human
# - and check only if solution was succesfull
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

