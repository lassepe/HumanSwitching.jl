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

include("test_utils.jl")
include("test_pomdp_main.jl")
include("test_visualization.jl")
include("test_estimate_value_policies.jl")
include("test_simulation_utils.jl")
include("test_type_inference.jl")
