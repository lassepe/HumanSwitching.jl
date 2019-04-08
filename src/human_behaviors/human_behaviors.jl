"""
# HumanBehaviorModel
"""
# basic models don't have further submodels
select_submodel(hbm::HumanBehaviorModel, hbs_type::Type{<:HumanBehaviorState}) = hbm
select_submodel(hbm::HumanBehaviorModel, hbs::HumanBehaviorState)::HumanBehaviorModel = select_submodel(hbm, typeof(hbs))

function free_evolution end
function rand_hbs end
