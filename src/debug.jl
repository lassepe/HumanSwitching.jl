using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Revise
    using HumanSwitching
end
