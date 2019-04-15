using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using HumanSwitching
end
