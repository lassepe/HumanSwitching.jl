# HumanSwitching.jl

Partially Observable Motion Planning written in Julia.

## Installation Instructions

Make use you have installed:
- julia (preferably version 1.1.0)
- imagemagick (used for rendering gifs)

**Setup POMDPs.jl**

First you need to setup the POMDPs.jl registry so that all dependencies can be found:

To install `POMDPs.jl` and setup the registry, run the following from the Julia `REPL`:
```julia
Pkg.add("POMDPs")
using POMDPs
POMDPs.add_registry()
```

**Install Julia Dependencies**

Start a julia REPL in project mode from the root directory of this repo that you cloned to your machine.

```bash
julia --project
```

Hit the `]` key to enter package mode and install the package dependencies (listed in `Project.toml`) by running

```julia
instantiate
```
## Testing

This package comes with a bunch of unit tests that can be run to see whether the setup was succesfull. In order to run these tests from a julia REPL in project mode enter `]` to enter package mode. Then simply type:

```julia
test
```

# Scripts and Running Experiments on clusters

Instructions can be found in `./bash_scripts/README.md`.
