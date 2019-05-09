#!/bin/bash
#
#SBATCH --job-name=julia_test
#SBATCH --output=results/slurm-%A_%a_%N.out
#SBATCH --account=fc_hybrid

#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24

#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lasse.peters@berkeley.edu

## Run the actual job
# unload all modules, just to make sure we have a clean environment
module purge
# load the julia 1.1 module
module load julia/1.1.0
# run the experiments. The ClusterMangers interace will make sure that
# available resources from within this batch configuration are being used to
# launch workers
julia --project src/run_experiments.jl
