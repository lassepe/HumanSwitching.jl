#!/bin/bash
#
#SBATCH --job-name=julia_test
#SBATCH --output=results/slurm-%A_%a_%N.out
#SBATCH --account=fc_hybrid

#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --mem=32G

#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lasse.peters@berkeley.edu

## Run the actual job
module load julia/1.1.0
julia --project src/run_experiments.jl
