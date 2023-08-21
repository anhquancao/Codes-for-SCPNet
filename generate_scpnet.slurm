#!/bin/bash
#SBATCH --job-name=gen_scpnet
##SBATCH --nodes=1
#SBATCH -A trg@v100
#SBATCH --ntasks-per-node=1

#SBATCH --time=08:59:00

#SBATCH --output=gen_scpnet_%j.out
#SBATCH --error=gen_scpnet_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10


srun python generate_scpnet_output.py
