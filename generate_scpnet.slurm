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

##SBATCH -C v100-32g
##SBATCH --cpus-per-task=10

#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu_p2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.11.0+py3.9.12/lib
srun python generate_scpnet_output.py
