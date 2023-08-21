#!/bin/bash
#SBATCH --job-name=scpnet
##SBATCH --nodes=1
#SBATCH -A trg@v100
#SBATCH --ntasks-per-node=1

#SBATCH --time=50:59:00
#SBATCH --qos=qos_gpu-t4

#SBATCH --output=scpnet_%j.out
#SBATCH --error=scpnet_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10

##SBATCH --cpus-per-task=3
##SBATCH --partition=gpu_p2


srun python train_scpnet_comp.py
