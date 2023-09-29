#!/bin/bash
#SBATCH --job-name=scp_a100
#SBATCH --nodes=1
#SBATCH -A trg@a100
#SBATCH --ntasks-per-node=1

##SBATCH --time=19:00:00
#SBATCH --time=20:00:00

##SBATCH --time=01:59:00
##SBATCH --qos=qos_gpu-dev

#SBATCH --output=scp_a100_%j.out
#SBATCH --error=scp_a100_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1    

#SBATCH -C a100
#SBATCH --cpus-per-task=6

##SBATCH --signal=SIGUSR1@90

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.11.0+py3.9.12/lib
srun python train_scpnet_comp.py
