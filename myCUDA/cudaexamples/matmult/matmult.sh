#!/bin/bash

#SBATCH --job-name=matmult
#SBATCH --output=res_matmult
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=1

./matmult2