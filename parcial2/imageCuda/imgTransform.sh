#!/bin/bash

#SBATCH --job-name=imgTransform
#SBATCH --output=res_imgTransform
#SBATCH --ntasks=5
#SBATCH --nodes=5
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

for i in {1..3};
do
  echo "Iteration $i"
  ./sobelFilterGCM images/image10k.jpg
done