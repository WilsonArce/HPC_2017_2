#!/bin/bash

#SBATCH --job-name=imgTransform
#SBATCH --output=res_imgTransform.txt
#SBATCH --ntasks=5
#SBATCH --nodes=5
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

FILES=images/*

for f in $FILES
do
  file=${f##*/}
  echo -n ${file%.*} = [
  for i in {1..10};
  do
    ./sobelFilterGCM $f
    if [ $i -lt 3 ]
    then
      echo -n ","
    fi
  done
  echo "]"
done