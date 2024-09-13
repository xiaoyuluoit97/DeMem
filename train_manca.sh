#!/bin/bash

# 初始化CUDA设备
cuda_device=0

# Define the list of specific models to train
models=(18 20 39 42 44 46 52 53 64 67 71 78 85 92 95 98 105 116 118 119 124)

# Loop over the specified models
for i in "${models[@]}"; do
  # Calculate t_mode value
  t_mode=$((i + 1))

  # Construct the command
  cmd="python train_luo.py --train --s_model $i --t_mode $t_mode --max_grad_norm 10 --aug_type DP --cuda $cuda_device"

  # Print and execute the command
  echo "Executing: $cmd"
  $cmd

  # Wait for the current task to complete
  wait
done
