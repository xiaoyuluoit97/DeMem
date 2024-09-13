#!/bin/bash

# 初始化CUDA设备
cuda_device=7

# 循环从1到64，生成并执行训练命令
for i in $(seq 64 80); do
  # 计算t_mode的值
  t_mode=$((i + 1))

  # 构建命令
  cmd="python train_luo.py --train --s_model $i --t_mode $t_mode --aug_type DP_pgdat --cuda $cuda_device"

  # 打印并执行命令
  echo "Executing: $cmd"
  $cmd

  # 等待当前任务完成
  wait

  # 交替CUDA设备
  #if [ $cuda_device -eq 6 ]; then
    #cuda_device=7
  #else
    #cuda_device=6
  #fi
done