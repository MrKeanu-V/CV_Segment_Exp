#!/usr/bin/env bash

# Config
CONDA_PATH="/opt/miniconda3/bin/activate"   # 可以使用whereis activate查找
ENV_NAME="mycv"
script_path="train.py"

# Define Model to Use
declare -A model
model["unet"]="unet"
model["u2net"]="u2net"

source $CONDA_PATH $ENV_NAME

python $script_path --model ${model["u2net"]} --epochs 20