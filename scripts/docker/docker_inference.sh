#!/usr/bin/env bash

WORKDIR="$(pwd)"
INPUT_PATH=$WORKDIR/input
MODEL_PATH=$WORKDIR/model
OUTPUT_PATH=$WORKDIR/demo-output

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -ip|--input_path)
      INPUT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -mp|--model_path)
      MODEL_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -op|--output_path)
      OUTPUT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done


docker run --gpus all --rm -it \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/immunet/input \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/immunet/model \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/immunet/demo-output \
   immunet python demo-inference.py
