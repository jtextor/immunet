#!/usr/bin/env bash

WORKDIR="$(pwd)"
IMAGE_PATH=$WORKDIR/tilecache
DATA_PATH=$WORKDIR/data
MODEL_PATH=$WORKDIR/train_output
OUTPUT_PATH=$WORKDIR/demo_evaluation

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

cmd=$"python evaluation.py match
python evaluation.py run"

sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_evaluation \
   immunet bash -c "eval $cmd"