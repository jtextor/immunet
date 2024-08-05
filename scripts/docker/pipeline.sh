#!/usr/bin/env bash

WORKDIR="$(pwd)"
IMAGE_PATH=$WORKDIR/data/tilecache
DATA_PATH=$WORKDIR/data
MODEL_PATH=$WORKDIR/train_output
OUTPUT_PATH=$WORKDIR/evaluation
EPOCHS=100

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -ip|--image_path)
      IMAGE_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -dp|--data_path)
      DATA_PATH="$2"
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
    -e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

cmd=$"python immunet/train.py --epochs $EPOCHS
python immunet/evaluation.py run"

sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/data/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_evaluation \
   immunet bash -c "eval $cmd"
