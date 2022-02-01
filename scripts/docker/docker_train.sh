#!/usr/bin/env bash

WORKDIR="$(pwd)"
DATA_PATH=$WORKDIR/tilecache
ANNOTATIONS_PATH=$WORKDIR/annotations
MODEL_PATH=$WORKDIR/model
EPOCHS=100

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -dp|--data_path)
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -ap|--annotations_path)
      ANNOTATIONS_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -mp|--model_path)
      MODEL_PATH="$2"
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


docker run --gpus all --rm -it \
   --mount type=bind,source=$DATA_PATH,target=/home/user/immunet/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/immunet/annotations \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/immunet/model \
   immunet python train.py --epochs $EPOCHS
