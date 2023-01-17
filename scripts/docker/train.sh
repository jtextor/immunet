#!/usr/bin/env bash

WORKDIR="$(pwd)"
IMAGE_PATH=$WORKDIR/tilecache
ANNOTATIONS_PATH=$WORKDIR/data/annotations
OUTPUT_PATH=$WORKDIR/train_output
EPOCHS=100

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -dp|--IMAGE_PATH)
      IMAGE_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -ap|--annotations_path)
      ANNOTATIONS_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -mp|--OUTPUT_PATH)
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


docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/data/annotations \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/train_output \
   immunet python train.py --epochs $EPOCHS
