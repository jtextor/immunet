#!/usr/bin/env bash

WORKDIR="$(pwd)"
DATA_PATH=$WORKDIR/tilecache
ANNOTATIONS_PATH=$WORKDIR/annotations
EPOCHS=100
INPUT_PATH=$WORKDIR/input
OUTPUT_PATH=$WORKDIR/demo-output

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
    -e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -ip|--input_path)
      INPUT_PATH="$2"
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

cmd=$"python train.py --epochs $EPOCHS
python demo-inference.py"

# Run training
docker run --gpus all --rm -it \
   --mount type=bind,source=$DATA_PATH,target=/home/user/immunet/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/immunet/annotations \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/immunet/input \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/immunet/demo-output \
   immunet bash -c "eval $cmd"
