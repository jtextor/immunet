# ImmuNet: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping

This repository contains the implementation of the machine learning model introduced in the following manuscript:

Shabaz Sultan, Mark A. J. Gorris, Lieke van der Woude, Franka Buytenhuijs, Evgenia Martynova, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor:   
A SegmentationFree Machine Learning Architecture for Immune Landscape Phenotpying in Solid Tumors by Multichannel Imaging.   
biorxiv, 2021, doi: [10.1101/2021.10.22.464548](https://doi.org/10.1101/2021.10.22.464548)

The proposed model enables automated detection and phenotyping of immune cells in multiplex immunohistochemistry data. ImmuNet is designed to be applied in dense tissue environments such as solid tumors, where segmentation-based phenotyping can be inaccurate due to segmentation errors or overlapping cell boundaries. In the ImmuNet architecture, this problem is addressed by inferring the positions and phenotyppes of immune cells directly, without using segmentation as an intermediate step.

This repository contains the source code of the model and scripts to train the model and to perform inference. A subset of the immunohistochemistry images used to train the model, the corresponding annotations, and the final trained model can be downloaded from https://zenodo.org/record/5638697.

## System requirements 

The model is implemented in Python 3.6.9 and [Tensorflow 1.14.0](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs). ImmuNet training and demo inference has been tested on Ubuntu 18.04 on our own server and on [Google Colab](https://colab.research.google.com/). We advise using the GPU mode to achieve reasonable training and inference times. 

We used an NVIDIA GeForce RTX 2080 Ti GPU (RAM: 11 GB, CUDA Version: 11.0) and an Intel Core i9-9820X @ 3.30GHz CPU (10 cores, 2 threads per core) to train the network and perform inference. The network discussed in the manuscript was trained on 231851 environments (i.e., 63x63x6 input images) taken from 36856 cells. Training was run for 76 epochs, which took ~12 hours on our system.

## Installation guide

### Virtual environment
We strongly recommend creating a virtual environment to run the code. The dependencies are specified in the `requirements.txt` file and the conda environment file `environment.yml`. To create a conda environment with `.yml` file, run the following command in a shell:
```
conda env create -f environment.yml
```
An environment named `immunet` will be created. Activate it with:
```
conda activate immunet
```
If you do not have a GPU, change `tensorflow-gpu` to `tensorflow` in the requirements before creating the environment.

### Docker image 

Another option is to run ImmuNet in a docker container. We provide a Dockerfile that can be used to build an image:
```
docker build -t immunet .
```

This command will create a lightweight docker image named `immunet` which can be used to run model training and inference separately as well as in a pipeline. The example images and annotations are not copied to the image and should be mounted when running the container together with the output directories.

## Demo

### ImmuNet training

To run the training, please download the data sample `tilecache.tar.gz` and annotations `annotations_train.json.gz` from [zenodo](https://zenodo.org/record/5638697). Move `annotations_train.json.gz` to `annotations` folder so that it is relative path was `annotations/annotations_train.json.gz`. If your system automatically uncompressed the file to `annotations_train.json`, compress it with the command:
 ```
gzip annotations_train.json
```
Then, uncompress the folder with the data sample, move it to the root folder of the repository and run:
 ```
python train.py
```
It is possible to change the number of epochs and paths to the data and annotations with command-line arguments:

`--data_path` - a path to a folder that contains an immunohistochemistry data sample, default: `tilecache`   
`--annotations_path` - a path to a file that contains annotations, default: `annotations/annotations_train.json.gz`  
`--epochs` - a number of epochs to run training, default: 100 

### Demo inference

Inference of cells' positions and phenotypes is demonstrated for a single immunohistochemistry image. To run inference with the model use in the paper, download `immunet.h5` from https://zenodo.org/record/5638697, place it in the root folder of the repository and run the command:
```
python demo-inference.py
```
The visualised network output will be saved in the `demo-output` folder as a set of six `.png` files.

- `cell-center-distance-prediction.png` - a map of the predicted distance to the nearest cell center.
- `pseudochannel-{0-4}.png` - maps of the predicted pseudomarkers expression.

The model prediction made based on the network output is saved in `demo-output/prediction.txt`. Each line corresponds to a detected cell and contains its location and phenotype expression in the format:
```
y x pseudomarker0 pseudomarker1 pseudomarker2 pseudomarker3 pseudomarker4
```
Where (y, x) are cell coordinates and pseudomarker{0-4} are expressions of respective pseudomarkers.

The paths to a model used for inference and an example image can be changed with the following command-line arguments:

`--model_path` - a path to a model, default: `immunet.h5`   
`--tile_path` - a path to an image, default: `tilecache/2020-01-27-phenotyping-paper-cytoagars/tonsil01/57055,8734/components.tiff` 

A patch of immunohistochemistry image we use for demo inference is rather large, which can cause an OOM exception on some GPUs. In our code, this exception is caught and handled by cropping the middle subregion of size 512x512 from the input image and running inference for this subregion. If you still get OOM, please decrease the subregion size by changing `target_size` parameter in the `crop_image_center` call.

### Training and inference in a docker container

To run model training for images and annotations located at `$DATA_PATH` and `$ANNOTATIONS_PATH` respectively and have the model saved at `$MODEL_PATH`, run the command:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$DATA_PATH,target=/home/user/immunet/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/immunet/annotations \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/immunet/model \
   immunet python train.py
```

After training is finished, the inference can be run as:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/immunet/input \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/immunet/model \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/immunet/demo-output \
   test_immunet python demo-inference.py
```

where `$INPUT_PATH` is the path to the folder where the image to run inference for is located, `$MODEL_PATH` is a path where the model is saved and  `$OUTPUT_PATH` a location to save the result.

To run training and inference in a pipeline, execute the following commands:

```
cmd=$"python train.py
python demo-inference.py"

docker run --gpus all --rm -it \
   --mount type=bind,source=$DATA_PATH,target=/home/user/immunet/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/immunet/annotations \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/immunet/input \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/immunet/demo-output \
   immunet bash -c "eval $cmd"
```

Alternatively, you can use auxiliary scripts located in the `scripts/docker` folder. The scripts assume that all input and output folders are located inside the root repository directory and are named as:

- `tilecache` - folder with images
- `annotations` - folder with annotations
- `model` - folder with saved model
- `input` - folder where a file to run the inference for is located
- `demo-output` - folder to save the output of demo inference

To run the pipeline with a default local folders use the command:
```
scripts/docker/docker_pipeline.sh
```

Custom folders locations and epochs number can be provided via self-explanatory scripts parameters `--data_path`, `--annotations_path`, `--model_path`, `--input_path`, `--output_path` and `--epochs`.

