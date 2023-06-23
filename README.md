# ImmuNet: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping

This repository contains the implementation of the machine learning model introduced in the following manuscript:

Shabaz Sultan, Mark A. J. Gorris, Evgenia Martynova, Lieke van der Woude, Franka Buytenhuijs, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor:   
A Segmentation-Free Machine Learning Architecture for Immune Landscape Phenotpying in Solid Tumors by Multichannel Imaging.   
biorxiv, 2021, doi: [10.1101/2021.10.22.464548](https://doi.org/10.1101/2021.10.22.464548)

The proposed model enables automated detection and phenotyping of immune cells in multiplex immunohistochemistry data. ImmuNet is designed to be applied in dense tissue environments such as solid tumors, where segmentation-based phenotyping can be inaccurate due to segmentation errors or overlapping cell boundaries. In the ImmuNet architecture, this problem is addressed by inferring the positions and phenotyppes of immune cells directly, without using segmentation as an intermediate step.

This repository contains the source code of the model, scripts to train the model and to perform inference and evaluation. A subset of the immunohistochemistry images used to train and evaluate the model, the corresponding annotations, and the final trained model can be downloaded from [zenodo](https://zenodo.org/record/5638697)

## System requirements 

The model is implemented in Python 3.6.9 and [Tensorflow 1.14.0](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs). ImmuNet training and demo inference has been tested on Ubuntu 18.04 and 20.04 on our own server and on [Google Colab](https://colab.research.google.com/). We advise using the GPU mode to achieve reasonable training and inference times. 

We used an NVIDIA GeForce RTX 2080 Ti GPU (RAM: 11 GB, CUDA Version: 11.0) and an Intel Core i9-9820X @ 3.30GHz CPU (10 cores, 2 threads per core) to train the network and perform inference. The network discussed in the manuscript was trained on 183678 environments (i.e., 63x63x7 input images) taken from 27888 cells. Training was run for 100 epochs, which took ~12 hours on our system.

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

This command will create a lightweight docker image named `immunet` which can be used to run model training, inference and evaluation separately as well as in a pipeline. The example images and annotations are not copied to the image and should be mounted when running the container together with the output directories.

## Demo

### ImmuNet training

To run the training, please download the data sample `tilecache.tar.gz` and annotations `annotations_train.json.gz` from [zenodo](https://zenodo.org/record/5638697). Move `annotations_train.json.gz` to `data/annotations` folder. Then, uncompress the folder with the data sample, move it to the root folder of the repository and run:
 ```
python immunet/train.py
```

The files created during training will be saved in a `train_output` filder. They include the model checkpoint, the final comressed model in `.h5` format and the loss history per epoch.  

It is possible to change the number of epochs, paths to the data and annotations, and some hyperparameters with command-line arguments. See the `main` block for the full overview of arguments. 

### Demo inference

Inference of cells' positions and phenotypes is demonstrated for a single immunohistochemistry image. To run inference with the model used in the paper, download `immunet.h5` from [zenodo](https://zenodo.org/record/5638697) and place it in the `train_output`. Then, select any `components.tiff` from `tilecache`, make an `input` folder inside the root folder of the repository and put it there. Run the command:
```
python immunet/inference.py demo
```

The network prediction will be saved in a `prediction.tsv` file that contains coordinates and phenotyping markers for each predicted cell. In additon, the prediction is visualised. By default, the predicted phenotype maps are merged into an RGB image based on the maximum intensity per pixel and the positions of detected cells are drawn on the map. This visualisation is saved in `phenotype_prediction.png` file. It is also possible to visualise the positions of detected cells on a custom file (e.g. RGB repsesentation of multi-channel images generated by other software) using `--display_image_path` parameter. The supposted image formats are TIFF, PNG and JPEG. This visualisation is saved in `prediction_vis.jpg` file. 

The paths used to look for a model, an example image and to save the prediction as well as some hyperparameters can be changed with command-line arguments. See the `demo(argv)` method for the full overview of arguments. 

A patch of immunohistochemistry image we use for demo inference is rather large, which can cause an OOM exception on some GPUs. In our code, this exception is caught and handled by cropping the middle subregion of size 512x512 from the input image and running inference for this subregion. If you still get OOM, please decrease the subregion size by changing `target_size` parameter in the `crop_image_center` call.

### Evaluation

It is also possible to evaluate a model on a set of annotations. The file with annotations of interest should be placed in `data/annotations/` folder, TIFF files for all tiles that occur in a file with annotations should be in the `tilecache` folder and the model should be in the `train_output` folder. 

The evaluation is performed in 2 steps:

1. Matching model prediction and annotations
2. Calculation of error rate by phenotype based on the results of matching (as in figure 3E in the paper). In addition at this step the list of errors and confusion matrix are generated.

#### Matching

Run

```
python evaluation.py match
```
A .tsv file with matched annotations and model prediction will be saved in the `data/prediction` folder. If you want to evaluate the model on a few different sets of annotations, you can use `--s` command-line argument to add a suffix to a name of a .tsv file. For instance:

```
python evaluation.py match --s train
```

Paths and hyperparameters can be changed with other command-line arguments, see the `match` function.

#### Performance evaluation

A .tsv file obtained at the previous step is the only input needed for evaluation. Run

```
python evaluation.py run
```

The output of the evaluation will be saved in the `demo_evaluation` folder. It includes:

- .csv files with error rates for cell types defined in a panel: 2 separate files for main cell types and subtypes
- Confusion matrices for main cell types and subtypes
- A JSON file with a list of errors. Each entry contains the detailed information about annotation and matched prediction. This can be used to investigate error cases.

To change paths and hyperparameters use command-line arguments, see the `evaluate_arg` function.

### Training, inference and evaluation in a docker container

To run model training for `$EPOCHS` epochs, for images and annotations located at `$IMAGE_PATH` and `$ANNOTATIONS_PATH` respectively and have the model saved at `$OUTPUT_PATH`, run the command:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/data/annotations \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/train_output \
   immunet python train.py --epochs $EPOCHS
```

After training is finished, the inference can be run as:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/input \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_inference \
   immunet python inference.py demo
```

where `$INPUT_PATH` is the path to the folder where the image to run inference for is located, `$MODEL_PATH` is a path where the model is saved and  `$OUTPUT_PATH` a location to save the result.

Evaluation can be run with the commands:
```
cmd=$"python evaluation.py match
python evaluation.py run"

sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_evaluation \
   immunet bash -c "eval $cmd"
```

Finally, to run training and evaluation in a pipeline, execute the following commands:

```
cmd=$"python train.py --epochs $EPOCHS
python evaluation.py match
python evaluation.py run"

sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_evaluation \
   immunet bash -c "eval $cmd"
```

Alternatively, you can use auxiliary scripts located in the `scripts/docker` folder. The scripts assume that all input and output folders are located inside the root repository directory and are named as:

- `tilecache` - folder with images
- `data/annotations` - folder with annotations
- `data/prediction` - folder to save the matched annotations and prediction
- `train_output` - folder to save a model and training history
- `input` - folder where a file to run the inference for is located
- `demo_inference` - folder to save the output of demo inference
- `demo_evaluation` - folder to save the output of demo evaluation

To run the pipeline with a default local folders use the command:
```
scripts/docker/pipeline.sh
```

Custom folders locations and epochs number can be provided via self-explanatory scripts parameters `--image_path`, `--annotations_path`, `--data_path`, `--output_path`, `--model_path`, and `--epochs`.

## Usage for your own data 

We aimed at making the code sufficiently flexible and extendable to make it easier to use for custom data. All scripts have a few comman-line arguments that make the model adjust to somewhat different images.

### Dataset and annotations format

Both dataset and annotations have hierarchical structure: dataset -> slide -> tile. The dataset is a collection of slides that contain a complete image of a tissue specimem. Slides are split into tiles.

The folder hierarchy for the whole dataset is 
```
- root folder
   - dataset1
      - slide1
         - tile1
            components.tiff
         - tile2
         ...
         - tileN
      - slide2
      ...
      - slideN
   - dataset2
   ...
   - datasetN

```

Annotations should be saved in a JSON fitle with the following structure:
```
[ 
   {
      "ds": "dataset1",
      "panel": "panel_name" 
      "slides": [
         {
            "slide": "slide1", 
            "tiles": [
               {
                  "tile": "tile1", 
                  "annotations": [
                     {
                        "id": "5efdd2543de07424cab4fd3c",
                        "type": "Other cell", 
                        "x": 1235, 
                        "y": 108, 
                        "positivity": [1, 1, 1, 1, 1], 
                        "background": true
                     }, 
                     {
                        "id": "5efdd2543de07424cab4fd80",
                        "type": “T cell", 
                        "x": 985, 
                        "y": 846, 
                        "positivity": [5, 5, 1, 1, 1]
                     },
                     ....
                     ]
               }
               ...
            ]
         }
         ...
      ]
   }
]
```

The values of `ds`, `slide` and `tile` attributes should match the corresponding folders where the images are stored. `panel` attribute of the `ds` object identifies the panel used for a dataset and is needed at the evaluation step. Its value should match the id of a panel defined in `panels.py` that is needed to be used for phenotyping (see below).  

Each annotation object should have its coordinates on a tile, id, annotation type and positivity according to the panel. Backgrond annotations (i.e. those places on the images where a lymphocyte should not be detected) should have an attribute `"background": true`. See the paper for details. `positivity` attribute contains an array of the Likert scale positivities for each channel used for phenotyping in the order the channels appear in TIFF files. For instance, in a panel used in the paper, TIFF files have the following channels: 

DAPI, CD3, FOXP3, CD20, CD45RO, CD8, Tumor marker, Autofluorescence

But only  CD3, FOXP3, CD20, CD45RO, CD8 chaneels define phenotypes of interest. Therefore, for a T cell in the example above, the positivity [5,5,1,1,1] means CD3+, FOXP3+, CD20-, CD45RO-, CD8- phenotype. Positivities of all background annotations should be set to [1,1,1,1,1].   

### train.py

`--in_channels_num` - a number of image channels to use in model input,
`--out_markers_num` - a number of phenotype markers to predict,
`--cell_radius` - a radius in pixels to use for labels generation. Increase (decrease) it if your cells are larger (smaller).

### inference.py

`--log_th` - a threshold for LoG blob detection algorithm,
`--min_log_std` - the minimum standard deviation for Gaussian kernel of LoG, decrease to detect smaller cells,
`--max_log_std` - the maximum standard deviation for Gaussian kernel of LoG, increase to detect larger cells.

### evaluation.py

At the matching step same arguments as in `inference.py`.

At the performance evaluation step:

`--marker_th` - a threshold for marker expression (see fig. 3C)
`--radius` - a detection radius to use in micrometers (see fig. 3B)
`--pix_pmm` - number of pixels per micro meter

But **most importantly** you need to add definitions specific to your panel to the `panels.py` file. It includes cell types and implementation of a class for your panel where you define logic for phenotyping. The panel class should inherit from an abstract class `Panel` and implement its abstract methods. See `LymPanel` and `LymNkRoPanel` classes of panels used in the paper to get an idea of how to implement a class for your own panel. Then, your custom class instance should be added to the `panels` dictionary at the bottom of the file. The correct panel id should be specified in a json with annotations at the dataset level, e.g `"panel": "lymphocyte"`. See the provided annotation files.

### Inference for the whole dataset

To obtain the prediction for the whole dataset you can iterate through slides folders, obtain prediction for each tile (`find_cells` function in the `inference.py`) and concatenate predictions of all tiles in a slide. Note that to get a correct position of the cell on a whole slide you need to know the origin of each tile. This information can be extracted either from TIFF files or metadata generated by the software used to split a scanned file into tiles.

