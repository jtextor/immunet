# ImmuNet: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping

This repository contains the implementation of the machine learning model introduced in the following manuscript:

Shabaz Sultan, Mark A. J. Gorris, Evgenia Martynova, Lieke van der Woude, Franka Buytenhuijs, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor:   
ImmuNet: A Segmentation-Free Machine Learning Pipeline for Immune Landscape Phenotyping in Tumors by Muliplex Imaging.   
biorxiv, 2021, doi: [10.1101/2021.10.22.464548](https://doi.org/10.1101/2021.10.22.464548)

The proposed model enables automated detection and phenotyping of immune cells in multiplex immunohistochemistry data. ImmuNet is designed to be applied in dense tissue environments such as solid tumors, where segmentation-based phenotyping can be inaccurate due to segmentation errors or overlapping cell boundaries. The ImmuNet architecture addresses this problem by inferring immune cell positions and phenotypes directly, without using segmentation as an intermediate step.

This repository contains the source code of the model and scripts to train the model and perform inference and evaluation. A subset of the immunohistochemistry images used to train and evaluate the model, the corresponding annotations, and the final trained model can be downloaded from [zenodo](https://zenodo.org/record/8084976).

## System requirements 

The model is implemented in Python 3.6.9 and [Tensorflow 1.14.0](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs). ImmuNet training and demo inference has been tested on Ubuntu 18.04 and 20.04 on our own server and on [Google Colab](https://colab.research.google.com/). We advise using the GPU mode to achieve reasonable training and inference times. 

We used an NVIDIA GeForce RTX 2080 Ti GPU (RAM: 11 GB, CUDA Version: 11.0) and an Intel Core i9-9820X @ 3.30GHz CPU (10 cores, 2 threads per core) to train the network and perform inference. The network discussed in the manuscript was trained on 183,678 patches of size 63x63x7 taken from 27,888 cell annotations. Training was run for 100 epochs, which took ~12 hours on our system.

## Installation guide

### Virtual environment
We strongly recommend creating a virtual environment to run the code. The dependencies are specified in the `requirements.txt` file and the conda environment file `environment.yml`. To create a conda environment with a `.yml` file, run the following command in a shell::
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

This command creates a lightweight docker image named `immunet` that can be used to run model training, inference, and evaluation separately or in a pipeline. The example images and annotations are not copied to the image and should be mounted when running the container as well as the output directories.

## Demo

All commands below are assumed to be run from the root folder of the repository.

### ImmuNet training

To run the training, please download the data sample `tilecache.tar.gz` and annotations `annotations_train.json.gz` from [zenodo](https://zenodo.org/record/8084976). Move `annotations_train.json.gz` to the `data/annotations` folder. Then, uncompress the data sample folder and move it to the `data` folder. The expected file hierarchy is:
```
- data
   - annotations
      - annotations_train.json.gz
   - tilecache
```

Now, model training can simply be performed with the command:
 ```
python immunet/train.py
```

The files generated during training are saved in a `train_output` folder. They contain a model checkpoint, a final compressed model in `.h5` format, and a loss history per epoch.  

It is possible to change the number of epochs, paths to data and annotations, and some hyperparameters with command line arguments. See the `main` block of `train.py` for a complete list of arguments. 

### Demo inference

Inference of cells positions and phenotypes is demonstrated on a single immunohistochemistry image. To run inference with the model used in the paper, download `immunet.h5` from [zenodo](https://zenodo.org/record/8084976) and place it in the `train_output`. Then, select any `components.tiff` from `tilecache`, make a `demo_input` folder in the root of the repository and put it there. Run the command:
```
python immunet/inference.py demo
```

The network prediction is saved in a `prediction.tsv` file with the predicted coordinates and phenotyping markers of each cell. In additon, the prediction is visualised. By default, the predicted phenotype maps are merged into an RGB image based on the maximum intensity per pixel and the positions of the detected cells are drawn on the map. This visualization is saved in the file `phenotype_prediction.png`. It is also possible to visualize the positions of the detected cells on a custom image (e.g. RGB repsesentation of a multi-channel image generated by other software) using the `--display_image_path` parameter. Supported image formats are TIFF, PNG and JPEG. This visualization is saved in the `prediction_vis.jpg`. 

The paths used search for a model, an example image, and to save the prediction, as well as some hyperparameters can be changed with command line arguments. See the `demo(argv)` method for a complete overview. 

A patch of the immunohistochemistry image we use for the demo inference is rather large, which can cause an OOM exception on some GPUs. In our code, this exception is caught and handled by cropping the middle 512x512 subregion of the input image and running the inference on that region. If you still get OOM, please reduce the subregion size by changing  the`target_size` parameter in the `crop_image_center` call.

### Evaluation

A model can be evaluated on a set of annotations as well. The file with annotations should be placed in the `data/annotations/` folder, TIFF images of all tiles that occur in the annotation file should be in the `tilecache` folder, and the model in the `train_output` folder. 

The evaluation is done in 2 steps, which can be run separately or in a pipeline:

1. Perform inference and match model predictions and annotations
2. Calculate the error rate by phenotype based on the results of matching (as in Figure 3E of the paper). This step also generates the error list and the confusion matrix.

#### Complete evaluation

Run

```
python immunet/evaluation.py run
```
This command 1) performs inference for all tiles with annotations, 2) matches cell predictions and annotations, 3) computes simple performance metrics and statistics, i.e. accuracy per annotation type and confusion matrix. The output is described in more detail in the next two sectons.

#### Matching

Run

```
python immunet/evaluation.py match
```
A .tsv file of matched annotations and model predictions is saved in the `data/prediction` folder. If you want to evaluate the model on different sets of annotations, you can use the `--s` command line argument to add a suffix to the name of a .tsv file. For instance:

```
python immunet/evaluation.py match --s train
```

Paths and hyperparameters can be changed with other command line arguments, see the `match` function.

#### Performance evaluation

A .tsv file obtained in the previous step and a file describing the phenotypes of a panel (see below) are the required inputs. Run

```
python immunet/evaluation.py run
```

The results of the evaluation are saved in the `evaluation` folder. It contains

- .csv files with error rates for phenotypes defined in a panel: 2 separate files for main cell types and subtypes
- Confusion matrices for main cell types and subtypes
- A JSON file with a list of errors. Each entry contains the detailed information about an annotation and the prediction matched with it. This can be used to investigate error cases.

To change paths and hyperparameters, use command line arguments; see the `evaluate_arg` function.

### Training, inference and evaluation in a docker container

To run model training for `$EPOCHS` epochs, for images and annotations located at `$IMAGE_PATH` and `$ANNOTATIONS_PATH` respectively and have the model saved at `$OUTPUT_PATH`, run the command:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/data/tilecache \
   --mount type=bind,source=$ANNOTATIONS_PATH,target=/home/user/data/annotations \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/train_output \
   immunet python immunet/train.py --epochs $EPOCHS
```

After training is finished, the demo inference can be run as:

```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$INPUT_PATH,target=/home/user/demo_input \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/demo_inference \
   immunet python immunet/inference.py demo
```

where `$INPUT_PATH` is a path to the folder with an input TIFF image, `$MODEL_PATH` is a path where the model is saved and `$OUTPUT_PATH` is a folder to save the result.

Evaluation can be run with the command:
```
sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/data/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/evaluation \
   immunet python immunet/evaluation.py run
```

Finally, to run training and evaluation in a pipeline, execute:

```
cmd=$"python immunet/train.py --epochs $EPOCHS
python immunet/evaluation.py run"

sudo docker run --gpus all --rm -it \
   --mount type=bind,source=$IMAGE_PATH,target=/home/user/data/tilecache \
   --mount type=bind,source=$DATA_PATH,target=/home/user/data \
   --mount type=bind,source=$MODEL_PATH,target=/home/user/train_output \
   --mount type=bind,source=$OUTPUT_PATH,target=/home/user/evaluation \
   immunet bash -c "eval $cmd"
```

Alternatively, you can use the auxiliary scripts located in the `scripts/docker` folder. The scripts assume that all input and output folders are located inside the root repository folder, and are named as follows:

- `data/tilecache` - a folder with images
- `data/annotations` - a folder with annotations
- `data/prediction` - a folder to save the matched annotations and predictions
- `data/panels.json` - a json file describing a panel
- `train_output` - a folder to save a model and training history
- `demo_input` - a folder with an input file for demo inference
- `demo_inference` - a folder to save the demo inference output
- `evaluation` - a folder to save the evaluation output

To run the pipeline with default local folders, use the command:
```
scripts/docker/pipeline.sh
```

Custom folders locations and epochs number can be provided via self-explanatory scripts' parameters `--image_path`, `--annotations_path`, `--data_path`, `--output_path`, `--model_path`, and `--epochs`.

## Usage for your own data 

We aimed at making the code sufficiently flexible to make it easier to use for custom data. All scripts have a few comman line arguments that make the model adapt to somewhat different images.

### Dataset and annotations format

Both dataset and annotations have a hierarchical structure: dataset -> slide -> tile. The dataset is a collection of slides that contain a complete image of a tissue specimem. Slides are split into tiles.

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

Annotations should be saved in a JSON file with the following structure:
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

The values of the `ds`, `slide` and `tile` attributes should match the corresponding folders where the images are stored. The `panel` attribute of the `ds` object identifies the panel used for a dataset and is needed in the evaluation step. Its value should match the id of a panel described in the `data/panels.json` file that defines the phenotypes (see below).  

Each annotation object should have its coordinates on a tile, an id, an  annotation type, and a `positivity` attribute with cellular marker expressions. Background annotations (i.e. places in images where there are no lymphocytes) should have the attribute `"background": true`. See the paper for details. The `positivity` attribute contains an array of the marker expression on a Likert scale for each channel used for phenotyping, in the order in which the channels appear in TIFF files. For instance, in a panel used in the paper, the TIFF files have the following channels: 

DAPI, CD3, FOXP3, CD20, CD45RO, CD8, Tumor marker, Autofluorescence.

But only the  CD3, FOXP3, CD20, CD45RO, CD8 channels define phenotypes of interest. Therefore, for a T cell in the example above, the positivity [5,5,1,1,1] means CD3+, FOXP3+, CD20-, CD45RO-, CD8- phenotype. The positivities of all background annotations should be set to [1,1,1,1,1].   

### train.py

`--in_channels_num` - a number of image channels to use in training patches,
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

### Definiton of panels in panel.json

We provide a user-friendly way to define custom panels with the JSON representation contained in the `data/panels.json` file. It is possible to describe multiple panels as a list of dictionaries that define a panel. Custom panels can simply be added to the `data/panels.json`. A dictionary representing a panel should have three attributes: `panel` specifies a panel id, `markers` provides a list of cellular markers used in a panel, and `phenotypes` is a list of dictionaries that describe phenotypes defined by the cellular markers. With these conventions, the JSON that defines the panel used in the paper is the following:

```
{
    "panel": "lymphocyte",
    "markers": ["CD3", "FOXP3", "CD20", "CD45RO", "CD8"],
    "phenotypes": [
      {"type": "T cell", "subtype": "Thelp", "phenotype": {"CD3": true, "FOXP3": false, "CD20": false, "CD45RO": "*", "CD8": false}},
      {"type": "T cell", "subtype": "Treg", "phenotype": {"CD3": "*", "FOXP3": true, "CD20": false, "CD45RO": "*", "CD8": false}},
      {"type": "T cell", "subtype": "Tcyt", "phenotype": {"CD3": "*", "FOXP3": false, "CD20": false, "CD45RO": "*", "CD8": true}},
      {"type": "T cell", "subtype": "Tmemory", "phenotype": {"CD3": false, "FOXP3": false, "CD20": false, "CD45RO": true, "CD8": false}},
      {"type": "B cell", "phenotype": {"CD3": false, "FOXP3": false, "CD20": true, "CD45RO": false, "CD8": false}},
      {"type": "Tumor cell", "background": true}
    ]
}
```

Since ImmuNet is designed to be trained on sparse annotations, there are two types of annotations: **foreground** and **background**. Foreground annotations are intended to be recognized by the model, i.e. these are immune cells we want to detect. Background annotations serve as negative examples of places in the image where we do not want the model to detect a cell. For example, in this definition, we guide a mode to ignore tumor cells. Background phenotypes should be declared by setting the `background` attribute of a phenotype to `true`. Then, even if a `phenotype` dictionary is provided, it is ignored. Other types of background annotations used for the paper are "No cell" (any location without a cell) and "Other cell" (nucleus is visible on DAPI, but no cellular markers are expressed). Even if not specified in the JSON, these phenotypes are added to the panel definition during the JSON parsing.

Some foreground annotatons can have subtypes. In the given panel definition, the "B cell" phenotype does not have any subtypes. Those phenotypes that have subtupes, like "T cell", should be declared as many times as their number of subtypes. And each such phenotype dictionary should have `type`, `subtype` and `phenotype` attributes. The `phenotype` attribute defines the expression of cellular markers expected for a phenotype. It should be a dictionary with **all** markers used in a panel as keys and the values specify the marker expression which can be `true`, `false` or `"*"` (wildcard). The wildcard symbol means that any expression of a given marker is allowed in a phenotype. For instance, the definition of the phenotype "Thelp" is equivalent to CD3+, FOXP3-, CD20-, CD45RO+/-, CD8-.

The panel definition is parsed and applied by the `panels.py` module, which is extensively covered by unit tests.

### Inference for the whole dataset

You need to write a custom script to obtain the prediction for the whole dataset. For example, you can iterate through slide folders, get the prediction for each tile (`find_cells()` function in the `inference.py`), and concatenate the predictions of all tiles in a slide. Note that to get the correct position of a cell on a whole slide, you need to know the origin of each tile. This information can be extracted either from TIFF files or from metadata generated by the software used to split a scanned file into tiles.

