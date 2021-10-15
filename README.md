# ImmuNet: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping

This repository contains the implementation of the model introduced in the following manuscript:

Shabaz Sultan, Mark A. J. Gorris, Lieke van der Woude, Franka Buytenhuijs, Evgenia Martynova, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor:   
A SegmentationFree Machine Learning Architecture for Immune Landscape Phenotpying in Solid Tumors by Multichannel Imaging.   
biorxiv, 2021, doi: tbc

The proposed model enables automated detection and phenotyping of immune cells in multiplex immunohistochemistry data. ImmuNet is designed to be applied in dense tissue environments such as solid tumors, where segmentation-based phenotyping can be inaccurate due to segmentation errors or overlapping cell boundaries. In the ImmuNet architecture, this problem is addressed by inferring cells' positions and phenotypes directly, without the segmentation map of the input image.

This repository contains the source code of the model and scripts to run training and demo inference. Annotations used for model training are located in `data/annotations_train.json.gz`. A sample of the immunohistochemistry dataset used in the paper will be uploaded to zenodo and we will post the link here.

## System requirements 

The model is implemented in Python 3.6.9 and [Tensorflow 1.14.0](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs). ImmuNet training and demo inference has been tested on Ubuntu 18.04 on our private server and [Google Colab](https://colab.research.google.com/). We advise using GPU to achieve reasonable running time. 

We used NVIDIA GeForce RTX 2080 Ti GPU (RAM: 11 GB, CUDA Version: 11.0) and Intel Core i9-9820X @ 3.30GHz CPU (10 cores, 2 threads per core) to train the network and perform inference. The network dicussed in the manuscript was trained on 231851 environments taken from 36856 cells. Training was
run for 76 epochs, which took 12 hours on our system.

## Installation guide

We strongly recommend creating a virtual to run the code. The dependencies are specified in the `requirements.txt` file and the conda environment file `environment.yml`. To create a conda environment with `.yml` file, run the following command in a shell:
```
conda env create -f environment.yml
```
An environment named `immunet` will be created. Activate it with:
```
conda activate immunet
```
If you do not have a GPU, change `tensorflow-gpu` to `tensorflow` in the requirements before creating the environment.

## Demo

### ImmuNet training

To run the training, please download the sample of the data (will be uploaded to zenodo) and place it in the folder with the source code. Then, run:
 ```
python train.py
```

### Demo inference

Inference of cells' positions and phenotypes is demonstrated for a single immunohistochemistry image. Run inference with the command:
```
python demo-inference.py
```
The visualised network output will be saved in the `demo-output` folder as a set of 6 `.png` files.

- `cell-center-distance-prediction.png` - a map of the predicted distance to the nearest cell center.
- `pseudochannel-{0-4}.png` - maps of the predicted pseudomarkers expression.

The model prediction made based on the network output is saved in `demo-output/prediction.txt`. Each line corresponds to a detected cell and contains its location and phenotype expression in the format:
```
y x pseudomarker0 pseudomarker1 pseudomarker2 pseudomarker3 pseudomarker4
```
Where (y, x) are cell coordinates and pseudomarker{0-4} are expressions of respective pseudomarkers.

To change the image used for inference demonstration, specify a different path in the `tifffile.imread` call. 

A patch of immunohistochemistry image we use for demo inference is rather large, which can cause OOM exception on some GPUs. In our code, this exception is handled by cropping the middle subregion of size 512x512 from the input image and running inference for this subregion. If you still get OOM, please decrease the subregion size by changing `target_size` parameter in the `crop_image_center` call.

