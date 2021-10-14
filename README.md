# NoSePh: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping

This repository contains the implementation of the model introduced the following manuscript:

A SegmentationFree Machine Learning Architecture for Immune Landscape Phenotpying in Solid Tumors by Multichannel Imaging. Shabaz Sultan, Mark A. J. Gorris, Lieke van der Woude, Franka Buytenhuijs, Evgenia Martynova, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor TODO; doi: TODO

The proposed model enables automated detection and phenotyping of immune cells in multiplex immunohistochemistry data. NoSePh is designed to be applied in dense tissue environments such as solid tumors, where segmentation-based phenotyping can be inaccurate due to segmentation errors or overlapping cell boundaries. NoSePh identifies cells' positions and phenotypes directly, without the segmentation map of the input image. 

This repository contains the source code of the model and scripts to run training and demo inference. Annotations used for model training are located in `data/annotations_train.json.gz`. A sample of the immunohistochemistry dataset unsed in the paper can be downloaded [here](TODO). 

## System requirements 

The model is implemented in Python 3.6.9 and Tensorflow 1.14.0. TODO

## Installation guide

We recommend to create a virtual to run the code. The dependencies are specified in the `requirements.txt` file and in the conda environment file `environment.yml`. To create a conda environment with `.yml` file, run the following command in shell:
```
conda env create -f environment.yml
```
An enviroment named `noseph` will be created. Activate it with:
```
conda activate noseph
```

## Demo

### NoSePh training

To run the training, please download [the sample of the data](TODO) and place it in the folder with the source code. Then, run:
 ```
python train.py
```

### Demo inference

Inference of cells' positions and phenotypes is demonstrated for one immunohistochemistry image in the dataset. Run inference with the command:
```
python demo-inference.py
```
The visualised prediction will be saved in the `demo-output` folder as a set of 6 `.png` files.

- `cell-center-distance-prediction.png` - a map of the predicted distance to the nearest cell center.
- `pseudochannel-{0-4}.png` - maps of the predicted pseudomarkers expression.

The image used for inference demonstration can be changed by changing the path in the `tifffile.imread` call. 

A singe immunohistochemistry image is rather large, which can cause OOM exception during inference on some GPUs. In our code this exception is handled by cropping the middle subregion of size 512x512 from the input image and running inference for this subregion. If you still get OOM, please decrease the subregion size by changing `target_size` parameter in the `crop_image_center` call.

