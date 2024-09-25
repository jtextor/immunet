This folder contains the code needed to reproduce some of the results from the paper "ImmuNet: a Segmentation-Free Machine Architecture Pipeline for Lymphocyte Phenotyping".

# Training / validation split 

The script `data_split.py` contains the logic to split total annotations into training and validation. The split is made such that the share of validation data is about 20 %, and the distribution of annotations by type is similar to that in the whole dataset. Also, the validation annotations are evenly distributed by tissue type (dataset). This is achieved by splitting annotations separately for each dataset and merging the results. For every dataset, we first sample validation tiles randomly and, if necessary, add tiles with more annotations of underrepresented types. The number of annotations and their distribution were manually monitored; `data_split.py` contains the final code that gave satisfactory results.

To obtatin the training / validation split used in the paper:

- Download the file with all annotations, `annotations_all.json.gz`, from [zenodo](https://zenodo.org/records/8084976).
- Put it into the folder `data/annotations`.
- Run the shell script `make_data_split.sh` from the `paper` folder.

The files with training and validation annotations will be saved in the `train_val_split` folder. In addition, the script saves the number of training and validation annotations per tissue and annotation type in the `.csv` files in the `train_val_split/stat` folder. To demonstrate that these training and validation annotations are balanced, we make the bar plots for all tissue types, which are saved in the `train_val_split/stat` folder as well.
