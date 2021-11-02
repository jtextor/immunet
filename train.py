import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import argparse
import numpy as np
from tqdm import tqdm
import tifffile
from pathlib import Path
from csbdeep.utils import normalize

import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import extract_labels, rnd
from models import model_for_training
from annotations import load_tile_annotations


def make_training_data(tile_annotations,
                       window,
                       tile_cache_path,
                       cell_radius=5, 
                       max_examples_per_tile=1000):
    examples = []
    y_dist = []
    y_category_phenotypes = []
    
    for tile in tqdm(tile_annotations):
        if len(tile.annotations) == 0:
            continue
    
        tile_path = tile.build_path(tile_cache_path)
    
        if not tile_path.is_file():
            print("\t cannot find cached TIFF for {} ...".format(str(tile_path)))
            continue
        try:
            components = tifffile.imread(tile_path, key=range(0, 6))
        except:
            print("\t cannot read TIFF for {}".format(str(tile_path)))
            continue
    
        components = np.moveaxis(normalize(components), 0, -1)
        h, w = components.shape[0], components.shape[1]
    
        labels = extract_labels(components, tile.annotations, cell_radius)
        known_status = labels[:, :, 0] != -1
    
        # sharpen cell edges a little more
        labels[labels[:, :, 0] == 0] = -1
    
        coordinates = np.transpose(np.nonzero(known_status))
        np.random.shuffle(coordinates)
        coordinates = coordinates[0:rnd(coordinates.shape[0] / 8.)]
        if max_examples_per_tile < coordinates.shape[0]:
            coordinates = coordinates[0:max_examples_per_tile]
        for i, j in coordinates:
            if i <= window or j <= window or i > h - window - 1 or j > w - window - 1:
                continue
    
            y_dist.append(np.ndarray.flatten(labels[i - 1:i + 2, j - 1:j + 2, 0]))
            y_category_phenotypes.append(np.ndarray.flatten(labels[i - 1:i + 2, j - 1:j + 2, 1:]))
            examples.append(components[i - window:i + window + 1, j - window:j + window + 1, :])
    
    examples = np.stack(examples)
    y_dist = np.array(y_dist)
    y_category_phenotypes = np.stack(y_category_phenotypes)
    y_total = np.concatenate([y_dist, y_category_phenotypes], axis=1)
    
    return examples, y_total


def generate_data_generator(generator, X, Y, batch_size):
    i = generator.flow(X, Y, seed=7, batch_size=batch_size)

    while True:
        img, y = i.next()
        for j in range(img.shape[0]):
            for k in range(img[j].shape[2]):
                img[j, :, :, k] = img[j, :, :, k] * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        yield img, [y[:, 0:9], y[:, 9:]]


def run_training(examples, y_total, window, epochs=100, batch_size=64):
    Mt = model_for_training(input_shape=(2 * window + 1, 2 * window + 1, 6))
    
    Mt.compile(optimizer="adam",
               loss=["mean_squared_error", "mean_squared_error"],
               loss_weights=[1, 20])
    
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode="reflect",
                                 rotation_range=90)
    
    checkpoint = ModelCheckpoint("current_best_model.hdf5", monitor="loss", verbose=1,
                                 save_best_only=True, mode="auto")
    
    Mt.fit(generate_data_generator(datagen, examples, y_total, batch_size),
           steps_per_epoch=len(examples) // batch_size,
           epochs=epochs,
           callbacks=[checkpoint])


if __name__ == "__main__":
    # Random seed is for extracting pixels from the training data
    np.random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="tilecache", required=False,
                        help="a path to a folder that contains an immunohistochemistry data sample")
    parser.add_argument('--annotations_path', type=str, default="annotations/annotations_train.json.gz", required=False,
                        help="a path to a file that contains annotations")
    parser.add_argument('--epochs', type=int, default=100, required=False,
                        help="a number of epochs to run training")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    annotations_path = Path(args.annotations_path)
    epochs = args.epochs

    datasets_to_include = ["2020-01-27-phenotyping-paper-cytoagars"]

    tile_annotations = load_tile_annotations(annotations_path)

    # Filter out datasets
    if datasets_to_include is not None:
        tile_annotations = [tile_annotation for tile_annotation in tile_annotations
                            if tile_annotation.dataset_id in datasets_to_include]

    # Number of annotations
    n_training_cells = 0
    for tile in tile_annotations:
        n_training_cells += len(tile.annotations)

    print("Intent to train on {} cells".format(n_training_cells))

    print("Making training data ...")
    window = 31

    X, Y = make_training_data(tile_annotations, window, data_path)
    print("training on {} examples".format(len(X)))

    run_training(X, Y, window, epochs)

    