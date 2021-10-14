import os
# TODO: change
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from tqdm import tqdm
import tifffile
from pathlib import Path
from csbdeep.utils import normalize

# TODO: check which warnings we get
import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import extract_labels, rnd
from models import model_for_training
from annotations import load_tile_annotations

epochs = 100

window = 31

cell_radius = 5
max_examples_per_tile = 1000

datasets_to_include = [
        "2020-01-27-phenotyping-paper-cytoagars",
        "2020-01-27-phenotyping-paper-tonsils",
        "2020-01-31-phenotyping-paper-bladder",
        "2020-01-31-phenotyping-paper-melanoma",
        "2020-01-31-phenotyping-paper-prostate",
        "2020-02-12-phenotyping-paper-lung-bcell"]

tile_cache_path = Path("tilecache")

# Random seed is for extracting pixels from the training data
np.random.seed( 123 )

# TODO: extract data loading, making training data and training into separate functions
tile_annotations = load_tile_annotations("training")

# Filter out datasets
if datasets_to_include is not None:
    tile_annotations = [tile_annotation for tile_annotation in tile_annotations
                        if tile_annotation.dataset_id in datasets_to_include]

# Number of annotations
n_training_cells = 0
for tile in tile_annotations:
    n_training_cells += len(tile.annotations)

print("Intent to train on {} cells".format(n_training_cells))

examples = []
y_dist = []
y_category_phenotypes = []

print("Making training data ...")

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

print("training on {} examples".format(len(examples)))

Mt = model_for_training(input_shape=(2*window+1,2*window+1,6))

Mt.compile(optimizer='adam',
            loss= ['mean_squared_error','mean_squared_error'],
            loss_weights = [1,20])

examples = np.stack(examples)
y_dist = np.array(y_dist)
y_category_phenotypes = np.stack(y_category_phenotypes)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    rotation_range=90)

y_total = np.concatenate([y_dist, y_category_phenotypes], axis=1)

def generate_data_generator(generator, X, Y1, batchsize):
    i = generator.flow(X, Y1,  seed=7, batch_size=batchsize)

    while True:
        img, y = i.next()
        for j in range(img.shape[0]):
            for k in range(img[j].shape[2]):
                img[j,:,:,k] = img[j,:,:,k]*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        yield img, [y[:,0:9], y[:,9:]]

checkpoint = ModelCheckpoint("current_best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

# Include into if statement
#Mt.load_weights("best_model.hdf5")

Mt.fit(
    generate_data_generator(datagen, examples, y_total, 64),
    steps_per_epoch=len(examples)//64,
    epochs=epochs,
    callbacks=[checkpoint])
