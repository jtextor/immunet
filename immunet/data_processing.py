import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from tqdm import tqdm
import tifffile
from csbdeep.utils import normalize
from config import *


def rnd(i):
    return int(round(i))


def background(annotation):
    return BG_KEY in annotation and annotation[BG_KEY]


def map_phenotype(annotation, out_markers_num):
    return (np.array(annotation[PHENO_KEY][:out_markers_num]) - 1) / 4.0


def extract_labels(components, annotations, out_markers_num=5, cell_radius=5):
    h = components.shape[0]
    w = components.shape[1]
    # The first output channel is a prediction of a distance map,
    # the rest are predictions of phenotype marker expression maps
    out_channels_num = 1 + out_markers_num
    out = np.zeros((h, w, out_channels_num), np.float16)
    known_status = np.zeros((h, w), np.uint8)

    # 0 - no cell, i - cell number (just an order in which we got annotations)
    cell_at = np.zeros((h, w), np.uint32)
    cell_ph = np.zeros((len(annotations), out_markers_num), np.float16)
    cell_fg = np.zeros(len(annotations))
    for i, annotation in enumerate(annotations):
        if len(annotation[PHENO_KEY]) < out_markers_num:
            raise ValueError("Positivity of all annotations must be greater or equal to the number of markers to predict")

        x, y = rnd(annotation[X_KEY]), rnd(annotation[Y_KEY])
        if x >= w:
            x = w - 1
        if y >= h:
            y = h - 1
        cell_at[y, x] = i + 1
        cell_ph[i, :] = map_phenotype(annotation, out_markers_num)
        cell_fg[i] = not background(annotation)

    # distance to the nearest cell (will be false in cell_at == 0 matrix) and indices
    # cell_at == 0 means >=1 - background, 0 - foreground
    # od - euclidean distance to the cell center (foreground points)
    # oi - indices of the closest cell center
    od, oi = distance_transform_edt(cell_at == 0, return_indices=True)
    in_cell = od <= cell_radius

    # Sets 255 for all circles that represent cells, 0 otherwise
    known_status[in_cell] = 255

    # for each x, y get cell index in order of cell_at (how we got annotations)
    which_cell = cell_at[oi[0][in_cell], oi[1][in_cell]] - 1
    # Distances channel
    # for places inside cells set "cell_radius - distance to cell" so that max value were at the center of the cell
    # and -2 for background
    out[in_cell, 0] = np.where(cell_fg[which_cell], cell_radius - od[in_cell], -2)
    # Phenotypes channel - same ph value for all points inside the cell
    out[in_cell, 1:] = cell_ph[which_cell, :]
    # places without annotations have unknown status
    out[known_status == 0, 0] = -1

    return out


def make_samples(
    tile_annotations,
    image_path,
    window,
    max_examples_per_tile,
    in_channels_num,
    out_markers_num=5,
    cell_radius=5
):

    patches = []
    y_dist = []
    y_phenotypes = []

    for tile in tqdm(tile_annotations):
        if len(tile.annotations) == 0:
            continue

        tile_path = tile.build_path(image_path)

        if not tile_path.is_file():
            print("\t cannot find cached TIFF for {} ...".format(str(tile_path)))
            continue
        try:
            components = tifffile.imread(tile_path, key=range(0, in_channels_num))
        except:
            print("\t cannot read TIFF for {}".format(str(tile_path)))
            continue

        components = np.moveaxis(normalize(components), 0, -1)
        h, w = components.shape[0], components.shape[1]

        label_maps = extract_labels(components, tile.annotations, out_markers_num, cell_radius)
        known_status = label_maps[:, :, 0] != -1

        # sharpen cell edges a little more
        label_maps[label_maps[:, :, 0] == 0] = -1

        # sample training patches
        coordinates = np.transpose(np.nonzero(known_status))
        np.random.shuffle(coordinates)
        coordinates = coordinates[0:int(round(coordinates.shape[0] / 8.0))]
        if max_examples_per_tile < coordinates.shape[0]:
            coordinates = coordinates[0:max_examples_per_tile]
        for i, j in coordinates:
            if i <= window or j <= window or i > h - window - 1 or j > w - window - 1:
                continue
            y_dist.append(np.ndarray.flatten(label_maps[i - 1:i + 2, j - 1:j + 2, 0]))
            y_phenotypes.append(
                np.ndarray.flatten(label_maps[i - 1:i + 2, j - 1:j + 2, 1:])
            )
            patches.append(
                components[i - window:i + window + 1, j - window:j + window + 1, :]
            )

    patches = np.stack(patches)
    labels = np.concatenate([np.array(y_dist), np.stack(y_phenotypes)], axis=1)

    return patches, labels
