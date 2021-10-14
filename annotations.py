import json
import gzip
from pathlib import Path
from enum import Enum, unique

DATA_PATH = Path("data")
ANNOT_TRAIN_FILE = "annotations_train.json.gz"
ANNOT_VAL_FILE = "annotations_val.json.gz"

# JSON KEYS
DATASET_ID_JSON_KEY = "ds_id"
DATASET_TYPE_JSON_KEY = "type"
SLIDE_ID_JSON_KEY = "slide_id"
TILE_ID_JSON_KEY = "tile_id"
ANNOT_TYPE_JSON_KEY = "type"
ANNOT_X_JSON_KEY = "x"
ANNOT_Y_JSON_KEY = "y"
ANNOT_POSITIVITY_JSON_KEY = "positivity"
SLIDES_JSON_KEY = "slides"
TILES_JSON_KEY = "tiles"
ANNOTATIONS_JSON_KEY = "annotations"

@unique
class AnnotationType(str, Enum):
    training = "training"
    validation = "validation"


class Dataset:

    def __init__(self, dict):
        self.id = dict[DATASET_ID_JSON_KEY]
        self.type = dict[DATASET_TYPE_JSON_KEY] if DATASET_TYPE_JSON_KEY in dict else None
        self.slides = []
        if SLIDES_JSON_KEY in dict:
            slide_dicts = dict[SLIDES_JSON_KEY]
            for slide_dict in slide_dicts:
                self.add_slide(Slide(slide_dict))

    def add_slide(self, slide):
        slide.dataset_id = self.id

        for tile in slide.tiles:
            tile.dataset_id = self.id

        self.slides.append(slide)

    @property
    def tiles(self):
        _tiles = []
        for slide in self.slides:
            _tiles += slide.tiles

        return _tiles


class Slide:

    def __init__(self, dict):
        self.id = dict[SLIDE_ID_JSON_KEY]
        self.dataset_id = None
        self.tiles = []

        if TILES_JSON_KEY in dict:
            tile_dicts = dict[TILES_JSON_KEY]
            for tile_dict in tile_dicts:
                self.add_tile(Tile(tile_dict))

    def add_tile(self, tile):
        tile.slice_id = self.id
        self.tiles.append(tile)


class Tile:

    def __init__(self, dict):
        self.id = dict[TILE_ID_JSON_KEY]
        self.dataset_id = None
        self.slice_id = None

        self.annotations = []
        if ANNOTATIONS_JSON_KEY in dict:
            self.annotations = dict[ANNOTATIONS_JSON_KEY]

    def build_path(self, relative_path, file_name="components.tiff"):
        if self.dataset_id is None or self.slice_id is None:
            return None

        return relative_path / self.dataset_id / self.slice_id / self.id / file_name


def load_tile_annotations(type=AnnotationType.training):
    if type not in [AnnotationType.training, AnnotationType.validation]:
        raise ValueError("Unknown annotation type: "+str(type))
    annotations_path = Path("data") / (ANNOT_TRAIN_FILE if type == AnnotationType.training else ANNOT_VAL_FILE)

    with gzip.open(annotations_path) as f:
        annotations = json.loads(f.read())

    datasets = [Dataset(dataset_dict) for dataset_dict in annotations]

    tiles = []
    for dataset in datasets:
        tiles += dataset.tiles

    return tiles
