import json
import gzip
from config import *


class Dataset:
    def __init__(self, dict):
        self.id = dict[DATASET_KEY]
        self.panel = dict[PANEL_KEY]
        self.slides = []
        if SLIDES_KEY in dict:
            slide_dicts = dict[SLIDES_KEY]
            for slide_dict in slide_dicts:
                self.add_slide(Slide(slide_dict))

    def add_slide(self, slide):
        slide.dataset_id = self.id
        slide.panel = self.panel

        for tile in slide.tiles:
            tile.dataset_id = self.id
            tile.panel = self.panel

        self.slides.append(slide)

    @property
    def tiles(self):
        _tiles = []
        for slide in self.slides:
            _tiles += slide.tiles

        return _tiles


class Slide:
    def __init__(self, dict):
        self.id = dict[SLIDE_KEY]
        self.dataset_id = None
        self.panel = None
        self.tiles = []

        if TILES_KEY in dict:
            tile_dicts = dict[TILES_KEY]
            for tile_dict in tile_dicts:
                self.add_tile(Tile(tile_dict))

    def add_tile(self, tile):
        tile.slide_id = self.id
        self.tiles.append(tile)


class Tile:
    def __init__(self, dict):
        self.id = dict[TILE_KEY]
        self.dataset_id = None
        self.slide_id = None
        self.panel = None

        self.annotations = []
        if ANNOTATIONS_KEY in dict:
            self.annotations = dict[ANNOTATIONS_KEY]

    @property
    def full_id(self):
        if self.dataset_id is None or self.slide_id is None:
            return None

        return "/".join((self.dataset_id, self.slide_id, self.id))

    def build_path(self, relative_path, file_name="components.tiff"):
        if self.dataset_id is None or self.slide_id is None:
            return None

        return relative_path / self.dataset_id / self.slide_id / self.id / file_name


def load_annotations(annotations_path):
    if annotations_path.suffix == ".gz":
        with gzip.open(annotations_path) as f:
            annotations = json.loads(f.read())
    else:
        with open(annotations_path) as f:
            annotations = json.load(f)

    datasets = [Dataset(dataset_dict) for dataset_dict in annotations]

    tiles = []
    for dataset in datasets:
        tiles += dataset.tiles

    return tiles
