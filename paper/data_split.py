import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../immunet")
import json
import numpy as np
import random
from pathlib import Path
from collections import Counter
from panels import load_panels
from config import PANEL_FILE
import matplotlib.pyplot as plt
from annotations import Dataset
import pandas as pd
from enum import Enum, unique
import gzip

@unique
class Stage(str, Enum):
    bladder = "bladder"
    lung = "lung"
    melanoma = "melanoma"
    prostate = "prostate"
    tonsils = "tonsils"
    makeSplit = "makeSplit"
    saveStat = "saveStat"

# Constants
BLADDER_DS = "2020-01-31-phenotyping-paper-bladder"
CYTOAGARS_DS = "2020-01-27-phenotyping-paper-cytoagars"
LUNG_DS = "2020-02-12-phenotyping-paper-lung-bcell"
MELANOMA_DS = "2020-01-31-phenotyping-paper-melanoma"
PROSTATE_DS = "2020-01-31-phenotyping-paper-prostate"
TONSILS_DS = "2020-01-27-phenotyping-paper-tonsils"

BLADDER_TS = "bladder"
CYTOAGARS_TS = "cytoagars"
LUNG_TS = "lung"
MELANOMA_TS = "melanoma"
PROSTATE_TS = "prostate"
TONSILS_TS = "tonsils"

TISSUE_CSV_KEY = "tissue"
CELL_TYPE_CSV_KEY = "ct"
TRAINING_CSV_KEY = "train"
VALIDATION_CSV_KEY = "val"

DS_TS = {BLADDER_DS: BLADDER_TS,
         CYTOAGARS_DS: CYTOAGARS_TS,
         LUNG_DS: LUNG_TS,
         MELANOMA_DS: MELANOMA_TS,
         PROSTATE_DS: PROSTATE_TS,
         TONSILS_DS: TONSILS_TS}

# Paths
data_path = Path("data")
annotations_folder = data_path / "annotations"
annotations_file = annotations_folder / "annotations_all.json.gz"
split_path = data_path / "train_val_split"
split_stat_path = split_path / "stat"
split_stat_path.mkdir(exist_ok=True, parents=True)

# Panels
panels = load_panels(Path("../") / PANEL_FILE)


def load_dataset(annotations_path, ds_name):

    with gzip.open(annotations_path, 'rt', encoding='UTF-8') as f:
        annotations = json.loads(f.read())
        dataset = [Dataset(ds) for ds in annotations if ds["ds"] == ds_name][0]

    return dataset


def bar_plot(cell_type, num_val, num_train, title, target_path):
    """ Plots distribution of annotations
        useful for summary and controlling validation and training split
    """

    plt.figure()
    ax = plt.gca()
    plt.title(title)

    x_axis = np.arange(len(cell_type))
    ax.bar(x_axis - 0.2, num_val, 0.4, label='Val')
    ax.bar(x_axis + 0.2, num_train, 0.4, label='Train')

    for i in x_axis:
        ax.text(i - 0.45, num_val[i] + 10, str(num_val[i]))
        ax.text(i, num_train[i] + 10, str(num_train[i]))

    plt.xticks(x_axis,cell_type, rotation=90)
    plt.ylabel("Number of annotations")
    plt.legend()
    # Custom the subplot layout
    plt.subplots_adjust(bottom=0.3)

    plt.savefig(target_path)


def tiles_info(dataset, tile_ids_exclude=None):
    """ Returns an array of dictionaries with basic tile info (id, slide, dataset)
        and number of annotations of cells of different types on a tile
    """
    tiles = []
    panel = panels[dataset.panel]
    for tile in dataset.tiles:
        if tile_ids_exclude is None or tile.full_id not in tile_ids_exclude:
            tile_info = {
                "ds": tile.dataset_id,
                "slide": tile.slide_id,
                "tile": tile.id,
            }

            for name in panel.main_types:
                tile_info[name] = 0

            for annotation_d in tile.annotations:
                tile_info[annotation_d["type"]] += 1

            tile_info["total"] = len(tile.annotations)
            tiles.append(tile_info)

    return tiles


def sample_tiles(tiles, val_tiles, tiles_num, panel):
    """ Samples a specified number of tiles and returns the statistics of how many cells of different types
        has been sampled. tiles and val_tiles parameters are changed by the method
    """

    added_cells = Counter()
    for i in range(tiles_num):
        tile = np.random.choice(tiles)
        val_tiles.append(tile)
        for name in panel.main_types:
            if name in tile:
                added_cells[name] += tile[name]

        tiles.remove(tile)

    return added_cells


def extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, tiles_num, panel):
    """ Adds a specified number of tiles to validation tiles and updates statistics of how many cells
        of different types has been sampled. val_tiles and added_cells parameters are changed by the method
    """

    val_no_cell_tiles = filtered_tiles[:tiles_num]
    val_tiles.extend(val_no_cell_tiles)

    for tile in val_no_cell_tiles:
        for name in panel.main_types:
            added_cells[name] += tile[name]


def save_train_val_split(dataset, val_tiles, tissue, output_dir):
    """ Saves json files with training and validation tile ids
    """

    val_ids = ["/".join((t["ds"], t["slide"], t["tile"])) for t in val_tiles]
    val_ids_file = output_dir / "{}_val_tile_ids.json".format(tissue)
    with open(val_ids_file, "w") as f:
        json.dump(val_ids, f)

    all_tiles = tiles_info(dataset)
    all_ids = ["/".join((t["ds"], t["slide"], t["tile"])) for t in all_tiles]
    train_ids = list(set(all_ids).difference(val_ids))
    train_ids_file = output_dir / "{}_train_tile_ids.json".format(tissue)
    with open(train_ids_file, "w") as f:
        json.dump(train_ids, f)


def split_bladder(annotations_path, output_dir=split_path):
    tissue = BLADDER_TS
    dataset = load_dataset(annotations_path, BLADDER_DS)
    panel = panels[dataset.panel]
    tiles = tiles_info(dataset)

    val_tiles = []
    # Sample some tiles randomly
    added_cells = sample_tiles(tiles, val_tiles, 9, panel)

    # Sample enough Other cells but restrict number of T cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if tile["Other cell"] >= 5 and tile["T cell"] < 2]
    random.shuffle(filtered_tiles)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 5, panel)

    # Second round, sample more B cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if 5 <= tile["B cell"] < 50
    ]
    random.shuffle(filtered_tiles)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 5, panel)
    save_train_val_split(dataset, val_tiles, tissue, output_dir)


def split_lung(annotations_path, output_dir=split_path):
    tissue = LUNG_TS
    dataset = load_dataset(annotations_path, LUNG_DS)
    panel = panels[dataset.panel]
    tiles = tiles_info(dataset)

    val_tiles = []
    # Sample some tiles randomly
    added_cells = sample_tiles(tiles, val_tiles, 9, panel)

    # Sample enough No and Other cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if (tile["No cell"] >= 5 or tile["Other cell"] >= 2) and tile["Other cell"] < 10
    ]
    # Include as few tumor cells as possible
    filtered_tiles = sorted(filtered_tiles, key=lambda x: x["Tumor cell"], reverse=False)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 13, panel)

    # Step 2 sample enough B and T cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if tile["B cell"] >= 5 and tile["T cell"] >= 2
    ]
    # Include as few tumor cells as possible
    filtered_tiles = sorted(filtered_tiles, key=lambda x: x["Tumor cell"], reverse=False)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 5, panel)
    save_train_val_split(dataset, val_tiles, tissue, output_dir)


def split_melanoma(annotations_path, output_dir=split_path):
    tissue = MELANOMA_TS
    dataset = load_dataset(annotations_path, MELANOMA_DS)
    panel = panels[dataset.panel]
    tiles = tiles_info(dataset)

    val_tiles = []
    # Sample some tiles randomly
    added_cells = sample_tiles(tiles, val_tiles, 23, panel)

    # Sample enough B and No cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if (tile["B cell"] >= 5 or tile["No cell"] >= 2) and tile["No cell"] < 11
    ]
    # Include as few T cells as possible
    filtered_tiles = sorted(filtered_tiles, key=lambda x: x["T cell"], reverse=False)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 6, panel)
    save_train_val_split(dataset, val_tiles, tissue, output_dir)


def split_prostate(annotations_path, output_dir=split_path):
    tissue = PROSTATE_TS
    dataset = load_dataset(annotations_path, PROSTATE_DS)
    panel = panels[dataset.panel]
    tiles = tiles_info(dataset)

    val_tiles = []
    # Sample some tiles randomly
    sample_tiles(tiles, val_tiles, 10, panel)
    save_train_val_split(dataset, val_tiles, tissue, output_dir)


def split_tonsils(annotations_path, output_dir=split_path):
    tissue = TONSILS_TS
    dataset = load_dataset(annotations_path, TONSILS_DS)
    panel = panels[dataset.panel]
    tiles = tiles_info(dataset)

    val_tiles = []
    # Sample some tiles randomly
    added_cells = sample_tiles(tiles, val_tiles, 16, panel)

    # Sample enough No, Other and Tumor cells
    added_tiles = ["/".join((tile["ds"], tile["slide"], tile["tile"])) for tile in val_tiles]

    tiles = tiles_info(dataset, tile_ids_exclude=added_tiles)
    filtered_tiles = [
        tile for tile in tiles if tile["No cell"] >= 5 or tile["Other cell"] >= 5 or tile["Tumor cell"] >= 5
    ]
    # Include as few T and B cells as possible
    filtered_tiles = sorted(filtered_tiles, key=lambda x: x["T cell"] + x["B cell"], reverse=False)

    extend_sampled_tiles(filtered_tiles, val_tiles, added_cells, 7, panel)
    save_train_val_split(dataset, val_tiles, tissue, output_dir)


def merge_split_by_tissue(split_dir=split_path):
    """ Merges files with training and validation tile ids made for different tissues
    """
    tissues = [BLADDER_TS, LUNG_TS, MELANOMA_TS, PROSTATE_TS, TONSILS_TS]

    # For some reason a few tiles have been lost in new inForm analysis
    # For fair comparison between inForm and ImmuNet we need to remove these tiles from validation set
    # There are only 33 annotations on these tiles
    val_tiles_to_exclude = ["2020-01-31-phenotyping-paper-melanoma/T14-18776-I1_Scan1/56327,7224",
                            "2020-01-31-phenotyping-paper-prostate/T16-03444-I1_Scan1/53816,8453"]

    val_tile_ids = []
    train_tile_ids = []

    for tissue in tissues:
        val_tiles_file = "{}_val_tile_ids.json".format(tissue)
        with open(split_dir / val_tiles_file) as f:
            tissue_val_tile_ids = json.load(f)
            tissue_val_tile_ids = [tile_id for tile_id in tissue_val_tile_ids if tile_id not in val_tiles_to_exclude]
            val_tile_ids.extend(tissue_val_tile_ids)

        train_tiles_file = "{}_train_tile_ids.json".format(tissue)
        with open(split_dir / train_tiles_file) as f:
            tissue_train_tile_ids = json.load(f)
            train_tile_ids.extend(tissue_train_tile_ids)

    with open(split_dir / "val_tile_ids.json", "w") as f:
        json.dump(val_tile_ids, f)

    with open(split_dir / "train_tile_ids.json", "w") as f:
        json.dump(train_tile_ids, f)


def extract_annotations_on_tiles(
    annotations_path, tile_ids_path, out_file_path, datasets_to_skip=None
):
    """ Splits a file with all annotations into files with training and validation annotations
        using files with training and validation tile ids
    """
    with open(tile_ids_path) as f:
        tile_ids = json.load(f)

    with gzip.open(annotations_path) as f:
        datasets = json.loads(f.read())

    datasets_filtered = []
    for dataset in datasets:
        if datasets_to_skip is not None and dataset["ds"] in datasets_to_skip:
            datasets_filtered.append(dataset)
            continue

        slides_filtered = []
        for slide in dataset["slides"]:
            tiles = slide["tiles"]
            tiles_filtered = [tile for tile in tiles if "/".join((dataset["ds"], slide["slide"], tile["tile"])) in tile_ids]
            if len(tiles_filtered) > 0:
                slide["tiles"] = tiles_filtered
                slides_filtered.append(slide)

        if len(slides_filtered) > 0:
            dataset["slides"] = slides_filtered
            datasets_filtered.append(dataset)

    with open(out_file_path, "w") as f:
        json.dump(datasets_filtered, f)


def extract_annotations_statistics(annotations_path, detailed=False):
    """ Extracts number of annotations of cells of different phenotypes
        for each dataset in annotations file
    """

    with open(annotations_path) as f:
        annotations = json.load(f)

    datasets = [Dataset(dataset_dict) for dataset_dict in annotations]
    ds_count = {}
    for dataset in datasets:
        cells = Counter()
        panel = panels[dataset.panel]
        for tile in dataset.tiles:
            for ann in tile.annotations:
                if detailed:
                    ann_cell_type = panel.annotation_phenotype(ann["type"], ann["positivity"])
                    cells[ann_cell_type.detailed] += 1
                else:
                    cells[ann["type"]] += 1

        ds_count[DS_TS[dataset.id]] = cells

    return ds_count


def make_ann_stat_csv(stat_train, stat_val, output_file):
    """ Combines statistics of number of annotations of cells of different phenotypes
        in training and validation data and saves it into a single csv file
    """
    data = {TISSUE_CSV_KEY: [], CELL_TYPE_CSV_KEY: [], TRAINING_CSV_KEY: [], VALIDATION_CSV_KEY: []}
    tissues = set(stat_train.keys()).union(set(stat_val.keys()))

    for tissue in tissues:
        tissue_train_stat = stat_train[tissue]
        tissue_val_stat = stat_val[tissue] if tissue in stat_val else None
        main_types = list(tissue_train_stat.keys())
        if "Invalid" in main_types:
            main_types.remove("Invalid")

        for cell_type in main_types:
            data[TISSUE_CSV_KEY].append(tissue)
            data[CELL_TYPE_CSV_KEY].append(cell_type)
            data[TRAINING_CSV_KEY].append(tissue_train_stat[cell_type])
            val_num = tissue_val_stat[cell_type] if tissue_val_stat is not None else 0
            data[VALIDATION_CSV_KEY].append(val_num)

    annotations_df = pd.DataFrame(data)
    annotations_df.to_csv(output_file, index=False)


def save_bar_plots(stat_train, stat_val, split_stat_path):
    """ Visualises training / validation split by tissue
    """

    tissues = set(stat_train.keys()).intersection(set(stat_val.keys()))
    for tissue in tissues:
        tissue_train_stat = stat_train[tissue]
        tissue_val_stat = stat_val[tissue]

        cell_types = sorted(set(tissue_train_stat.keys()).union(set(tissue_val_stat.keys())))
        if "Invalid" in cell_types:
            cell_types.remove("Invalid")

        nums_train = []
        nums_val = []
        for cell_type in cell_types:
            num_train = tissue_train_stat[cell_type] if cell_type in tissue_train_stat else 0
            nums_train.append(num_train)

            num_val = tissue_val_stat[cell_type] if cell_type in tissue_train_stat else 0
            nums_val.append(num_val)

        bar_plot(cell_types, nums_val, nums_train, "Annotation split {}".format(tissue), split_stat_path / "{}.png".format(tissue))


def save_ann_stat(train_ann_path, val_ann_path, output_dir=split_stat_path):
    """ Makes statistics of annotations number of cells of different phenotypes for
        main phenotypes and detailed phenotypes (T cell subtypes)
    """
    stat_train = extract_annotations_statistics(train_ann_path, detailed=False)
    stat_val = extract_annotations_statistics(val_ann_path, detailed=False)

    make_ann_stat_csv(stat_train, stat_val, output_dir / "train_val_split.csv")

    stat_train_subtypes = extract_annotations_statistics(train_ann_path, detailed=True)
    stat_val_subtypes = extract_annotations_statistics(val_ann_path, detailed=True)

    make_ann_stat_csv(stat_train_subtypes, stat_val_subtypes, output_dir / "train_val_split_subtypes.csv")
    save_bar_plots(stat_train_subtypes, stat_val_subtypes, split_stat_path)


def clean_up_temp_files(split_path):
    temp_files_glob = split_path.glob("*tile_ids.json")

    for temp_file in temp_files_glob:
        temp_file.unlink()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        required=True,
        help="Document file path containing cell ids, separated by commas.",
    )

    args = parser.parse_args()
    stage = Stage(args.stage)
    
    seed = 99
    random.seed(seed)
    np.random.seed(seed)

    if stage == Stage.bladder:
        split_bladder(annotations_file)
    elif stage == Stage.lung:
        split_lung(annotations_file)
    elif stage == Stage.melanoma:
        split_melanoma(annotations_file)
    elif stage == Stage.prostate:
        split_prostate(annotations_file)
    elif stage == Stage.tonsils:
        split_tonsils(annotations_file)
    elif stage == Stage.makeSplit:
        merge_split_by_tissue(split_path)

        # Make final training / validation annotation split
        val_tile_ids = split_path / "val_tile_ids.json"
        val_annotations_path = split_path / "annotations_val.json"

        extract_annotations_on_tiles(annotations_file, val_tile_ids, val_annotations_path)

        train_tile_ids = split_path / "train_tile_ids.json"
        train_annotations_path = split_path / "annotations_train.json"

        extract_annotations_on_tiles(annotations_file, train_tile_ids, train_annotations_path,
                                     CYTOAGARS_DS)

        clean_up_temp_files(split_path)
    elif stage == Stage.saveStat:
        train_ann_path = split_path / "annotations_train.json"
        val_ann_path = split_path / "annotations_val.json"
        save_ann_stat(train_ann_path, val_ann_path)
