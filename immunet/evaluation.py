import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.spatial import cKDTree
from annotations import load_annotations
from panels import panels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from inference import find_cells
from models import model_for_inference
from config import *


def match_cells(annotations_path,
                model_path,
                log_threshold,
                suffix=None,
                images_path=Path("../tilecache"),
                output_path=Path("../data/prediction")):
    # Matches annotations with predictions and saves the results to a file
    output_path.mkdir(exist_ok=True)
    model = model_for_inference(weights=str(model_path))
    out_markers_num = model.layers[-1].input_shape[-1]
    tiles = load_annotations(annotations_path)

    # for each tile get predictions and compare with annotations as below
    if suffix is None:
        prediction_path = output_path / "prediction.tsv"
    else:
        prediction_path = output_path / "prediction-{}.tsv".format(suffix)

    with open(prediction_path, "w+") as f:
        # Write TSV columns
        columns = [PANEL_COL, DATASET_COL, SLIDE_COL, TILE_COL, ID_COL, ANN_TYPE_COL, ANN_PHENO_COL, PRED_PHENO_COL, DISTANCE_COL]
        f.write("\t".join(columns))
        f.write("\n")

        # Write predictions matched to annotations
        for tile in tqdm(tiles):
            tile_path = tile.build_path(images_path)
            # get prediction
            prediction = find_cells(tile_path, model, log_threshold)

            # If no cells is predicted
            if len(prediction) == 0:
                prediction = [[[-100000, -100000], [0] * out_markers_num]]

            tr = cKDTree(np.array([cell[0] for cell in prediction]))
            panel = tile.panel
            dataset = tile.dataset_id
            slide = tile.slide_id
            tile_id = tile.id
            annotations = tile.annotations
            for annotation in annotations:
                f.write("\t".join((panel, dataset, slide, tile_id, annotation[ID_KEY], annotation[TYPE_KEY])))
                f.write("\t")
                positivity = annotation[PHENO_KEY][:out_markers_num]
                f.write(",".join(["%g" % i for i in positivity]))
                f.write("\t")
                nn = tr.query([annotation[Y_KEY], annotation[X_KEY]])
                f.write(",".join(["%.2f" % i for i in prediction[nn[1]][1]]))
                f.write("\t%.1f\n" % (nn[0]))
                f.flush()


def match(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        default="../data/annotations/annotations_val.json.gz",
        required=False,
        help="a path to annotations file"
    )
    parser.add_argument(
        "--model_path",
        default="../train_output/immunet.h5",
        type=str,
        required=False,
        help="a path to a model to use for prediction",
    )
    parser.add_argument(
        "--th",
        type=float,
        required=False,
        default=0.07,
        help="a threshold for a blob detection algorithm",
    )
    parser.add_argument(
        "--s",
        type=str,
        required=False,
        help="a suffix for a file name to save the results",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="../tilecache",
        required=False,
        help="a path to a folder with images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/prediction",
        required=False,
        help="a path to save evaluation results",
    )

    args = parser.parse_args(argv)
    annotations_path = Path(args.ann_path)
    model_path = Path(args.model_path)
    threshold = args.th
    suffix = args.s
    images_path = Path(args.images_path)
    output_path = Path(args.output_path)

    match_cells(annotations_path, model_path, threshold, suffix, images_path, output_path)


def plot_confusion_matrix(y_true, y_pred, output_path, level="main_types", label=None):
    if label is not None:
        file_name = "CM_{}_{}.png".format(level, label)
    else:
        file_name = "CM_{}.png".format(level)

    plt.rcParams.update({"font.size": 16})

    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(
        output_path / file_name,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def make_performance_csv(eval_df, type_level_column, hit_column, output_path, label=None):
    performance_group = eval_df.groupby([type_level_column])

    cases = performance_group[hit_column].count()
    errors = cases - performance_group[hit_column].sum()
    err_rates = 1 - performance_group[hit_column].mean()

    errors_df = pd.concat([errors.to_frame(), cases.to_frame(), err_rates.to_frame()], axis=1)
    errors_df.columns = [ERROR_NUM_COL, CASES_NUM_COL, ERROR_RATE_COL]
    if label is not None:
        errors_df[LABEL_COL] = label

    errors_df = errors_df.reset_index()
    errors_df.rename(columns={type_level_column: CELL_TYPE_COL}, inplace=True)

    for _, row in errors_df.iterrows():
        print("{} {} {}".format(row[CELL_TYPE_COL], row[ERROR_RATE_COL], row[CASES_NUM_COL]))

    if label is not None:
        file_name = "{}_perf_{}.csv".format(type_level_column, label)
    else:
        file_name = "{}_perf.csv".format(type_level_column)

    errors_df.to_csv(output_path / file_name, index=False)


def calculate_performance(
    eval_df, output_path, label=None, verbose=True
):
    if verbose:
        print("Main cell types performance:")

    make_performance_csv(eval_df, ANN_TYPE_COL, HIT_MAIN_COL, output_path, label=label)

    if verbose:
        print("\nCell subtypes performance:")

    make_performance_csv(eval_df, ANN_SUBTYPE_COL, HIT_SUBTYPE_COL, output_path, label=label)


def extract_error_dict(eval_df):
    errors = []
    errors_df = eval_df[eval_df[HIT_SUBTYPE_COL] == 0]
    for _, row in errors_df.iterrows():
        error_dict = {
            PANEL_COL: row[PANEL_COL],
            DATASET_COL: row[DATASET_COL],
            SLIDE_COL: row[SLIDE_COL],
            TILE_COL: row[TILE_COL],
            ID_COL: row[ID_COL],
            DISTANCE_COL: row[DISTANCE_COL],
            PRED_TYPE_COL: row[PRED_TYPE_COL],
            PRED_PHENO_KEY: pred_pheno_from_string(row[PRED_PHENO_COL]),
            TYPE_FROM_ANN_KEY: row[ANN_SUBTYPE_COL],
            ANN_PHENO_KEY: ann_pheno_from_string(row[ANN_PHENO_COL])
        }
        errors.append(error_dict)

    return errors


def ann_pheno_from_string(pheno_str):
    return [int(x) for x in pheno_str.split(",")]


def pred_pheno_from_string(pheno_str):
    return [float(x) for x in pheno_str.split(",")]


def evaluate(
    prediction_path,
    output_path=Path("../demo_evaluation"),
    activation_th=0.4,
    detection_radius=3.5,
    pix_pmm=2,
    label=None
):
    """
    Performs evaluation of pipeline output based on the file with matched annotations and prediction
    Evaluation is done for 2 types of cell types: main cell types and all possible cell subtypes
    Evaluation metrics:
        FG cells: error rate
        BG cells: FPs rate
    Also for both BG and FG cells confusion matrices are plotted and
    a list of errors with detailed information about prediction is saved
    Parameters
    ----------
    prediction_path: Path
        A path to a file with  matched annotations and prediction
    output_path: Path
        A path to a folder to save the results
    activation_th: float
        A threshold of phenotype marker prediction to consider it activated
    detection_radius: float
        A cell radius used to decide whether prediction is in the vicinity
    pix_pmm: float
        A number of pixels per micro meter
    label: str
        An identifier to assign to evaluation results. Can be based on the data subset ("train", val) or used model
    """

    def ann_subtype(row):
        panel_id = row[PANEL_COL]
        panel = panels[panel_id]
        ann_type = row[ANN_TYPE_COL]
        ann_pheno = ann_pheno_from_string(row[ANN_PHENO_COL])
        return panel.cell_from_annotation(ann_type, ann_pheno)

    def foreground(row):
        panel_id = row[PANEL_COL]
        panel = panels[panel_id]
        ann_type = row[ANN_TYPE_COL]
        return ann_type in panel.fg_cell_types

    def predicted_type(row):
        panel_id = row[PANEL_COL]
        panel = panels[panel_id]
        pred_pheno = pred_pheno_from_string(row[PRED_PHENO_COL])
        return panel.cell_from_prediction(pred_pheno, activation_th)

    def hit_bg(row):
        panel_id = row[PANEL_COL]
        panel = panels[panel_id]
        return not row[DETECTED_COL] or row[PRED_TYPE_COL] in panel.bg_cell_types

    def hit_main(row):
        fg_ann = row[FG_KEY]

        if fg_ann:
            return int(row[DETECTED_COL] & (row[PRED_TYPE_COL] == row[ANN_TYPE_COL]))
        else:
            return int(hit_bg(row))

    def hit_subtype(row):
        fg_ann = row[FG_KEY]

        if fg_ann:
            return int(row[DETECTED_COL] & (row[PRED_SUBTYPE_COL] == row[ANN_SUBTYPE_COL]))
        else:
            return int(hit_bg(row))

    output_path.mkdir(exist_ok=True)
    prediction_df = pd.read_csv(prediction_path, sep="\t", index_col=False)

    # Find and filter out invalid annotations
    prediction_df[INVALID_COL] = prediction_df.apply(lambda x: ann_subtype(x).is_invalid, axis=1)
    # Invalid annotations warning
    invalid_ann_df = prediction_df[prediction_df[INVALID_COL] == True]
    for _, row in invalid_ann_df.iterrows():
        annotated_positivity = ann_pheno_from_string(row[ANN_PHENO_COL])
        print("Warning: Annotation {} of {} type has inconsistent phenotype {}".format(row[ID_COL],
                                                                                       row[ANN_TYPE_COL],
                                                                                       annotated_positivity))

    # Remove rows with invalid annotations
    prediction_df = prediction_df[prediction_df[INVALID_COL] == False]
    # Add columns for evaluation
    prediction_df[DETECTED_COL] = prediction_df.apply(lambda x: x[DISTANCE_COL] < detection_radius * pix_pmm, axis=1)
    prediction_df[FG_KEY] = prediction_df.apply(lambda x: foreground(x), axis=1)
    prediction_df[ANN_SUBTYPE_COL] = prediction_df.apply(lambda x: ann_subtype(x).desc if x[FG_KEY] else x[ANN_TYPE_COL], axis=1)
    prediction_df[PRED_TYPE_COL] = prediction_df.apply(lambda x: predicted_type(x).main_type, axis=1)
    prediction_df[PRED_SUBTYPE_COL] = prediction_df.apply(lambda x: predicted_type(x).desc, axis=1)
    prediction_df[HIT_MAIN_COL] = prediction_df.apply(lambda x: hit_main(x), axis=1)
    prediction_df[HIT_SUBTYPE_COL] = prediction_df.apply(lambda x: hit_subtype(x), axis=1)
    # A tweak to correctly calculate CM for background annotation. Since a hit means that nothing is predicted in
    # the vicinity of a background annotation, we change predicted types to annotated for hits
    prediction_df[PRED_TYPE_COL] = prediction_df.apply(lambda x: x[ANN_TYPE_COL] if not x[FG_KEY] and x[HIT_MAIN_COL] else x[PRED_TYPE_COL],
                               axis=1)
    prediction_df[PRED_SUBTYPE_COL] = prediction_df.apply(lambda x: x[ANN_SUBTYPE_COL] if not x[FG_KEY] and x[HIT_SUBTYPE_COL] else x[PRED_SUBTYPE_COL],
                               axis=1)

    # Calculate error rates for main cell types and subtypes
    # and save as csv files
    calculate_performance(prediction_df, output_path, label)

    # Save errors list
    errors = extract_error_dict(prediction_df)
    errors_file_name = "errors_{}.json".format(label) if label is not None else "errors.json"
    with open(output_path / errors_file_name, "w") as f:
        json.dump(errors, f)

    # Plot confusion matrices
    # Main types
    plot_confusion_matrix(
        prediction_df[ANN_TYPE_COL].values,
        prediction_df[PRED_TYPE_COL].values,
        output_path,
        label=label
    )

    # Subtypes
    plot_confusion_matrix(
        prediction_df[ANN_SUBTYPE_COL].values,
        prediction_df[PRED_SUBTYPE_COL].values,
        output_path,
        "subtypes",
        label=label
    )


def evaluate_arg(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="../data/prediction/prediction.tsv",
        required=False,
        help="a path to a file with matched pipeline prediction"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="../demo_evaluation",
        help="a path to save the evaluation results",
    )
    parser.add_argument(
        "--marker_th",
        type=float,
        required=False,
        default=0.4,
        help="a threshold for marker expression",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=False,
        default=3.5,
        help="a detection radius to use in micrometers",
    )
    parser.add_argument(
        "--pix_pmm",
        type=float,
        required=False,
        default=2.0,
        help="number of pixels per mm",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        help="a label to add to evaluation results. E.g. to specify that data are from training or validation set"
    )

    args = parser.parse_args(argv)
    prediction_path = Path(args.prediction_path)
    output_path = Path(args.output_path)
    marker_th = args.marker_th
    detection_radius = args.radius
    pix_pmm = args.pix_pmm
    label = args.label

    evaluate(prediction_path, output_path, marker_th, detection_radius, pix_pmm, label)


if __name__ == "__main__":
    # Very first argument determines action
    actions = {"match": match, "run": evaluate_arg}

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print("Usage: " + "/".join(actions.keys()) + " ...")
    else:
        action(sys.argv[2:])
