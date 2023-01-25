import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json

import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from prepare_data import make_samples
from models import model_for_training, compress_model
from annotations import load_annotations


def plot_train_history(losses, val_losses=None, file_name="history"):
    plt.figure()
    plt.plot(losses)
    has_val_losses = val_losses is not None and len(val_losses) > 0
    if has_val_losses:
        plt.plot(val_losses)
    plt.title("Training history")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    legend = ["train", "val"] if has_val_losses else ["train"]
    plt.legend(legend, loc="upper right")
    plt.savefig("{}.png".format(file_name))


class HistoryVis(Callback):
    def __init__(self, history_name):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.history_name = history_name

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        if "val_loss" in logs:
            self.val_losses.append(logs["val_loss"])
        # Plot train history
        plot_train_history(self.losses, self.val_losses, self.history_name)
        # Save train history
        with open("{}.json".format(self.history_name), "w") as f:
            hist_dict = {"loss": self.losses}
            if len(self.val_losses) > 0:
                hist_dict["val_loss"] = self.val_losses
            json.dump(hist_dict, f)


def make_data_generator(generator, X, Y, batch_size):
    i = generator.flow(X, Y, seed=7, batch_size=batch_size)

    while True:
        # Get next batch
        img, y = i.next()
        # For each sample in batch
        for j in range(img.shape[0]):
            # for each channel
            for k in range(img[j].shape[2]):
                # TODO: explain what this transformation is about?
                img[j, :, :, k] = img[j, :, :, k] * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        # Return modified image and y split into distances and phenotype labels
        yield img, [y[:, 0:9], y[:, 9:]]


def process_annotations(annotations,
                        images_path,
                        max_examples_per_tile,
                        in_channels_num,
                        out_channels_num,
                        cell_radius,
                        window,
                        batch_size):
    (patches, labels) = make_samples(
        annotations,
        images_path,
        window,
        max_examples_per_tile,
        in_channels_num,
        out_channels_num,
        cell_radius
    )

    datagen = ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, fill_mode="reflect", rotation_range=90
    )

    data_generator = make_data_generator(
        datagen, patches, labels, batch_size
    )

    patches_num = patches.shape[0]
    steps = patches_num // batch_size

    print("{} patches".format(patches_num))

    return data_generator, steps


def run_training(
    train_annotations_path,
    val_annotations_path,
    images_path,
    in_channels_num,
    out_markers_num,
    cell_radius,
    epochs,
    model_cp_name,
    history_name,
    weights_path=None,
    batch_size=64,
    max_examples_per_tile=1000,
    window=31
):
    np.random.seed(99)

    print("loading annotations...")
    train_annotations = load_annotations(train_annotations_path)

    train_cells_num = sum(
        [len(tile.annotations) for tile in train_annotations]
    )
    print(f"{train_cells_num} cells for training")

    print("making training data ...")
    print("training on", end="")

    train_data_gen, train_steps = process_annotations(
        train_annotations,
        images_path,
        max_examples_per_tile,
        in_channels_num,
        out_markers_num,
        cell_radius,
        window,
        batch_size)

    has_val_data = val_annotations_path is not None
    if has_val_data:
        val_annotations = load_annotations(val_annotations_path)
        val_cells_num = sum(
            [len(tile.annotations) for tile in val_annotations]
        )
        print(f"{val_cells_num} cells for validation")

        print("making validation data ...")
        print("validation on", end="")

        val_data_gen, val_steps = process_annotations(
            val_annotations,
            images_path,
            max_examples_per_tile,
            in_channels_num,
            out_markers_num,
            cell_radius,
            window,
            batch_size)
    else:
        val_data_gen = None
        val_steps = None


    Mt = model_for_training(
        input_shape=(2 * window + 1, 2 * window + 1, in_channels_num), out_markers_num=out_markers_num
    )

    Mt.compile(
        optimizer="adam",
        loss=["mean_squared_error", "mean_squared_error"],
        loss_weights=[1, 20],
    )

    if weights_path is not None:
        Mt.load_weights(weights_path)

    loss_to_monitor = "val_loss" if has_val_data else "loss"

    checkpoint = ModelCheckpoint(
        "{}.hdf5".format(str(model_cp_name)),
        monitor=loss_to_monitor,
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    history_vis = HistoryVis(history_name)

    history = Mt.fit(
        train_data_gen,
        steps_per_epoch=train_steps,
        validation_data=val_data_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[checkpoint, history_vis],
    )

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"] if has_val_data else None

    plot_train_history(train_loss, val_loss, history_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_ann_path",
        type=str,
        default="../data/annotations/annotations_train.json.gz",
        required=False,
        help="a path to a json file with training annotations",
    )
    parser.add_argument(
        "--val_ann_path",
        type=str,
        default="../data/annotations/annotations_val.json.gz",
        required=False,
        help="a path to a json file with validation annotations",
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
        default="../train_output",
        required=False,
        help="a path to save output",
    )
    parser.add_argument(
        "--in_channels_num",
        type=int,
        default=7,
        required=False,
        help="a number of mIHC image channels to use",
    )
    parser.add_argument(
        "--out_markers_num",
        type=int,
        default=5,
        required=False,
        help="a number of phenotype markers to predict",
    )
    parser.add_argument(
        "--cell_radius",
        type=int,
        default=5,
        required=False,
        help="a cell radius in pixels to use for labels generation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        required=False,
        help="a number of epochs to run training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        required=False,
        help="a batch size to use during training",
    )
    parser.add_argument(
        "--max_patch",
        type=int,
        default=1000,
        required=False,
        help="maximum number of patches to extract from tile",
    )
    parser.add_argument(
        "--model_cp_name",
        type=str,
        default="model_cp",
        required=False,
        help="a file name to save model checkpoint",
    )
    parser.add_argument(
        "--history_name",
        type=str,
        default="train_history",
        required=False,
        help="a file name to save training history",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=False,
        help="a path to a model to use to resume training",
    )

    args = parser.parse_args()
    train_annotations_path = Path(args.train_ann_path)
    val_annotations_path = (
        Path(args.val_ann_path) if args.val_ann_path is not None else None
    )
    images_path = Path(args.images_path)
    train_output = Path(args.output_path)
    if not train_output.is_dir():
        train_output.mkdir()

    in_channels_num = args.in_channels_num
    out_markers_num = args.out_markers_num
    cell_radius = args.cell_radius
    epochs = args.epochs
    batch_size = args.batch_size
    max_examples_per_tile = args.max_patch
    model_cp_name = train_output / args.model_cp_name
    history_name = train_output / args.history_name
    weights_path = args.weights_path

    run_training(
        train_annotations_path,
        val_annotations_path,
        images_path,
        in_channels_num,
        out_markers_num,
        cell_radius,
        epochs,
        model_cp_name,
        history_name,
        weights_path,
        batch_size,
        max_examples_per_tile
    )

    # Compress model and save compressed model under the same name as model checkpoint
    compress_model(model_cp_name, train_output / "immunet")
