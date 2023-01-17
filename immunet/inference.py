import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import sys
import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage.feature import blob_log
from skimage import draw
import argparse
from PIL import Image
from pathlib import Path
from models import model_for_inference


def get_input(tile_path, in_channels_num):
    tile = tifffile.imread(tile_path, key=range(0, in_channels_num))
    return np.moveaxis(normalize(tile), 0, -1)


def scale_image(image, scalar, low_clip=0, up_clip=255):
    image = scalar * image
    image[image < low_clip] = low_clip
    image[image > up_clip] = up_clip
    return image


def predict(input, model, dist_scalar=50):
    try:
        dist_map, pheno_map = model.predict(np.array([input]))
    except tf.errors.ResourceExhaustedError:
        raise ValueError("Isufficient GPU memory to run inference for a single tile. Change the code to perform stitching")

    dist_map = dist_map[0][:, :, 0]
    scale_image(dist_map, dist_scalar)

    pheno_map = pheno_map[0]
    return dist_map, pheno_map


def find_cells(tile_path,
               model,
               log_threshold=0.07,
               min_log_std=3,
               max_log_std=5,
               dist_scalar=50):
    # Determine number of input channels from the model
    in_channels_num = model.layers[0].input_shape[0][3]
    input = get_input(tile_path, in_channels_num)

    dist_map, pheno_map = predict(input, model, dist_scalar)

    cell_centers = [(int(x[0]), int(x[1]))
                    for x in blob_log(dist_map, min_sigma=min_log_std, max_sigma=max_log_std, threshold=log_threshold)]

    out_markers_num = pheno_map.shape[2]
    predicted_cells = []
    for coords in cell_centers:
        rr, cc = draw.circle(coords[0], coords[1], 2, dist_map.shape)
        phenotype = [float(np.mean(pheno_map[rr, cc, j])) for j in range(out_markers_num)]
        predicted_cells.append([coords, phenotype])

    return predicted_cells


# Extracts the central subregion of a specified size from the image
def crop_image_center(image, target_size=(512, 512)):
    h, w, _ = image.shape
    new_h, new_w = target_size
    origin_y = h // 2 - (new_h // 2)
    origin_x = w // 2 - (new_w // 2)

    return image[origin_y:(origin_y + new_h), origin_x:(origin_x + new_w), :]


def save_as_png(image_array, target_path):
    image = image_array.astype("uint8")
    image = Image.fromarray(image)
    image.type = "I"
    image.save(target_path, "PNG")


def demo_inference(tile_path,
                   model_path,
                   output_path,
                   log_threshold=0.07,
                   min_log_std=3,
                   max_log_std=5,
                   dist_scalar=50):
    output_path.mkdir(exist_ok=True)

    model = tf.keras.models.load_model(str(model_path), custom_objects={"pool": tf.nn.pool, "pad": tf.pad}, compile=False) 

    in_channels_num = model.layers[0].input_shape[0][3]
    input = get_input(tile_path, in_channels_num)

    try:
        dist_map, pheno_map = predict(input, model, 0)
    except ValueError:
        # If the whole sample image is too large to fit into a GPU memory during inference,
        # take image subregion
        print("Warning: isufficient GPU memory to run inference for a single tile. Doing inference for a cetntral 512x512 subimage")
        subregion = crop_image_center(input)
        dist_map, pheno_map = predict(subregion, model, dist_scalar)

    # Output 1:
    # Make PNG visualization of distance prediction and all pseudochannels
    # Save distance map prediction as image
    save_as_png(dist_map, output_path / "distance-prediction.png")

    # Save phenotype maps prediction as images
    for i in range(pheno_map.shape[2]):
        channel = scale_image(pheno_map[:, :, i], 255)
        save_as_png(channel, output_path / "pseudochannel-{}.png".format(i + 1))

    # Output 2:
    # Use Laplacian-Of-Gaussian filter to detect cell locations and output
    # locations and phenotypes
    cell_centers = [(int(x[0]), int(x[1]))
                    for x in blob_log(dist_map, min_sigma=min_log_std, max_sigma=max_log_std, threshold=log_threshold)]

    print("{} lymphocytes has been detected".format(len(cell_centers)))

    prediction_path = output_path / "prediction.tsv"
    # Add pseudo-channels for phenotype identification
    with open(prediction_path, "w") as f:
        out_markers_num = pheno_map.shape[2]
        # Write column names first
        f.write("\t".join(["Y", "X"] + ["Marker %d" % (i + 1) for i in range(out_markers_num)]) + "\n")
        for coords in cell_centers:
            rr, cc = draw.circle(coords[0], coords[1], 2, dist_map.shape)
            cell_prediction = list(coords) + [float(np.mean(pheno_map[rr, cc, j])) for j in range(out_markers_num)]
            f.write("\t".join(["%g" % i for i in cell_prediction]) + "\n")


def demo(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tile_path',
        type=str,
        default="../input/components.tiff",
        required=False,
        help="a path to an image to use for demo inference")
    parser.add_argument(
        '--model_path',
        type=str,
        default="../train_output/immunet.h5",
        required=False,
        help="a path to a model to use for demo inference")
    parser.add_argument(
        '--output_path',
        type=str,
        default="../demo_inference",
        required=False,
        help="a path to save output of demo inference")
    parser.add_argument(
        "--log_th",
        type=float,
        default=0.07,
        required=False,
        help="a threshold for LoG blob detection algoritm",
    )
    parser.add_argument(
        "--min_log_std",
        type=float,
        default=3,
        required=False,
        help="the minimum standard deviation for Gaussian kernel of LoG, decrease to detect smaller cells",
    )
    parser.add_argument(
        "--max_log_std",
        type=float,
        default=5,
        required=False,
        help="the maximum standard deviation for Gaussian kernel of LoG, increase to detect larger cells",
    )

    args = parser.parse_args(argv)
    tile_path = Path(args.tile_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    log_th = args.log_th
    min_log_std = args.min_log_std
    max_log_std = args.max_log_std

    demo_inference(tile_path, model_path, output_path, log_th, min_log_std, max_log_std)


if __name__ == "__main__":
    # Very first argument determines action
    actions = {"demo": demo}

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print("Usage: " + "/".join(actions.keys()) + " ...")
    else:
        action(sys.argv[2:])
