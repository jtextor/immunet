# Script with inference: network output + cells detection

## configuration parameter: Threshold for Laplacian-Of-Gaussian cell detection
log_threshold = 0.07

dist_multiplier = 50

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
from models import model_for_inference
import tifffile
from csbdeep.utils import normalize
from PIL import Image
from skimage.feature import blob_log
from skimage import draw
from pathlib import Path


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


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model/immunet.h5", required=False,
                        help="a path to a model to use for demo inference")
    parser.add_argument('--tile_path',
                        type=str,
                        default="input/components.tiff",
                        required=False,
                        help="a path to an image to use for demo inference")

    args = parser.parse_args()

    model_path = args.model_path
    tile_path = Path(args.tile_path)

    output_path = Path("demo-output")
    output_path.mkdir(exist_ok=True)

    Mt = model_for_inference(model_path)
    # Optional: look at model structure
    # print(Mt.summary())

    x = tifffile.imread(tile_path, key=range(0,6))
    x = normalize(x)
    x = np.moveaxis(x, 0, -1)

    try:
        dist, psi = Mt.predict(np.array([x]))
    except tf.errors.ResourceExhaustedError as e:
        # If the whole sample image is too large to fit into a GPU memory during inference,
        # take image subregion
        x = crop_image_center(x)
        dist, psi = Mt.predict( np.array([x]) )

    dist = dist[0, :, :, 0]
    psi = psi[0]

    # Output 1:
    # Make PNG visualization of distance prediction and all pseudochannels
    dist = dist_multiplier * dist
    dist[dist < 0] = 0
    dist[dist > 255] = 255

    save_as_png(dist, output_path / "cell-center-distance-prediction.png")

    for i in range(psi.shape[2]):
        ch = 255 * psi[:, :, i]
        ch[ch < 0] = 0
        ch[ch > 255] = 255

        save_as_png(ch, output_path / f"pseudochannel-{i}.png")

    # Output 2:
    # Use Laplacian-Of-Gaussian filter to detect cell locations and output
    # locations and phenotypes

    cell_locations = [(int(x[0]), int(x[1]))
            for x in blob_log(dist, min_sigma=3, max_sigma=5, threshold=log_threshold)]

    print("{} lymphocytes has been detected".format(len(cell_locations)))

    prediction_path = output_path / "prediction.txt"
    # Add pseudo-channels for phenotype identification
    with open(prediction_path, "w") as f:
        for x in cell_locations:
            rr, cc = draw.circle(x[0], x[1], 2, dist.shape)
            ph = list(x) + [float(np.mean(psi[rr, cc, j])) for j in range(psi.shape[2])]
            f.write("\t".join(["%g" % i for i in ph]) + "\n")
