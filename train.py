import os
# TODO: change
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# TODO: check which warnings we get
import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


import numpy as np
from tqdm import tqdm
import requests
import json

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tifffile

from utils import training_image

from pathlib import Path
from PIL import Image

#from sklearn.metrics import pairwise_distances_argmin_min
from csbdeep.utils import normalize

from models import model_for_training_2

epochs = 100

window = 31

pheno_shrink = 0
cell_radius = 5
radius_magnify = 5
max_examples_per_tile = 1000

# TODO: remove annotations download
def jget(url):
    return json.loads(requests.get(url).content)

annourl="http://bert/v/annotations"
dsurl="http://bert/v/datasets"

training_slides = set()
annotations = {}
boxes = {}


print( "collecting annotations ...")

n_training_cells = 0

for datasetdoc in tqdm(jget(f"{annourl}/trainingdatasets")):
#for datasetdoc in [{"dataset":"2020-01-27-phenotyping-paper-cytoagars"}]:
    dataset = datasetdoc['dataset']
    #print(dataset)
    dsu = annourl+dataset+"/"
    for slide in jget(f"{annourl}/{dataset}/"):
        dsu = annourl+dataset+"/"
        tu = dsu+slide+"/"
        for tile in jget(f"{annourl}/{dataset}/{slide}"):
            ttile = f"{dataset}/{slide}/{tile}"
            training_slides.add(ttile)
            annotations[ttile] = []
            boxes[ttile] = []
            tile_annotations = requests.get(f"{dsurl}/{ttile}/annotations.json").json()
            #if len(tile_annotations) < 5 : continue
            for pos in tile_annotations:
                if 'purpose' in pos and pos['purpose'] == "validation": continue
                annotations[ttile].append(pos)
                n_training_cells += 1
            #for box in jget(f"{dsurl}/{ttile}/boxes.json"):
            #    boxes[ttile].append(box)


# TODO: extract data loading, making training data and training into separate functions

print(f"intent to train on {n_training_cells} cells")
examples = []
y_dist = []
y_category_phenotypes = []

n_bg = 0

print("making training data ...")
for ttile in tqdm(training_slides):
    if len(annotations[ttile]) == 0: continue
    # get standardized components for this image
    if not Path(f"tilecache/{ttile}/components.tiff").is_file():
        print(f"\tcannot find cached TIFF for {ttile} ...")
        continue
    try:
        components = tifffile.imread(f"tilecache/{ttile}/components.tiff",key=range(0,6))
    except:
        print("\tcannot read TIFF for {ttile}")
        continue

    dataset = ttile.partition('/')[0]

    components = np.moveaxis(normalize(components),0,-1)
    h = components.shape[0]
    w = components.shape[1]

    #boxes[ttile] = []

    o = training_image( components, boxes[ttile], annotations[ttile], cell_radius )

    known_status = o[:,:,0] != -1

    # Includes some unknown pixels in training, more focus on edges this way
    # known_status = uniform_filter( known_status, mode='constant' ) == 1

    # sharpen cell edges a little more
    o[o[:,:,0]==0] = -1

    coordinates = np.transpose( np.nonzero( known_status ) )
    np.random.shuffle( coordinates )
    coordinates = coordinates[0:int(round(coordinates.shape[0]/8.))]
    if max_examples_per_tile < coordinates.shape[0]:
        coordinates = coordinates[0:max_examples_per_tile]
    for i,j in coordinates:
        if i <= window or j <= window or i > h-window-1 or j > w-window-1: continue
        y_dist.append( np.ndarray.flatten(o[i-1:i+2,j-1:j+2,0]) )
        y_category_phenotypes.append( np.ndarray.flatten(o[i-1:i+2,j-1:j+2,1:]) )
        examples.append( components[i-window:i+window+1, j-window:j+window+1,:] )

print( f"training on {len(examples)} examples" )

Mt = model_for_training_2(input_shape=(2*window+1,2*window+1,6))

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
        #r = (np.random.random(img.shape) > 0.01).astype(np.float32)
        #img = img * r
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


