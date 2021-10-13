import numpy as np
import requests
from scipy.ndimage.morphology import distance_transform_edt
import json


def rnd(i):
    return int(round(i))


# TODO: remove
def jget(url):
    return json.loads(requests.get(url).content)


def background(c):
    return c['t'] not in ['T cell', 'B cell', 'Dendritic cell', 'NK cell', 'NKT cell']


def phenotyped(c):
    return (c['t'] in ['No cell', 'Other cell', 'Neural structure', 'Tumor cell', 'B cell']) or ('positivity' in c)


def phenotype(c):
    if c['t'] == "B cell":
        return np.array([0, 0, 1, 0, 0])
    if c['t'] in ["T cell", "Dendritic cell", "NK cell", "NKT cell"]:
        return (np.array(c['positivity'][1:6]) - 1) / 4.
    return np.array([0, 0, 0, 0, 0])


# Rename to sth like extract labels
def training_image(components, annotations, cell_radius=5):
    h = components.shape[0]
    w = components.shape[1]

    out = np.zeros((h, w, 6), np.float16)
    known_status = np.zeros((h, w), np.uint8)

    cell_at = np.zeros((h, w), np.uint32)
    cell_ph = np.zeros((len(annotations), 5), np.float16)
    cell_known = np.zeros(len(annotations))
    cell_fg = np.zeros(len(annotations))

    for i, a in enumerate(annotations):
        x, y = map(rnd, [a['x'], a['y']])
        if x >= w:
            x = w-1
        if y >= h:
            y = h-1
        cell_at[y, x] = i+1
        cell_known[i] = phenotyped(a)
        if cell_known[i]:
            cell_ph[i, :] = phenotype(a)
        cell_fg[i] = not background(a)

    od, oi = distance_transform_edt(cell_at == 0, return_indices=True)
    in_cell = od <= cell_radius

    which_cell = cell_at[oi[0][in_cell], oi[1][in_cell]] - 1
    known_status[in_cell] = cell_known[which_cell] * 255

    out[in_cell, 0] = np.where(cell_fg[which_cell], cell_radius-od[in_cell], -2)
    out[in_cell, 1:] = cell_ph[which_cell, :]

    out[known_status == 0, 0] = -1

    return out

#bu = "http://computational-immunology.org:10000/v"

#annotations=jget(f"{bu}/datasets/2020-01-27-phenotyping-paper-cytoagars/tonsil01/57055%2C8734/annotations.json")

#boxes=jget(f"{bu}/datasets/2020-01-27-phenotyping-paper-cytoagars/tonsil01/57055%2C8734/boxes.json")

#components = tifffile.imread("components.tiff",key=[0,2,3])
#components = np.moveaxis(normalize(components),0,-1)


#for i in tqdm(range(10)):
#    out = training_image( components, boxes, annotations )

#oim = out[:,:,[0,1,3]]+3

#im = Image.fromarray((oim*20).astype("uint8"))
#im.save("o.png")

