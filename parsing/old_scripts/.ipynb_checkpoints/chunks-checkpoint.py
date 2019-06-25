# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.patches as patches
from mrcnn.visualize import random_colors, apply_mask, find_contours
import matplotlib.image as mpimg

# paths
ROOT = "/home/maxsen/git/master_thesis/data/"
DATA = "/home/maxsen/DEEPL/data/"
CHUNKS = DATA + 'training_data/large_data_extracted_396chunks_2+2/'


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

# take chunks and create annotation file from it

listofchunks = [CHUNKS + i for i in os.listdir(CHUNKS)]
data = np.load(listofchunks[2])
print(data)
anno_file = ROOT + "annotations/val_coco_id.json"
anno_file = ROOT + "annotations/instances_val2017.json"
data = json.load(open(anno_file))
img_folder = DATA + "/val/"
img_folder = "/home/maxsen/DEEPL/data/val2017/"


anno_file = ROOT + "annotations/val_coco_id.json"
anno_file = ROOT + "annotations/instances_val2017.json"
data = json.load(open(anno_file))
img_folder = DATA + "/val/"
img_folder = "/home/maxsen/DEEPL/data/val2017/"

def show_masks(img_folder, img, list_of_segmentations):
    
    N = len(list_of_segmentations)
    colors = random_colors(N)
    img = mpimg.imread(img_folder + img, img)
    
    ax = get_ax()
    masked_image = img.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        
        y1, x1, y2, x2 = list_of_segmentations[i]['bbox']
        print(y1, x1, y2, x2)
        l = [y1, x1, y2, x2]
        l2 = xyxy_to_xywh(l)
        y1, x1, y2, x2 = l2
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
        
        ax.add_patch(p)
    plt.imshow(img, cmap = "gray")
    plt.axis("off")
    
data = json.load(open(anno_file))
for one_img in data['images']:
    image_id = one_img['id']
    list_of_segmentations = [i for i in data['annotations'] if i['image_id'] == image_id ]
    
    show_masks(img_folder, one_img["file_name"], list_of_segmentations)
    s
    