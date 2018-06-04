from os import listdir
from os.path import join

import numpy as np
from skimage.io import imread

basepath = "../data/BBBC010_v1_images/"
files = listdir(basepath)

mins = []
maxs = []
for f in files:
    img = imread(join(basepath, f))
    mins.append(img.min())
    maxs.append(img.max())

print(np.mean(mins), np.mean(maxs))
np.save("../data/consts.npy", {"min": np.mean(mins), "max": np.mean(maxs)})
