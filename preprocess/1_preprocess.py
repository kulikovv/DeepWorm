"""
Deep Worm project 
Victor Kulikov 2018 Skoltech
"""

import os
import urllib2
import zipfile
from os import listdir
from os.path import basename, join, exists
from shutil import copyfile

import numpy as np
from skimage.draw import circle
from skimage.feature import corner_harris, corner_peaks
from skimage.io import imread, imsave


def download_unpack(url, destination="../data"):
    """
    Downloads and unpacks files from urls
    :param url: The zip file url
    :param destination: the 
    :return: 
    """

    path = join(destination, basename(url))
    if not exists(path):
        print("Downloading " + url)
        zipped_images = urllib2.urlopen(url)
        with open(path, 'wb') as output:
            output.write(zipped_images.read())

    if not exists(join(destination, basename(url)[:-4])):
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
    else:
        print(url + " downloaded and extracted. Nothing to do.")


download_unpack("https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_images.zip")
download_unpack("https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip")
download_unpack("https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip")

# Prepare data
destination = "../data"
files = listdir(join(destination, "BBBC010_v1_images"))
img_pairs = dict([(f.split("_")[6], join(destination, "BBBC010_v1_images", f))
                  for f in files
                  if os.path.isfile(join(destination, "BBBC010_v1_images", f))
                  and f.endswith(".tif")
                  and "w2"==f.split("_")[7]])

files = listdir(join(destination, "BBBC010_v1_foreground_eachworm"))

instances_pairs = dict()
for f in files:
    if os.path.isfile(join(destination, "BBBC010_v1_foreground_eachworm", f)) and f.endswith(".png"):
        key = f.split("_")[0]
        if not instances_pairs.has_key(key):
            instances_pairs[key] = []
        instances_pairs[key].append(join(destination, "BBBC010_v1_foreground_eachworm", f))


def get_head_tail(image, radius=12, sigma=4, min_distance=10):
    """
    Make a head tail mask of a worm
    :param image: binary worm image
    :param radius: radius used around point
    :param sigma: harris detector radius
    :param min_distance: distance between head and tail
    :return: mask of head and tail
    """
    hc = corner_harris(image, sigma=sigma)
    cp = corner_peaks(hc, min_distance=min_distance, num_peaks=2)
    mask = np.zeros_like(image)

    for c in cp:
        rr, cc = circle(c[0], c[1], radius, shape=mask.shape)
        mask[rr, cc] = 1

    return image & mask


# Make semantic data
if not os.path.exists(join(destination, "sematic_data")):
    os.makedirs(join(destination, "sematic_data"))
for key in img_pairs:
    copyfile(img_pairs[key], join(destination, "sematic_data", key + ".tif"))
    head_tails = 0
    mask = 0
    for annotation in instances_pairs[key]:
        image = imread(annotation)
        image[image > 0] = 1
        head_tails |= get_head_tail(image)
        mask += image
    mask[mask > 1] = 2  # intersections
    mask[head_tails > 0] = 3  # heads
    imsave(join(destination, "sematic_data", key + "_labels.png"), mask)


# Make instance data
if not os.path.exists(join(destination, "instance_data")):
    os.makedirs(join(destination, "instance_data"))
for key in img_pairs:
    copyfile(img_pairs[key], join(destination, "instance_data", key + ".tif"))
    counter = 0
    mask = 0
    for num,annotation in enumerate(instances_pairs[key]):
        image = imread(annotation)
        image[image > 0] = 1
        mask += image
        counter += image*(num+2)

    counter[mask > 1] = 1  # intersections
    imsave(join(destination, "instance_data", key + "_labels.png"), counter)
