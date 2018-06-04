from math import sqrt

import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, dilation
from skimage.transform import rescale


def calc_intensity(image, mask, size=45):
    sem = disk(size)
    around_mask = dilation(mask, sem)
    around_mask[mask] = 0
    return np.mean(image[mask]), np.mean(image[around_mask])


def get_color_shift(bbbc_fg, mipt_fg, bbbc_bg, mipt_bg):
    shift = mipt_fg - bbbc_fg
    scale = (mipt_bg - shift) / bbbc_bg

    def func(i):
        return (i - shift) * scale

    return func


def get_rescale_func(worm_bbbc_size, worm_target_size):
    k = sqrt(float(worm_bbbc_size) / float(worm_target_size))

    def func(I_target):
        return rescale(I_target, k, preserve_range = True)

    return func


def estimate_worm_mask(image, kernel_size=7):
    gaussian_image = gaussian(image, kernel_size)
    thresh = threshold_otsu(gaussian_image)
    return gaussian_image < thresh
