import argparse
import os
from os import listdir
from os.path import isfile, join

from skimage.io import imread, imsave

from convert_utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert a dataset to bbbc format.')


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


parser.add_argument("-o", dest="output_path", action=ReadableDir, default="../data/target_data/",
                    help="output directory")
parser.add_argument("-s", dest="source_folder", action=ReadableDir, help="set input file folder", required=True)
args = parser.parse_args()


def convert_files(folder, output, size_scale_func, intensity_func, suffix='.tif'):
    files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(suffix)]
    for f in files:
        print("Processing {0} from {1} saving to {2}".format(f,len(files),join(output, f)))
        source_image = imread(join(folder, f))
        target_image = intensity_func(size_scale_func(source_image))
        imsave(join(output, f), target_image)


bbbc_example = imread("../data/instance_data/B04.tif")
bbbc_labels = imread("../data/instance_data/B04_labels.png")

bbbc_fg, bbbc_bg = calc_intensity(bbbc_example, bbbc_labels > 0, 20)
bbbc_average_size = np.array([np.sum(bbbc_labels == i) for i in np.unique(bbbc_labels) if i > 0]).mean()
print('Average worm size for bbbc is {0}'.format(bbbc_average_size))

mipt_example = imread(join(args.source_folder, "Stream002.tif"))[1250:1750,600:1300]

mipt_mask = estimate_worm_mask(mipt_example)

mipt_fg, mipt_bg = calc_intensity(mipt_example, mipt_mask > 0, 40)
print('Average worm size for mipt is {0}'.format(np.sum(mipt_mask > 0)))

rescale_func = get_rescale_func(bbbc_average_size, np.sum(mipt_mask > 0))
intensity_func = get_color_shift(bbbc_fg, mipt_fg, bbbc_bg, mipt_bg)

convert_files(args.source_folder, args.output_path, rescale_func, intensity_func)
