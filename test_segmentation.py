from os import listdir
from os.path import join

import numpy as np
import torch
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from torch.autograd import Variable

from deepworm import get_segmentation_model


def process_image(image, net):
    consts = np.load('data/consts.npy').tolist()
    print(image.min(),image.max())
    image_norm = (image - consts['min']) / (consts['max'] - consts['min']) - 0.5
    #image_norm = (image - image.min()) / (image.max() - image.min()) - 0.5
    image_norm = np.expand_dims(np.expand_dims(image_norm, 0),0)
    print(image_norm.shape)
    vx = Variable(torch.from_numpy(image_norm).float()).cuda()
    res = net(vx)
    return np.argmax(res[0].data.cpu().numpy(), 0)


if __name__ == "__main__":
    net = get_segmentation_model()
    net.load_state_dict(torch.load('models/semantic_worms.t7'))
    net = net.cuda()
    net.eval()

    basepath = 'data/target_data'
    #basepath = 'data/semantic_data'
    for f in [f for f in listdir(basepath) if f.endswith(".tif")][:10]:
        image = imread(join(basepath, f)).astype(np.float32)
        res = process_image(image, net)
        plt.imshow(image)
        plt.waitforbuttonpress()
        plt.imshow(res)
        plt.waitforbuttonpress()