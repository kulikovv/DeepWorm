import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from skimage.io import imread
from skimage.io import imsave
import sys


def print_percent(percent):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
    sys.stdout.flush()


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return self.loss


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return self.loss


class MyVGG(nn.Module):
    def __init__(self, pretrained=True):
        super(MyVGG, self).__init__()
        self.mean = torch.nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.std = torch.nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        self.vgg = torchvision.models.vgg16(pretrained=pretrained)

    def forward(self, x):
        x = (x - self.mean) / self.std
        conv_1 = self.vgg.features[0](x)
        x = self.vgg.features[1](conv_1)
        conv_2 = self.vgg.features[2](x)
        x = self.vgg.features[3](conv_2)
        x = self.vgg.features[4](x)
        conv_3 = self.vgg.features[5](x)
        x = self.vgg.features[6](conv_3)
        conv_4 = self.vgg.features[7](x)
        x = self.vgg.features[8](conv_4)
        x = self.vgg.features[9](x)
        conv_5 = self.vgg.features[10](x)
        return conv_1, conv_2, conv_3, conv_4, conv_5


def optimize(content_img, style_img, vgg, k_style=150000.):
    input_img = content_img.clone()
    input_img.requires_grad = True
    optimizer = optim.LBFGS([input_img])
    style_features = vgg(style_img)
    style_losses = [StyleLoss(f) for f in style_features]
    content_features = vgg(content_img)
    content_losses = [ContentLoss(f) for f in content_features[3:]]

    run = [0]
    while run[0] <= 200:
        def closure():
            run[0] += 1
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            current_features = vgg(input_img)
            cl, sl = torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device)

            for loss, feature in zip(style_losses, current_features):
                sl += loss(feature)
            for loss, feature in zip(content_losses, current_features[3:]):
                cl += loss(feature)

            loss = k_style * sl + cl
            loss.backward()
            return loss

        optimizer.step(closure=closure)
    return input_img.data.clamp_(0, 1)


def readimg(path):
    image = imread(path).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return np.expand_dims(np.dstack([image, image, image]).transpose(2, 0, 1), 0)


class SliceMaker(object):
    def __getitem__(self, item):
        return item


slicemaker = SliceMaker()
slicex = slicemaker[140:420, 150:506]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = MyVGG(pretrained=False)
vgg.vgg.load_state_dict(torch.load('/media/hpc-4_Raid/vkulikov/vgg16-397923af.pth'))
vgg.to(device)

style_img = torch.from_numpy(readimg("../data/target_data/new_st.tif")).to(device)
num = len(os.listdir('../data/semantic_data_trans'))
train_images = [f for f in os.listdir('../data/semantic_data') if f.endswith('.tif')]
for i in range(len(train_images)):
    print_percent(int(float(i) / float(len(train_images)) * 4.))
    f = train_images[i]
    layout = imread(join('../data/semantic_data', f[:-4] + "_labels.png"))[slicex]
    imsave(join('../data/semantic_data_trans', "{0}_labels.png".format(num)), layout)
    content_img = torch.from_numpy(
        readimg(join('../data/semantic_data', f)).astype(np.float32)[slicemaker[:, :] + slicex]).to(device)
    res = optimize(content_img, style_img, vgg)
    numpy_res = res[0].mean(0).cpu().data.numpy()
    imsave(join('../data/semantic_data_trans', "{0}.tif".format(num)), numpy_res)
    num += 1

