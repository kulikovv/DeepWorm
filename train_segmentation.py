import torch
import torch.nn as nn
from torch.autograd import Variable

from deepworm import SemanticWorms
from deepworm import clip_align_3D, random_scale, clip_patch, rotate90, flip_vertically, flip_horizontally, \
    random_brightness, random_contrast, get_segmentation_model,print_percent


class TrainSemantic():
    def __init__(self, cuda=True):
        self.trans = [random_scale(0.2),
                      clip_patch((256, 256)),
                      flip_horizontally(),
                      flip_vertically(),
                      rotate90(),
                      random_brightness(0.1),
                      random_contrast(0.1)]
        self.cuda = cuda

    def get_batch(self, func, size=20):
        x, l = func(batch_size=size, transforms=self.trans)
        vx = Variable(torch.from_numpy(x).float(), requires_grad=False)
        vl = Variable(torch.from_numpy(l).long(), requires_grad=False)
        if self.cuda:
            vx = vx.cuda()
            vl = vl.cuda()
        return vx, vl

    def train_func(self, niter=4000):
        errors = []
        data = SemanticWorms("data/semantic_data/")
        criterion = nn.CrossEntropyLoss()
        net = get_segmentation_model()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        if self.cuda:
            net = net.cuda()

        for i in range(niter):
            vx, vl = self.get_batch(data.get_train_batch)
            p = net(vx)
            optimizer.zero_grad()
            loss = criterion(p, clip_align_3D(p, vl))
            loss.backward()
            optimizer.step()
            err = loss.data.select(0, 0)
            errors.append(err)
            if 0 == (i+1) % 10:
                #vx, vl = self.get_batch(data.get_train_batch, 10)
                #p = net(vx)
                #val_loss = criterion(p, clip_align_3D(p, vl))
                #scheduler.step(val_loss.cpu().data.numpy())
                print_percent(int(float(i) / float(niter) * 20.))

        return net


if __name__ == "__main__":
    print("Do train")
    training = TrainSemantic()
    net = training.train_func(1000)
    torch.save(net.state_dict(), 'models/semantic_worms.t7')
