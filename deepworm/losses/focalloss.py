# import torch
# import torch.nn.functional as F
#
# def focal_loss(x, y, alpha=0.25, gamma=2):
#     '''Focal loss.
#     Args:
#       x: (tensor) sized [N,D].
#       y: (tensor) sized [N,].
#     Return:
#       (tensor) focal loss.
#     '''
#
#     t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,21]
#     t = t[:, 1:]  # exclude background
#     t = Variable(t).cuda()  # [N,20]
#
#     p = x.sigmoid()
#     pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
#     w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
#     w = w * (1 - pt).pow(gamma)
#     return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)