# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
from od.models.modules.common import Conv
from od.models.backbone import build_backbone
from od.models.neck import build_neck
from od.models.head import build_head
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

class Model_flexible(nn.Module):
    def __init__(self, model_config, pretrain=False):
        """
        :param model_config:
        """

        super(Model_flexible, self).__init__()
        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'))
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, pretrain, **model_config.backbone)
        backbone_out = self.backbone.out_shape

        self.fpn = build_neck('FPN', **backbone_out)
        fpn_out = self.fpn.out_shape

        fpn_out['version'] = model_config.backbone.version
        self.pan = build_neck('PAN', **fpn_out)

        pan_out = self.pan.out_shape
        model_config.head['ch'] = pan_out
        self.detection = build_head('YOLOHead', **model_config.head)
        self.stride = self.detection.stride
        self._initialize_biases()

        initialize_weights(self.fpn)
        initialize_weights(self.pan)
        initialize_weights(self.detection)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.detection  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for module in [self.backbone, self.fpn, self.pan, self.detection]:
            for m in module.modules():
                if type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def forward(self, x, augment=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        else:
            return self.forward_once(x)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x):
        out = self.backbone(x)
        out = self.fpn(out)
        out = self.pan(out)
        y = self.detection(list(out))
        return y

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model = Model(model_config='../../configs/model_resnet.yaml').to(device)
    # model.fuse()
    import time

    tic = time.time()
    y = model(x)
    for item in y:
        print(item.shape)
