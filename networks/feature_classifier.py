import torch
import functools
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


# from the authors of https://github.com/Yangyangii/DANN-pytorch
class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.alpha = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha

        return output, None


class FeatureClassifier(nn.Module):
    def __init__(self, in_channel=512, width=16, height=8):
        super(FeatureClassifier, self).__init__()
        self.width = width
        self.height = height
        self.down = nn.Conv2d(in_channel, 128, 5, 2, padding=2, bias=False)
        self.relu0 = nn.ReLU(True)
        self.fc1 = nn.Linear(int((128 * (width / 8) * (height / 8))), 64)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(64, 1)
        self.activation = nn.Sigmoid()

    def forward(self, feature, lambda_=1):
        feature = GradientReversalLayer.apply(feature, lambda_)

        feat = self.down(feature)
        feat = self.relu0(feat)
        feat = feat.view(-1, int(128 * (self.width / 8) * (self.height / 8)))
        feat = self.fc1(feat)
        feat = self.relu1(feat)
        feat = self.fc2(feat)
        domain_output = self.activation(feat)

        return domain_output

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, 
                 input_nc, 
                 ndf=64, 
                 n_layers=3, 
                 norm_layer=nn.InstanceNorm2d,
                 kernel_size = 4,
                 padding = 1,
                 stride = 1
                 ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = kernel_size
        padw = padding
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return torch.sigmoid(self.model(input))