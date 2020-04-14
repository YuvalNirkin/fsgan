import torch.nn as nn
import numpy as np
from fsgan.utils.img_utils import create_pyramid


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class NLayerDiscriminator(nn.Module):
    """ Defines the PatchGAN discriminator.
    `"Image-to-Image Translation with Conditional Adversarial Networks" <https://arxiv.org/pdf/1611.07004.pdf>`_

    Args:
        input_nc (int): Input number of channels
        ndf (int): Number of the discriminator feature channels of the first layer
        n_layers (int): Number of intermediate layers
        norm_layer (nn.Module): Type of feature normalization
        use_sigmoid (bool): If True, a Sigmoid activation will be used after the final layer
        getIntermFeat (bool): If True, all intermediate features will be returned else only the final feature
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class MultiscaleDiscriminator(nn.Module):
    """ Defines the multi-scale descriminator.
    `"High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
    <https://arxiv.org/pdf/1711.11585.pdf>`_.

    Args:
        input_nc (int): Input number of channels
        ndf (int): Number of the discriminator feature channels of the first layer
        n_layers (int): Number of intermediate layers
        norm_layer (nn.Module): Type of feature normalization
        use_sigmoid (bool): If True, a Sigmoid activation will be used after the final layer
        num_D (int): Number of discriminators
        getIntermFeat (bool): If True, all intermediate features will be returned else only the final feature
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        input = create_pyramid(input, self.num_D)
        levels = len(input)
        result = []
        for i in range(levels):
            curr_input = input[i]
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(levels - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(levels - 1 - i))
            result.append(self.singleD_forward(model, curr_input))

        return result
