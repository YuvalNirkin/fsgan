import torch
import torch.nn as nn
import torch.nn.functional as F


pretrained_models = {'pascal': 'path/to/pretrained_model.pth'}


# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py
class UNet(nn.Module):
    """ Defines a variant of the UNet architecture described in the paper:
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/pdf/1505.04597.pdf>`_.

    Args:
        feature_scale (int): Divides the intermediate feature map number of channels
        n_classes (int): Output number of channels
        is_deconv (bool): If True, transposed convolution will be used for the upsampling operation instead of
            bilinear interpolation
        in_channels (int): Input number of channels
        is_batchnorm (bool): If True, enables the use of batch normalization
    """
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=False, in_channels=3, is_batchnorm=True):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


class UnetConv2(nn.Module):
    """ Defines the UNet's convolution block.

    Args:
        in_size (int): Input number of channels
        out_size (int): Output number of channels
        is_batchnorm (bool): If True, enables the use of batch normalization
    """
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):
    """ Defines the UNet's upsampling block.

    Args:
        in_size (int): Input number of channels
        out_size (int): Output number of channels
        is_deconv (bool): If True, transposed convolution will be used for the upsampling operation instead of
            bilinear interpolation
    """
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1d = nn.Conv1d(in_size, out_size, kernel_size=(1,1))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs2 = self.conv1d(outputs2,)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


def unet(num_classes=21, is_deconv=False, feature_scale=1, is_batchnorm=True, pretrained=False):
    """ Creates a UNet model with pretrained optiopn.

    Args:
        num_classes (int): Output number of channels
        is_deconv (bool): If True, transposed convolution will be used for the upsampling operation instead of
                bilinear interpolation
        feature_scale (int): Divides the intermediate feature map number of channels
        is_batchnorm (bool): If True, enables the use of batch normalization
        pretrained (bool): If True, return a pretrained model on Pascal dataset

    Returns:
        UNet model
    """
    if pretrained:
        model_path = pretrained_models['pascal']
        model = UNet(n_classes=num_classes, feature_scale=feature_scale, is_batchnorm=is_batchnorm, is_deconv=is_deconv)
        checkpoint = torch.load(model_path)
        weights = checkpoint['state_dict']
        weights['notinuse'] = weights.pop('final.weight')
        weights['notinuse2'] = weights.pop('final.bias')
        model.load_state_dict(weights, strict=False)
    else:
        model = UNet(n_classes=num_classes, feature_scale=feature_scale, is_batchnorm=is_batchnorm, is_deconv=is_deconv)

    return model
