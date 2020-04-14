import torch
import torch.nn as nn
from torchvision import models
from fsgan.models.vgg import vgg19


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class Vgg19(torch.nn.Module):
    """ First layers of the VGG 19 model for the VGG loss.
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        model_path (str): Path to model weights file (.pth)
        requires_grad (bool): Enables or disables the "requires_grad" flag for all model parameters
    """
    def __init__(self, model_path: str = None, requires_grad: bool = False):
        super(Vgg19, self).__init__()
        if model_path is None:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        else:
            model = vgg19(pretrained=False)
            checkpoint = torch.load(model_path)
            del checkpoint['state_dict']['classifier.6.weight']
            del checkpoint['state_dict']['classifier.6.bias']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class VGGLoss(nn.Module):
    """ Defines a criterion that captures the high frequency differences between two images.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_

    Args:
        model_path (str): Path to model weights file (.pth)
    """
    def __init__(self, model_path: str = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(model_path)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
