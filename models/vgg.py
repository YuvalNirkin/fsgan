import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG',
    'vgg19',
    'vgg_fcn'
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# Batchnorm removed from the code, as Johnson didn't use it for his transfer learning work
# First nn.Linear shape is changed from 512 * 7 * 7 to 512 * 8 * 8 to meet our 256x256 input requirements
# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, verification=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),   # for input 256, 8x8 instead of 7x7
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if verification:
            self.classifier = nn.Sequential(*list(self.classifier.children())[:-1])
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19(num_classes=1000, pretrained=False, batch_norm=True, verifcation=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        num_classes(int): the number of classes at dataset
        pretrained (bool): If True, returns a model pre-trained on ImageNet
                            with a new FC layer 512x8x8 instead of 512x7x7
        batch_norm: if you want to introduce batch normalization
        verifcation (bool): Toggle verification mode (removes last fc from classifier)
    """
    if pretrained:
        kwargs['init_weights'] = True
    model = VGG(make_layers(cfg['E'], batch_norm=batch_norm), num_classes,  **kwargs)

    # if verifcation:
    #     verifier = nn.Sequential()
    #     for x in range(2):
    #         verifier.add_module(str(x), model.classifier[x])
    #     for x in range(3, 5):
    #         verifier.add_module(str(x), model.classifier[x])
    #     model.classifier = verifier

    if pretrained:
        # loading weights
        if batch_norm:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19_bn'])
        else:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19'])
        # loading only CONV layers weights
        for i in [0, 3, 6]:
            w = 'classifier.{}.weight'.format(str(i))
            new_w = 'not_used_{}'.format(str(i))
            b = 'classifier.{}.bias'.format(str(i))
            new_b ='not_used_{}'.format(str(i*10))
            pretrained_weights[new_w] = pretrained_weights.pop(w)
            pretrained_weights[new_b] = pretrained_weights.pop(b)

        model.load_state_dict(pretrained_weights, strict=False)

    return model


def vgg_fcn(num_classes=1000, pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
       num_classes(int): the number of classes at dataset
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        batch_norm: if you want to introduce batch normalization
    """
    if pretrained:
        kwargs['init_weights'] = True
    model = VGG(make_layers(cfg['D'], batch_norm=batch_norm), num_classes, **kwargs)

    if pretrained:
        # loading weights
        if batch_norm:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19_bn'])
        else:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19'])
        model.load_state_dict(pretrained_weights, strict=False)

    return model
