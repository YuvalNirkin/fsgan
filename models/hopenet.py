import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck


class Hopenet(nn.Module):
    """ Defines a head pose estimation network with 3 output layers: yaw, pitch and roll.
    `"Fine-Grained Head Pose Estimation Without Keypoints" <https://arxiv.org/pdf/1710.00925.pdf>`_.

    Predicts Euler angles by binning and regression.

    Args:
        block (nn.Module): Main convolution block
        layers (list of ints): Number of blocks per intermediate layer
        num_bins (int): Number of regression bins
    """
    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3), num_bins=66):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.idx_tensor = None
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_yaw = self.fc_yaw(x)
        pred_pitch = self.fc_pitch(x)
        pred_roll = self.fc_roll(x)

        yaw_predicted = F.softmax(pred_yaw, dim=1)
        pitch_predicted = F.softmax(pred_pitch, dim=1)
        roll_predicted = F.softmax(pred_roll, dim=1)

        if self.idx_tensor is None:
            self.idx_tensor = torch.arange(0, 66, out=torch.FloatTensor()).to(x.device)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor, axis=1).unsqueeze(1) * 3 - 99

        return torch.cat((yaw_predicted, pitch_predicted, roll_predicted), axis=1)
