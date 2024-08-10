"""
Multi-Scale Binned Activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSBA(nn.Module):
    def __init__(self, out_nc=3, bins=64):
        super(MSBA, self).__init__()
        self.in_nc = out_nc * bins
        self.out_nc = out_nc
        self.bins = bins
        self.norm_factor = 2. / (self.bins - 1)

        # Initialize scales tensor
        self.register_buffer('scales', torch.arange(0., bins).view(1, -1))

    def forward(self, x):
        assert x.shape[1] == self.in_nc
        scales = self.scales.view(self.scales.shape + (1,) * (x.ndim - self.scales.ndim))

        out = []
        for i in range(self.out_nc):
            xc = F.softmax(x[:, i * self.bins:(i + 1) * self.bins], dim=1)
            xc = torch.sum(xc * scales, dim=1, keepdim=True).mul_(self.norm_factor).sub_(1.)
            out.append(xc)
        out = torch.cat(out, dim=1)

        return out


def main():
    msba = MSBA()
    img = torch.rand(2, msba.in_nc, 64, 64)
    out = msba(img)
    print(out.shape)


if __name__ == "__main__":
    main()
