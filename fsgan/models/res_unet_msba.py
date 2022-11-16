import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fsgan.utils.img_utils import create_pyramid
from fsgan.models.msba import MSBA


def make_conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, bias=False,
                     padding_type='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), use_dropout=False):
    conv_block = []
    p = 0
    if padding_type is not None:
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'zero':
            p = kernel_size // 2 if padding is None else padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    elif padding is not None:
        p = padding

    conv_block.append(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=p, bias=bias))
    if norm_layer is not None:
        conv_block.append(norm_layer(out_nc))
    if act_layer is not None:
        conv_block.append(act_layer)

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    return conv_block


class DownBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='reflect',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(DownBlock, self).__init__()
        model = make_conv_block(in_nc, out_nc, kernel_size, 2, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer)
        model += make_conv_block(out_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='reflect',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(UpBlock, self).__init__()
        # model = [nn.Upsample(scale_factor=2, mode='bilinear')]
        model = make_conv_block(in_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, expansion=1, padding_type='reflect',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        model = make_conv_block(planes, planes*expansion, kernel_size, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        model += make_conv_block(planes*expansion, planes, kernel_size, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=None, use_dropout=False)
        self.model = nn.Sequential(*model)
        self.act = act_layer

    def forward(self, x):
        out = x + self.model(x)
        out = self.act(out)
        return out


class FlatBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, layers=1, padding_type='reflect',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(FlatBlock, self).__init__()
        if layers <= 0:
            self.model = None
        else:
            model = []
            for i in range(layers):
                model.append(ResnetBlock(planes, kernel_size, 1, padding_type, norm_layer, act_layer))
            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.model is None:
            return x
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|-- flat --|
class SkipConnectionBlock(nn.Module):
    def __init__(self, ngf, sub_ngf, down_block=None, submodule=None, up_block=None, flat_block=None, flat_layers=1,
                 padding_type='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True), use_dropout=False):
        super(SkipConnectionBlock, self).__init__()
        self.submodule = submodule
        if submodule is not None:
            assert down_block is not None and up_block is not None
            self.down_block = down_block(ngf, sub_ngf, 3, padding_type, norm_layer, act_layer)
            self.up_block = up_block(sub_ngf, ngf, 3, padding_type, norm_layer, act_layer)
        if flat_block is not None:
            self.flat_block = flat_block(ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        else:
            self.flat_block = None

    def forward(self, x):
        if self.submodule is not None:
            x = x + self.up_block(self.submodule(self.down_block(x)))
        if self.flat_block is not None:
            return self.flat_block(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=3, max_nc=None,
                 ngf=64, flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True),
                 use_dropout=False):
        super(ResUNet, self).__init__()
        max_nc = 1000000 if max_nc is None else max_nc
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.in_conv = nn.Sequential(*make_conv_block(in_nc, ngf, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        out_act = MSBA(out_nc, bins=64)
        self.out_conv = make_conv_block(ngf, out_nc * out_act.bins, kernel_size=7, norm_layer=None, act_layer=None)
        self.out_conv.append(out_act)
        self.out_conv = nn.Sequential(*self.out_conv)

        self.levels = len(flat_layers)
        unet_block = None
        for i in range(1, self.levels + 1):
            curr_ngf = min(ngf * (2 ** (self.levels - i)), max_nc)
            curr_sub_ngf = min(ngf * (2 ** (self.levels - i + 1)), max_nc)
            unet_block = SkipConnectionBlock(curr_ngf, curr_sub_ngf,
                                             down_block, unet_block, up_block, flat_block, flat_layers=flat_layers[-i],
                                             norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        self.inner = unet_block

    def forward(self, x):
        x = self.in_conv(x)
        x = self.inner(x)
        x = self.out_conv(x)

        return x


# Local enhancer:
# x |-- in_conv --| ----identity---- |-- flat --|
# y |-------------| -- upsampling -- |
class LocalEnhancer(nn.Module):
    def __init__(self, ngf, sub_ngf, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=3,
                 flat_layers=0, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(inplace=True), use_dropout=False):
        super(LocalEnhancer, self).__init__()
        self.in_conv = nn.Sequential(*make_conv_block(in_nc, ngf, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        self.up_block = up_block(sub_ngf, ngf, 3, padding_type, norm_layer, act_layer)
        if flat_block is not None:
            # self.flat_block = flat_block(sub_ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
            self.flat_block = flat_block(ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        else:
            self.flat_block = None

        out_act = MSBA(out_nc, bins=64)
        self.out_conv = make_conv_block(ngf, out_nc * out_act.bins, kernel_size=7, norm_layer=None, act_layer=None)
        self.out_conv.append(out_act)
        self.out_conv = nn.Sequential(*self.out_conv)

    def extract_features(self, x, y):
        x = self.in_conv(x) + self.up_block(y)
        x = self.flat_block(x) if self.flat_block is not None else x

        return x

    def forward(self, x, y):
        x = self.extract_features(x, y)
        x = self.out_conv(x)

        return x


class MultiScaleResUNet(nn.Module):
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=3,
                 max_nc=1024, ngf=64, flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(inplace=True), use_dropout=False, n_local_enhancers=1):
        super(MultiScaleResUNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_local_enhancers = n_local_enhancers

        # Global
        ngf_global = ngf * (2 ** n_local_enhancers)
        self.base = ResUNet(down_block, up_block, flat_block, in_nc, out_nc, max_nc, ngf_global,
                            flat_layers[n_local_enhancers:], norm_layer, act_layer, use_dropout)

        # Local enhancers
        for n in range(1, n_local_enhancers + 1):
            curr_ngf = min(ngf * (2 ** (n_local_enhancers - n)), max_nc)
            curr_sub_ngf = min(curr_ngf * 2, max_nc)
            enhancer = LocalEnhancer(curr_ngf, curr_sub_ngf, down_block, up_block, flat_block, in_nc, out_nc,
                                     flat_layers[n - 1], 'reflect', norm_layer, act_layer, use_dropout)
            self.add_module('enhancer%d' % n, enhancer)

    def forward(self, pyd):
        pyd = create_pyramid(pyd, self.n_local_enhancers)

        # Call global at the coarsest level
        if len(pyd) == 1:
            return self.base(pyd[-1])

        x = pyd[-1]
        x = self.base.in_conv(x)
        x = self.base.inner(x)

        # Apply enhancer for each level
        for n in range(1, len(pyd)):
            enhancer = getattr(self, 'enhancer%d' % n)
            # x = enhancer(pyd[self.n_local_enhancers - n], x)
            x = enhancer.extract_features(pyd[self.n_local_enhancers - n], x)
            if n == self.n_local_enhancers:
                x = enhancer.out_conv(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        # Find base in state_dict
        base_only_weights = True
        for name in state_dict.keys():
            if name.startswith('base'):
                base_only_weights = False

        if base_only_weights:
            self.base.load_state_dict(state_dict, strict)
        else:
            super(MultiScaleResUNet, self).load_state_dict(state_dict, strict)


def res_unet(**kwargs):
    if 'num_classes' in kwargs:
        kwargs['out_nc'] = kwargs['num_classes']
        del kwargs['num_classes']

    return ResUNet(**kwargs)


def main(model='res_unet.ResUNet', res=(256,)):
    from fsgan.utils.obj_factory import obj_factory
    model = obj_factory(model)
    if len(res) == 1:
        img = torch.rand(1, model.in_nc, res, res)
        pred = model(img)
        print(pred.shape)
    else:
        img = []
        for i in range(1, len(res) + 1):
            img.append(torch.rand(1, model.in_nc, res[-i], res[-i]))
        pred = model(img)
        print(pred.shape)
        # for i in range(1, len(res) + 1):
        #     print(pred[-i].shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('res_unet test')
    parser.add_argument('model', default='res_unet.ResUNet',
                        help='model object')
    parser.add_argument('-r', '--res', default=(256,), type=int, nargs='+',
                        metavar='N', help='image resolution')
    args = parser.parse_args()
    main(args.model, args.res)
