import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


def make_linear_block(in_nc, out_nc, bias=False, norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU(True), use_dropout=False):
    linear_block = []
    linear_block.append(nn.Linear(in_nc, out_nc, bias=bias))
    if norm_layer is not None:
        linear_block.append(norm_layer(out_nc))
    if act_layer is not None:
        linear_block.append(act_layer)

    if use_dropout:
        linear_block += [nn.Dropout(0.5)]

    return linear_block


class Classifier(nn.Module):
    def __init__(self, in_nc=2048, out_nc=2, layers=(2048,), norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU(True),
                 use_dropout=False):
        super(Classifier, self).__init__()
        self.idx_tensor = None

        # Add linear layers
        channels = [in_nc] + list(layers) + [out_nc]
        self.model = []
        for i in range(1, len(channels) - 1):
            self.model += make_linear_block(channels[i - 1], channels[i], norm_layer=norm_layer, act_layer=act_layer,
                                            use_dropout=use_dropout)
        self.model += make_linear_block(channels[-2], channels[-1], norm_layer=None, act_layer=None,
                                        use_dropout=use_dropout)
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


def classifier(pretrained=False, **kwargs):
    model = Classifier(**kwargs)

    if pretrained:
        if os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError('Could not find weights file: %s' % pretrained)

    return model


def main(obj_exp):
    from fake_detection.utils.obj_batch import obj_factory
    obj = obj_factory(obj_exp)
    print(obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('classifier1d')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)