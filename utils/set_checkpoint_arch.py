""" Utility script for overriding the architecture of saved checkpoints. """

import argparse
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
parser.add_argument('checkpoint_path', metavar='PATH',
                    help='path to checkpoint file')
parser.add_argument('-a', '--arch',
                    help='network architecture')
parser.add_argument('-o', '--output', metavar='PATH',
                    help='output checkpoint path')
parser.add_argument('--override', action='store_true',
                    help='override existing architecture')


def main(checkpoint_path, arch, output=None, override=False):
    output = checkpoint_path if output is None else output
    checkpoint = torch.load(checkpoint_path)
    if 'arch' in checkpoint and not override:
        print('checkpoint already contains "arch": ' + checkpoint['arch'])
        return
    print('Setting checkpoint\'s arch: ' + arch)
    checkpoint['arch'] = arch
    print('Writing chechpoint to path: ' + output)
    torch.save(checkpoint, output)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
