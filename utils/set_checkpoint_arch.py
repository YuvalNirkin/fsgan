""" Utility script for overriding the architecture of saved checkpoints. """

import torch


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
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Set checkpoint architecture')
    parser.add_argument('checkpoint_path', metavar='PATH',
                        help='path to checkpoint file')
    parser.add_argument('-a', '--arch', type=str, help='network architecture')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output checkpoint path')
    parser.add_argument('--override', action='store_true',
                        help='override existing architecture')
    args = parser.parse_args()
    main(args.checkpoint_path, arch=args.arch, output=args.output, override=args.override)
