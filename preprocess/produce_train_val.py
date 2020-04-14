import os
from glob import glob1
import numpy as np


def parse_files(dir, postfix='.mp4', cache_postfix=None):
    files = glob1(dir, '*' + postfix)
    if cache_postfix is not None:
        files = [f for f in files if os.path.isfile(os.path.join(dir, os.path.splitext(f)[0] + cache_postfix))]
    dir = os.path.expanduser(dir)
    for fname in sorted(os.listdir(dir)):
        path = os.path.join(dir, fname)
        if os.path.isdir(path):
            files += [os.path.join(fname, f).replace('\\', '/') for f in parse_files(path, postfix, cache_postfix)]

    return sorted(files)


def main(in_dir, out_dir=None, ratio=0.1, postfix='.mp4', cache_postfix=None):
    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory does not exist: ' + in_dir)
    out_dir = in_dir if out_dir is None else out_dir
    if not os.path.isdir(out_dir):
        raise RuntimeError('Output directory does not exist: ' + out_dir)

    # Parse files
    file_rel_paths = np.array(parse_files(in_dir, postfix, cache_postfix))

    # Generate directory splits
    n = len(file_rel_paths)
    val_indices = np.random.choice(n, int(np.round(n * ratio)), replace=False).astype(int)
    train_indices = np.setdiff1d(np.arange(n), val_indices)
    train_indices.sort()
    val_indices.sort()

    train_file_list = file_rel_paths[train_indices]
    val_file_list = file_rel_paths[val_indices]

    # Output splits to file
    np.savetxt(os.path.join(out_dir, 'train_list.txt'), train_file_list, fmt='%s')
    np.savetxt(os.path.join(out_dir, 'val_list.txt'), val_file_list, fmt='%s')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('produce_train_val')
    parser.add_argument('input', help='dataset root directory')
    parser.add_argument('-o', '--output', help='output directory')
    parser.add_argument('-r', '--ratio', default=0.1, type=float, help='ratio of validation split')
    parser.add_argument('-p', '--postfix', default='.mp4', help='files postfix')
    parser.add_argument('-cp', '--cache_postfix', help='cache postfix')
    args = parser.parse_args()
    main(args.input, args.output, args.ratio, args.postfix, args.cache_postfix)
