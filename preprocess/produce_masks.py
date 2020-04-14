import os
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image


def main(root_dir, out_dir, rel_dir=None, min_ratio=0.1):
    in_dir = root_dir if rel_dir is None else os.path.join(root_dir, rel_dir)
    seg_files = []

    # For each sub-directory in the input directory
    subdirs = [os.path.join(in_dir, d) for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    for curr_dir in tqdm(sorted(subdirs)):
        curr_seg_files = [os.path.basename(curr_dir) + '/' + os.path.basename(f) for f in glob(curr_dir + '/*.png')]
        curr_seg_files = sorted(curr_seg_files)
        seg_files += curr_seg_files

    # Append relative directory to segmentation paths
    for i in range(len(seg_files)):
        seg_files[i] = os.path.join(rel_dir, seg_files[i])

    # Filter out bad segmentations
    valid_seg_files = []
    for seg_file in seg_files:
        seg_path = os.path.join(root_dir, seg_file)
        seg = np.array(Image.open(seg_path))
        mask_ratio = np.count_nonzero(seg) / seg.size
        if mask_ratio > min_ratio:
            valid_seg_files.append(seg_file)

    # Write landmarks and bounding boxes to file
    np.savetxt(os.path.join(out_dir, 'seg_list.txt'), valid_seg_files, fmt='%s')


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('produce_masks')
    parser.add_argument('input', metavar='DIR', help='input root directory')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    parser.add_argument('-r', '--rel_dir', metavar='DIR', help='relative directory')
    parser.add_argument('-mr', '--min_ratio', metavar='F', type=float, default=0.1,
                        help='minimum ratio of mask non zero pixels to image pixels')
    args = parser.parse_args()
    main(args.input, args.output, args.rel_dir, args.min_ratio)
