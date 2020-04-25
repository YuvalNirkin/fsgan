import os
import pickle
import numpy as np
import cv2
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from fsgan.utils.video_utils import Sequence


def main(input_path, output_dir=None, cache_path=None, seq_postfix='_dsfd_seq.pkl', out_postfix='.jpg', resolution=256,
         crop_scale=1.2):
    cache_path = os.path.splitext(input_path)[0] + seq_postfix if cache_path is None else cache_path
    if output_dir is None:
        output_dir = os.path.splitext(input_path)[0]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    # Verification
    if not os.path.isfile(input_path):
        raise RuntimeError('Input video does not exist: ' + input_path)
    if not os.path.isfile(cache_path):
        raise RuntimeError('Cache file does not exist: ' + cache_path)
    if not os.path.isdir(output_dir):
        raise RuntimeError('Output directory does not exist: ' + output_dir)

    print('=> Cropping image sequences from image: "%s"...' % os.path.basename(input_path))

    # Load sequences from file
    with open(cache_path, "rb") as fp:  # Unpickling
        seq_list = pickle.load(fp)

    # Read image from file
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError('Failed to read image: ' + input_path)

    # For each sequence
    for s, seq in enumerate(seq_list):
        det = seq[0]

        # Crop image
        bbox = np.concatenate((det[:2], det[2:] - det[:2]))
        bbox = scale_bbox(bbox, crop_scale)
        img_cropped = crop_img(img, bbox)
        img_cropped = cv2.resize(img_cropped, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

        # Write cropped image to file
        out_img_name = os.path.splitext(os.path.basename(input_path))[0] + '_seq%02d%s' % (seq.id, out_postfix)
        out_img_path = os.path.join(output_dir, out_img_name)
        cv2.imwrite(out_img_path, img_cropped)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('crop_image_sequences')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='output directory')
    parser.add_argument('-c', '--cache', metavar='PATH',
                        help='path to sequence cache file')
    parser.add_argument('-sp', '--seq_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                        help='input sequence file postfix')
    parser.add_argument('-op', '--out_postfix', default='.jpg', metavar='POSTFIX',
                        help='input sequence file postfix')
    parser.add_argument('-r', '--resolution', default=256, type=int, metavar='N',
                        help='output video resolution (default: 256)')
    parser.add_argument('-cs', '--crop_scale', default=1.2, type=float, metavar='F',
                        help='crop scale relative to bounding box (default: 1.2)')
    args = parser.parse_args()
    main(args.input, args.output, args.cache, args.seq_postfix, args.out_postfix, args.resolution, args.crop_scale)
