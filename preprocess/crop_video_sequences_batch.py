import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from itertools import groupby
import numpy as np
from fsgan.preprocess.crop_video_sequences import main as crop_video_sequences


def parse_videos(root):
    vid_rel_paths = []
    for r, d, f in os.walk(root):
        for file in f:
            if file.endswith('.mp4'):
                vid_rel_paths.append(os.path.join(os.path.relpath(r, root), file).replace('\\', '/'))

    return vid_rel_paths


def process_video(input, cache_postfix='_dsfd_seq.pkl', resolution=256, crop_scale=1.2, select='all'):
    file_path, out_dir = input[0], input[1]
    filename = os.path.basename(file_path)
    curr_out_cache_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '_seq00' + cache_postfix)
    if os.path.exists(curr_out_cache_path):
        return True

    # Process video
    crop_video_sequences(file_path, out_dir, None, cache_postfix, resolution, crop_scale, select, disable_tqdm=True)
    return True


def main(root, output_dir, file_lists=None, cache_postfix='_dsfd_seq.pkl', resolution=256, crop_scale=2.0, workers=4,
         select='all'):
    # Validation
    if not os.path.isdir(root):
        raise RuntimeError('root directory does not exist: ' + root)
    if not os.path.isdir(output_dir):
        raise RuntimeError('Output directory does not exist: ' + output_dir)

    # Parse files from directory or file lists (if specified)
    if file_lists is None:
        vid_rel_paths = parse_videos(root)
    else:
        vid_rel_paths = []
        for file_list in file_lists:
            vid_rel_paths.append(np.loadtxt(os.path.join(root, file_list), dtype=str))
        vid_rel_paths = np.concatenate(vid_rel_paths)

    vid_out_dirs = [os.path.join(output_dir, os.path.split(p)[0]) for p in vid_rel_paths]
    vid_paths = [os.path.join(root, p) for p in vid_rel_paths]

    # Make directory structure
    for out_dir in vid_out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # Process all videos
    f = partial(process_video, cache_postfix=cache_postfix, resolution=resolution, crop_scale=crop_scale, select=select)
    with Pool(workers) as p:
        list(tqdm(p.imap(f, zip(vid_paths, vid_out_dirs)), total=len(vid_paths)))

    # Parse generated sequence videos
    vid_seq_rel_paths = parse_videos(output_dir)
    vid_seq_keys, vid_seq_groups = zip(*[(key, list(group)) for key, group in
                                         groupby(vid_seq_rel_paths, lambda p: (p[:-10] + '.mp4'))])
    vid_seq_groups = np.array(vid_seq_groups)

    for file_list in file_lists:
        # Adjust file list to generated sequence videos
        list_rel_paths = np.loadtxt(os.path.join(root, file_list), dtype=str)
        _, indices, _ = np.intersect1d(vid_seq_keys, list_rel_paths, return_indices=True)
        list_seq_rel_paths = np.concatenate(vid_seq_groups[indices])

        # Write output list to file
        np.savetxt(os.path.join(output_dir, file_list), list_seq_rel_paths, fmt='%s')


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument('root', metavar='DIR',
                        help='root directory')
    parser.add_argument('-o', '--output', metavar='DIR', required=True,
                        help='output directory')
    parser.add_argument('-fl', '--file_lists', metavar='PATH', nargs='+',
                        help='file lists')
    parser.add_argument('-cp', '--cache_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                        help='cache file postfix')
    parser.add_argument('-r', '--resolution', default=256, type=int, metavar='N',
                        help='output video resolution (default: 256)')
    parser.add_argument('-cs', '--crop_scale', type=float, metavar='F', default=2.0,
                        help='crop scale relative to detection bounding box')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-s', '--select', default='all', metavar='STR',
                        help='selection method [all|longest]')
    args = parser.parse_args()
    main(args.root, args.output, args.file_lists, args.cache_postfix, args.resolution, args.crop_scale, args.workers,
         args.select)
