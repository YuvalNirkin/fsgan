"""
Sequence statistics: Count, length, bounding boxes size.
"""
import os
from glob import glob
import pickle
from tqdm import tqdm


def extract_stats(cache_path):
    # Load sequences from file
    with open(cache_path, "rb") as fp:  # Unpickling
        seq_list = pickle.load(fp)

    if len(seq_list) == 0:
        return 0, 0., 0.

    # For each sequence
    len_sum, size_sum = 0., 0.
    for seq in seq_list:
        len_sum += len(seq)
        size_sum += seq.size_avg

    return len(seq_list), len_sum / len(seq_list), size_sum / len(seq_list)


def main(in_dir, out_path=None, postfix='_dsfd_seq.pkl'):
    out_path = os.path.join(in_dir, 'sequence_stats.txt') if out_path is None else out_path

    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory not exist: ' + in_dir)

    # Parse file paths
    input_query = os.path.join(in_dir, '*' + postfix)
    file_paths = sorted(glob(input_query))

    # For each file in the input directory with the specified postfix
    pbar = tqdm(file_paths, unit='files')
    count_sum, len_sum, size_sum = 0., 0., 0.
    vid_count = 0
    for i, file_path in enumerate(pbar):
        curr_count, curr_mean_len, curr_mean_size = extract_stats(file_path)
        if curr_count == 0:
            continue
        count_sum += curr_count
        len_sum += curr_mean_len
        size_sum += curr_mean_size
        vid_count += 1
        pbar.set_description('mean_count = %.1f, mean_len = %.1f, mean_size = %.1f, valid_vids = %d / %d' %
                             (count_sum / vid_count, len_sum / vid_count, size_sum / vid_count, vid_count, i + 1))

    # Write result to file
    if out_path is not None:
        with open(out_path, "w") as f:
            f.write('mean_count = %.1f\n' % (count_sum / vid_count))
            f.write('mean_len = %.1f\n' % (len_sum / vid_count))
            f.write('mean_size = %.1f\n' % (size_sum / vid_count))
            f.write('valid videos = %d / %d\n' % (vid_count, len(file_paths)))


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('detections2sequences')
    parser.add_argument('input', metavar='DIR',
                        help='input directory')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output directory')
    parser.add_argument('-p', '--postfix', metavar='POSTFIX', default='_dsfd_seq.pkl',
                        help='the files postfix to search the input directory for')
    args = parser.parse_args()
    main(args.input, args.output, args.postfix)
