import fsgan.preprocess.video_landmark_keyframes as video_landmark_keyframes
import os
from glob import glob
import traceback
import logging


def main(in_dir, out_dir, pose_model_path, min_size=200, frame_sample_ratio=0.1, min_samples=5, sample_limit=100,
         min_res=720, start_index=0, end_index=None):
    vid_paths = sorted(glob(os.path.join(in_dir, '*.mp4')))
    if end_index is None:
        end_index = len(vid_paths) - 1

    # For each video file
    for vid_path in vid_paths[start_index:end_index + 1]:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        curr_out_dir = os.path.join(out_dir, vid_name)

        if os.path.exists(curr_out_dir):
            print('Skipping "%s"' % vid_name)
            continue
        else:
            print('Processing "%s"...' % vid_name)
            os.mkdir(curr_out_dir)

        # Process video
        try:
            video_landmark_keyframes.main(vid_path, curr_out_dir, pose_model_path, min_size, frame_sample_ratio,
                                          min_samples, sample_limit, min_res)
        except Exception as e:
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_keyframes_batch')
    parser.add_argument('input', metavar='DIR', help='input directory')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    parser.add_argument('-pm', '--pose_model', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-mb', '--min_bbox_size', default=200, type=int, metavar='N',
                        help='minimum bounding box size')
    parser.add_argument('-fs', '--frame_samples', default=0.1, type=float, metavar='F',
                        help='the number of samples per frame')
    parser.add_argument('-ms', '--min_samples', default=5, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-sl', '--sample_limit', default=100, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-mr', '--min_res', default=720, type=int, metavar='N',
                        help='minimum video resolution (height pixels)')
    parser.add_argument('-si', '--start_index', default=0, type=int, metavar='N',
                        help='starting video index')
    parser.add_argument('-ei', '--end_index', default=None, type=int, metavar='N',
                        help='end video index')
    args = parser.parse_args()
    main(args.input, args.output, args.pose_model, args.min_bbox_size, args.frame_samples, args.min_samples,
         args.sample_limit, args.min_res, args.start_index, args.end_index)
