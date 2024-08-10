import os
import sys
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from fsgan.utils.video_utils import Sequence


def main(input_path, output_dir=None, cache_path=None, seq_postfix='_dsfd_seq.pkl', resolution=256, crop_scale=2.0,
         select='all', disable_tqdm=False, encoder_codec='mp4v'):
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

    print('=> Cropping video sequences from video: "%s"...' % os.path.basename(input_path))

    # Load sequences from file
    with open(cache_path, "rb") as fp:  # Unpickling
        seq_list = pickle.load(fp)

    # Select sequences
    if select == 'longest':
        selected_seq_index = np.argmax([len(s) for s in seq_list])
        seq = seq_list[selected_seq_index]
        seq.id = 0
        seq_list = [seq]

    # Open input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # For each sequence initialize output video file
    out_vids = []
    fourcc = cv2.VideoWriter_fourcc(*encoder_codec)
    for seq in seq_list:
        curr_vid_name = os.path.splitext(os.path.basename(input_path))[0] + '_seq%02d.mp4' % seq.id
        curr_vid_path = os.path.join(output_dir, curr_vid_name)
        out_vids.append(cv2.VideoWriter(curr_vid_path, fourcc, fps, (resolution, resolution)))

    # For each frame in the target video
    cropped_detections = [[] for seq in seq_list]
    cropped_landmarks = [[] for seq in seq_list]
    pbar = range(total_frames) if disable_tqdm else tqdm(range(total_frames), file=sys.stdout)
    for i in pbar:
        ret, frame = cap.read()
        if frame is None:
            continue

        # For each sequence
        for s, seq in enumerate(seq_list):
            if i < seq.start_index or (seq.start_index + len(seq) - 1) < i:
                continue
            det = seq[i - seq.start_index]

            # Crop frame
            bbox = np.concatenate((det[:2], det[2:] - det[:2]))
            bbox = scale_bbox(bbox, crop_scale)
            frame_cropped = crop_img(frame, bbox)
            frame_cropped = cv2.resize(frame_cropped, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

            # Write cropped frame to output video
            out_vids[s].write(frame_cropped)

            # Add cropped detection to list
            orig_size = bbox[2:]
            axes_scale = np.array([resolution, resolution]) / orig_size
            det[:2] -= bbox[:2]
            det[2:] -= bbox[:2]
            det[:2] *= axes_scale
            det[2:] *= axes_scale
            cropped_detections[s].append(det)

            # Add cropped landmarks to list
            if hasattr(seq, 'landmarks'):
                curr_landmarks = seq.landmarks[i - seq.start_index]
                curr_landmarks[:, :2] -= bbox[:2]

                # 3D landmarks case
                if curr_landmarks.shape[1] == 3:
                    axes_scale = np.append(axes_scale, axes_scale.mean())

                curr_landmarks *= axes_scale
                cropped_landmarks[s].append(curr_landmarks)

    # For each sequence write cropped sequence to file
    for s, seq in enumerate(seq_list):
        # seq.detections = np.array(cropped_detections[s])
        # if hasattr(seq, 'landmarks'):
        #     seq.landmarks = np.array(cropped_landmarks[s])
        # seq.start_index = 0

        # TODO: this is a hack to change class type (remove this later)
        out_seq = Sequence(0)
        out_seq.detections = np.array(cropped_detections[s])
        if hasattr(seq, 'landmarks'):
            out_seq.landmarks = np.array(cropped_landmarks[s])
        out_seq.id, out_seq.obj_id, out_seq.size_avg = seq.id, seq.obj_id, seq.size_avg

        # Write to file
        curr_out_name = os.path.splitext(os.path.basename(input_path))[0] + '_seq%02d%s' % (out_seq.id, seq_postfix)
        curr_out_path = os.path.join(output_dir, curr_out_name)
        with open(curr_out_path, "wb") as fp:  # Pickling
            pickle.dump([out_seq], fp)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('crop_video_sequences')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='output directory')
    parser.add_argument('-c', '--cache', metavar='PATH',
                        help='path to sequence cache file')
    parser.add_argument('-sp', '--seq_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                        help='input sequence file postfix')
    parser.add_argument('-r', '--resolution', default=256, type=int, metavar='N',
                        help='output video resolution (default: 256)')
    parser.add_argument('-cs', '--crop_scale', default=2.0, type=float, metavar='F',
                        help='crop scale relative to bounding box (default: 2.0)')
    parser.add_argument('-s', '--select', default='all', metavar='STR',
                        help='selection method [all|longest]')
    parser.add_argument('-dt', '--disable_tqdm', dest='disable_tqdm', action='store_true',
                          help='if specified disables tqdm progress bar')
    parser.add_argument('-ec', '--encoder_codec', default='mp4v', metavar='STR',
                        help='encoder codec code')
    args = parser.parse_args()
    main(args.input, args.output, args.cache, args.seq_postfix, args.resolution, args.crop_scale, args.select,
         args.disable_tqdm, args.encoder_codec)
