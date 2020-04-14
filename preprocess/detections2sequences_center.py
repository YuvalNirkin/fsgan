import os
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from fsgan.utils.bbox_utils import batch_iou
from fsgan.utils.bbox_utils import smooth_bboxes
from fsgan.utils.video_utils import Sequence
# from fsgan.utils.video_utils import Sequence, smooth_detections_avg_center


def main(input_path, output_path=None, cache_path=None, iou_thresh=0.75, min_length=10, min_size=64, crop_scale=1.2,
         center_kernel=25, size_kernel=51, smooth=False, display=False, write_empty=False):
    cache_path = os.path.splitext(input_path)[0] + '_dsfd.pkl' if cache_path is None else cache_path
    output_path = os.path.splitext(input_path)[0] + '_dsfd_seq.pkl' if output_path is None else output_path
    min_length = 1 if os.path.splitext(input_path)[1] == '.jpg' else min_length

    # Validation
    if not os.path.isfile(cache_path):
        raise RuntimeError('Cache file does not exist: ' + cache_path)

    print('=> Extracting sequences from detections in video: "%s"...' % os.path.basename(input_path))

    # Load detections from file
    with open(cache_path, "rb") as fp:  # Unpickling
        det_list = pickle.load(fp)
    det_list.append(np.array([], dtype='float32'))  # Makes sure the final sequences are added to the seq_list

    # Open input video file
    if display:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError('Failed to read video: ' + input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # For each frame detection
    seq_list = []
    curr_seq_list = []
    # for i, frame_det in enumerate(det_list):    # Debug
    for i, frame_det in tqdm(enumerate(det_list), total=len(det_list)):
        frame_det = list(frame_det)
        if len(curr_seq_list) > 0:
            # For each sequence find matching detections
            keep_indices = np.full(len(curr_seq_list), False)
            for s, curr_seq in enumerate(curr_seq_list):
                if len(frame_det) > 0:
                    curr_seq_det_rep = np.repeat(np.expand_dims(curr_seq[-1], 0), len(frame_det), axis=0)
                    ious = batch_iou(curr_seq_det_rep, np.array(frame_det))
                    best_match_ind = ious.argmax()
                    if ious[best_match_ind] > iou_thresh:
                        # Match found
                        curr_seq.add(frame_det[best_match_ind])
                        del frame_det[best_match_ind]
                        keep_indices[s] = True

            # Remove unmatched sequences and add the suitable ones to the final sequence list
            if not np.all(keep_indices):
                seq_list += [seq for k, seq in enumerate(curr_seq_list)
                             if (not keep_indices[k]) and len(seq) >= min_length and
                             (seq.size_avg * crop_scale) >= min_size]
                curr_seq_list = [seq for k, seq in enumerate(curr_seq_list) if keep_indices[k]]

        # Add remaining detections to current sequences list as new sequences
        curr_seq_list += [Sequence(i, d) for d in frame_det]

        # Render current sequence list
        if display:
            ret, render_img = cap.read()
            if render_img is None:
                continue
            for j, seq in enumerate(curr_seq_list):
                rect = seq[-1]
                # cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[:2] + rect[2:]), (0, 0, 255), 1)
                cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 0, 255), 1)
                text_pos = (rect[:2] + np.array([0, -8])).astype('float32')
                cv2.putText(render_img, 'id: %d' % seq.id, tuple(text_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Reduce the n sequence ids to [0, ..., n - 1]
    ids = np.sort([seq.id for seq in seq_list])
    ids_map = {k: v for v, k in enumerate(ids)}
    for seq in seq_list:
        seq.id = ids_map[seq.id]

    # Smooth sequence bounding boxes or finalize detections (convert to numpy array)
    for seq in seq_list:
        if smooth:
            # seq.detections = smooth_detections_avg_center(seq.detections, center_kernel, size_kernel)
            seq.detections = smooth_bboxes(seq.detections, center_kernel, size_kernel)
        else:
            seq.finalize()

    # Write final sequence list to file
    if len(seq_list) > 0 or write_empty:
        with open(output_path, "wb") as fp:  # Pickling
            pickle.dump(seq_list, fp)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('detections2sequences_02')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', metavar='PATH',
                        help='output directory')
    parser.add_argument('-c', '--cache', metavar='PATH',
                        help='path to detections cache file')
    parser.add_argument('-it', '--iou_thresh', default=0.75, type=float,
                        metavar='F', help='IOU threshold')
    parser.add_argument('-ml', '--min_length', default=10, type=int,
                        metavar='N', help='minimum sequence length')
    parser.add_argument('-ms', '--min_size', default=64, type=int,
                        metavar='N', help='minimum sequence average bounding box size')
    parser.add_argument('-cs', '--crop_scale', default=1.2, type=float, metavar='F',
                        help='crop scale relative to bounding box (default: 1.2)')
    parser.add_argument('-ck', '--center_kernel', default=25, type=int,
                        metavar='N', help='center average kernel size')
    parser.add_argument('-sk', '--size_kernel', default=51, type=int,
                        metavar='N', help='size average kernel size')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='smooth the sequence bounding boxes')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    parser.add_argument('-we', '--write_empty', action='store_true',
                        help='write empty sequence lists to file')
    args = parser.parse_args()
    main(args.input, args.output, args.cache, args.iou_thresh, args.min_length, args.min_size, args.crop_scale,
         args.center_kernel, args.size_kernel, args.smooth, args.display, args.write_empty)
