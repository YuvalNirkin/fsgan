import os
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from fsgan.utils.video_utils import Sequence


def main(input_path, output_path=None, postfix='_dsfd_seq.pkl', smooth=False, fps=None):
    cache_path = os.path.splitext(input_path)[0] + postfix
    # output_path = os.path.splitext(input_path)[0] + '.mp4' if output_path is None else output_path

    # Load sequences from file
    with open(cache_path, "rb") as fp:  # Unpickling
        seq_list = pickle.load(fp)

    # Open input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
    input_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            output_filename = os.path.basename(input_path)
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'x264')
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (input_vid_width, input_vid_height))
    else:
        out_vid = None

    # Smooth sequence bounding boxes
    if smooth:
        for seq in seq_list:
            seq.smooth()

    # For each frame in the target video
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            continue

        # For each sequence
        render_img = frame
        for seq in seq_list:
            if i < seq.start_index or (seq.start_index + len(seq) - 1) < i:
                continue
            rect = seq[i - seq.start_index]
            cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 255, 0), 1)

            if hasattr(seq, 'landmarks'):
                landmarks = seq.landmarks[i - seq.start_index]
                for point in np.round(landmarks).astype(int):
                    cv2.circle(render_img, (point[0], point[1]), 1, (0, 0, 255), -1)

            text_pos = (rect[:2] + np.array([0, -8])).astype('float32')
            cv2.putText(render_img, 'id: %d' % seq.id, tuple(text_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Render
        if out_vid is not None:
            out_vid.write(render_img)
        cv2.imshow('render_img', render_img)
        delay = np.round(1000.0 / fps).astype(int) if fps > 1e-5 else 0
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('detections2sequences')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output directory')
    parser.add_argument('-p', '--postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                        help='input sequence file postfix')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='smooth the sequence bounding boxes')
    parser.add_argument('-f', '--fps', type=float, metavar='F',
                        help='force video fps, set 0 to pause after each frame')
    args = parser.parse_args()
    main(args.input, args.output, args.postfix, args.smooth, args.fps)
