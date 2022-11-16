import os
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsgan.models.hopenet import Hopenet
from fsgan.utils.utils import set_device
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from fsgan.utils.img_utils import rgb2tensor
from fsgan.utils.video_utils import Sequence
from fsgan.utils.img_utils import tensor2bgr    # Debug


def main(input_path, output_path=None, seq_postfix='_dsfd_seq.pkl', output_postfix='_dsfd_seq_lms_euler.pkl',
         pose_model_path='weights/hopenet_robust_alpha1.pkl', smooth_det=False, smooth_euler=False, gpus=None,
         cpu_only=False, batch_size=16):
    cache_path = os.path.splitext(input_path)[0] + seq_postfix
    output_path = os.path.splitext(input_path)[0] + output_postfix if output_path is None else output_path

    # Initialize device
    torch.set_grad_enabled(False)
    device, gpus = set_device(gpus, not cpu_only)

    # Load sequences from file
    with open(cache_path, "rb") as fp:  # Unpickling
        seq_list = pickle.load(fp)

    # Load pose model
    face_pose = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    face_pose.load_state_dict(checkpoint)
    face_pose.train(False)

    # Open input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Smooth sequence bounding boxes
    if smooth_det:
        for seq in seq_list:
            seq.smooth()

    # For each sequence
    total_detections = sum([len(s) for s in seq_list])
    pbar = tqdm(range(total_detections), unit='detections')
    for seq in seq_list:
        euler = []
        frame_cropped_tensor_list = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq.start_index)

        # For each detection bounding box in the current sequence
        for i, det in enumerate(seq.detections):
            ret, frame_bgr = cap.read()
            if frame_bgr is None:
                raise RuntimeError('Failed to read frame from video!')
            frame_rgb = frame_bgr[:, :, ::-1]

            # Crop frame
            bbox = np.concatenate((det[:2], det[2:] - det[:2]))
            bbox = scale_bbox(bbox, 1.2)
            frame_cropped_rgb = crop_img(frame_rgb, bbox)
            frame_cropped_rgb = cv2.resize(frame_cropped_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
            frame_cropped_tensor = rgb2tensor(frame_cropped_rgb).to(device)

            # Gather batches
            frame_cropped_tensor_list.append(frame_cropped_tensor)
            if len(frame_cropped_tensor_list) < batch_size and (i + 1) < len(seq):
                continue
            frame_cropped_tensor_batch = torch.cat(frame_cropped_tensor_list, dim=0)

            # Calculate euler angles
            curr_euler_batch = face_pose(frame_cropped_tensor_batch)  # Yaw, Pitch, Roll
            curr_euler_batch = curr_euler_batch.cpu().numpy()

            # For each prediction in the batch
            for b, curr_euler in enumerate(curr_euler_batch):
                # Add euler to list
                euler.append(curr_euler)

                # Render
                # render_img = tensor2bgr(frame_cropped_tensor_batch[b]).copy()
                # cv2.putText(render_img, '(%.2f, %.2f, %.2f)' % (curr_euler[0], curr_euler[1], curr_euler[2]), (15, 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.imshow('render_img', render_img)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

            # Clear lists
            frame_cropped_tensor_list.clear()

            pbar.update(len(frame_cropped_tensor_batch))

        # Add landmarks to sequence and optionally smooth them
        euler = np.array(euler)
        if smooth_euler:
            euler = smooth(euler)
        seq.euler = euler

    # Write final sequence list to file
    with open(output_path, "wb") as fp:  # Pickling
        pickle.dump(seq_list, fp)


def smooth(x, kernel_size=7):
    # Prepare smoothing kernel
    w = np.hamming(kernel_size)
    w /= w.sum()

    # Smooth euler
    x_padded = np.pad(x, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(x.shape[1]):
        x[:, i] = np.convolve(w, x_padded[:, i], mode='valid')

    return x


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('landmarks_sequences')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output directory')
    parser.add_argument('-sp', '--seq_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                        help='input sequence file postfix')
    parser.add_argument('-op', '--output_postfix', default='_dsfd_seq_lms_euler.pkl', metavar='POSTFIX',
                        help='output file postfix')
    parser.add_argument('-p', '--pose_model', default='weights/hopenet_robust_alpha1.pkl', metavar='PATH',
                        help='path to pose model file')
    parser.add_argument('-sd', '--smooth_det', action='store_true',
                        help='smooth the sequence detection bounding boxes')
    parser.add_argument('-se', '--smooth_euler', action='store_true',
                        help='smooth the sequence landmarks')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='force cpu only')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='batch size (default: 16)')
    args = parser.parse_args()
    main(args.input, args.output, args.seq_postfix, args.output_postfix, args.pose_model, args.smooth_det,
         args.smooth_euler, args.gpus, args.cpu_only, args.batch_size)
