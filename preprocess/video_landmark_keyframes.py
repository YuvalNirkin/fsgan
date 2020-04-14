import os
import face_alignment
import cv2
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import random
import utils
from fsgan.models.hopenet import Hopenet
import torch


def uniform_sample_with_min_dist(limit, n, min_dist):
    slack = limit - 1 - min_dist * (n - 1)
    if slack < 1:
        return np.array([], dtype=int)
    steps = random.randint(0, slack)

    increments = np.hstack([np.ones((steps,)), np.zeros((n,))])
    np.random.shuffle(increments)

    locs = np.argwhere(increments == 0).flatten()
    samples = np.cumsum(increments)[locs] + min_dist * np.arange(0, n)

    return np.array(samples, dtype=int)


def crop_img(img, bbox):
    min_xy = bbox[:2]
    max_xy = bbox[:2] + bbox[2:] - 1
    min_xy[0] = min_xy[0] if min_xy[0] >= 0 else 0
    min_xy[1] = min_xy[1] if min_xy[1] >= 0 else 0
    max_xy[0] = max_xy[0] if max_xy[0] < img.shape[1] else (img.shape[1] - 1)
    max_xy[1] = max_xy[1] if max_xy[1] < img.shape[0] else (img.shape[0] - 1)

    return img[min_xy[1]:max_xy[1] + 1, min_xy[0]:max_xy[0] + 1]


def scale_bbox(bbox, scale=1.35, square=True):
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def main(video_path, out_dir, pose_model_path, min_size=200, frame_sample_ratio=0.1, min_samples=5, sample_limit=100,
         min_res=720):
    cache_file = os.path.splitext(video_path)[0] + '.pkl'

    # Initialize models
    torch.set_grad_enabled(False)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)
    device, gpus = utils.set_device()

    # Initialize pose
    Gp = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    Gp.load_state_dict(checkpoint)
    Gp.train(False)

    # Validate video resolution
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if width < min_res or height < min_res:
        return
    cap.release()

    # Extract landmarks, bounding boxes, euler angles, and 3D face landmarks
    frame_indices, landmarks, bboxes, eulers, landmarks_3d = \
        utils.extract_landmarks_bboxes_euler_3d_from_video(video_path, Gp, fa, fa_3d, device=device)
    if frame_indices.size == 0:
        return

    ### Debug ###
    # cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in tqdm(range(total_frames)):
    #     ret, frame = cap.read()
    #     if i not in frame_indices:
    #         continue
    #     for point in landmarks[i]:
    #         cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
    #     bbox = bboxes[i]
    #     cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:]), (0, 255, 0), 1)
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(0)
    #############

    # Filter by bounding boxes size
    bboxes_sizes = bboxes[:, 2:]
    max_bbox_sizes = bboxes_sizes.max(axis=1)
    filter_map = max_bbox_sizes > min_size
    frame_indices = frame_indices[filter_map]
    landmarks = landmarks[filter_map]
    bboxes = bboxes[filter_map]
    eulers = eulers[filter_map]
    landmarks_3d = landmarks_3d[filter_map]

    if frame_indices.size == 0:
        return

    # Normalize landmarks and convert them to descriptor vectors
    landmark_descs = landmarks.copy()
    for lms in landmark_descs:
        lms -= np.mean(lms, axis=0)
        lms /= (np.max(lms) - np.min(lms))
    landmark_descs = landmark_descs.reshape(landmark_descs.shape[0], -1)                # Reshape landmarks to vectors
    landmark_descs -= np.mean(landmark_descs, axis=0)  # Normalize landmarks

    # Find a diverse sample of frames
    sample_size = min(int(np.round(len(frame_indices) * frame_sample_ratio)), sample_limit)
    if sample_size < min_samples:
        return
    max_mean_dist = 0.
    best_sample_indices = None
    for i in range(20000):
        sample_indices = uniform_sample_with_min_dist(len(frame_indices), sample_size, 4)
        landmark_desc_samples = landmark_descs[sample_indices]
        dist = euclidean_distances(landmark_desc_samples, landmark_desc_samples)
        mean_dist = np.mean(dist)
        if mean_dist > max_mean_dist:
            max_mean_dist = mean_dist
            best_sample_indices = sample_indices

    selected_frame_map = dict(zip(frame_indices[best_sample_indices], best_sample_indices))

    # Write frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if i in selected_frame_map:

            # Crop frame
            bbox = bboxes[selected_frame_map[i]]
            scaled_bbox = scale_bbox(bboxes[selected_frame_map[i]], scale=2.0, square=True)
            cropped_frame = crop_img(frame, scaled_bbox)

            ### Debug ###
            # for point in landmarks[selected_frame_map[i]]:
            #     cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
            # bbox = bboxes[selected_frame_map[i]]
            # cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:]), (0, 255, 0), 1)
            # cv2.rectangle(frame, tuple(scaled_bbox[:2]), tuple(scaled_bbox[:2] + scaled_bbox[2:]), (0, 0, 255), 1)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            #############

            # Adjust output landmarks and bounding boxes
            landmarks[selected_frame_map[i]] -= scaled_bbox[:2]
            landmarks_3d[selected_frame_map[i]][:, :2] -= scaled_bbox[:2]
            bboxes[selected_frame_map[i]][:2] -= scaled_bbox[:2]

            ### Debug ###
            # for point in landmarks[selected_frame_map[i]]:
            #     cv2.circle(cropped_frame, (point[0], point[1]), 2, (0, 0, 255), -1)
            # bbox = bboxes[selected_frame_map[i]]
            # cv2.rectangle(cropped_frame, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:]), (0, 255, 0), 1)
            # cv2.imshow('cropped_frame', cropped_frame)
            # cv2.waitKey(0)
            #############

            # Write frame to file
            cv2.imwrite(os.path.join(out_dir, 'frame_%04d.jpg' % i), cropped_frame)
    cap.release()

    # Write landmarks and bounding boxes
    np.save(out_dir + '_landmarks.npy', landmarks[best_sample_indices])
    np.save(out_dir + '_bboxes.npy', bboxes[best_sample_indices])
    np.save(out_dir + '_eulers.npy', eulers[best_sample_indices])
    np.save(out_dir + '_landmarks_3d.npy', landmarks_3d[best_sample_indices])


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_keyframes')
    parser.add_argument('video_path', help='path to video file')
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
    args = parser.parse_args()
    main(args.video_path, args.output, args.pose_model, args.min_bbox_size, args.frame_samples, args.min_samples,
         args.sample_limit, args.min_res)

