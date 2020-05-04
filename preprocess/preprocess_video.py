""" Video preprocessing.

This script implements all preprocessing required for both training and inference.
The preprocessing information will be cached in a directory by the file's name without the extension,
residing in the same directory as the file. The information contains: face detections, face sequences,
and cropped videos per sequence. In addition for each cropped video, the corresponding pose, landmarks, and
segmentation masks will be computed and cached.
"""

import os
import argparse
import sys
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from face_detection_dsfd.face_detector import FaceDetector
from fsgan.utils.utils import set_device, load_model
from fsgan.preprocess.detections2sequences_center import main as detections2sequences_main
from fsgan.preprocess.crop_video_sequences import main as crop_video_sequences_main
from fsgan.preprocess.crop_image_sequences import main as crop_image_sequences_main
from fsgan.datasets.video_inference_dataset import VideoInferenceDataset
import fsgan.datasets.img_landmarks_transforms as img_landmarks_transforms
from fsgan.datasets.img_landmarks_transforms import Resize, ToTensor
from fsgan.utils.temporal_smoothing import TemporalSmoothing
from fsgan.utils.landmarks_utils import LandmarksHeatMapEncoder, smooth_landmarks_98pts
from fsgan.utils.seg_utils import encode_binary_mask, remove_inner_mouth
from fsgan.utils.batch import main as batch


base_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

general = base_parser.add_argument_group('general')
general.add_argument('-r', '--resolution', default=256, type=int, metavar='N',
                     help='finest processing resolution')
general.add_argument('-cs', '--crop_scale', default=1.2, type=float, metavar='F',
                     help='crop scale relative to bounding box')
general.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                     help='list of gpu ids to use')
general.add_argument('--cpu_only', action='store_true',
                     help='force cpu only')
general.add_argument('-d', '--display', action='store_true',
                     help='display the rendering')
general.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                     help='verbose level')
general.add_argument('-ec', '--encoder_codec', default='avc1', metavar='STR',
                     help='encoder codec code')

detection = base_parser.add_argument_group('detection')
detection.add_argument('-dm', '--detection_model', metavar='PATH', default='../weights/WIDERFace_DSFD_RES152.pth',
                       help='path to face detection model')
detection.add_argument('-db', '--det_batch_size', default=8, type=int, metavar='N',
                       help='detection batch size')
detection.add_argument('-dp', '--det_postfix', default='_dsfd.pkl', metavar='POSTFIX',
                       help='detection file postfix')

sequences = base_parser.add_argument_group('sequences')
sequences.add_argument('-it', '--iou_thresh', default=0.75, type=float,
                       metavar='F', help='IOU threshold')
sequences.add_argument('-ml', '--min_length', default=10, type=int,
                       metavar='N', help='minimum sequence length')
sequences.add_argument('-ms', '--min_size', default=64, type=int,
                       metavar='N', help='minimum sequence average bounding box size')
sequences.add_argument('-ck', '--center_kernel', default=25, type=int,
                       metavar='N', help='center average kernel size')
sequences.add_argument('-sk', '--size_kernel', default=51, type=int,
                       metavar='N', help='size average kernel size')
sequences.add_argument('-dsd', '--disable_smooth_det', dest='smooth_det', action='store_false',
                       help='disable smoothing the detection bounding boxes')
sequences.add_argument('-sp', '--seq_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                       help='sequence file postfix')
sequences.add_argument('-we', '--write_empty', action='store_true',
                       help='write empty sequence lists to file')

pose = base_parser.add_argument_group('pose')
pose.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pth', metavar='PATH',
                       help='path to face pose model file')
pose.add_argument('-pb', '--pose_batch_size', default=128, type=int, metavar='N',
                       help='pose batch size')
pose.add_argument('-pp', '--pose_postfix', default='_pose.npz', metavar='POSTFIX',
                       help='pose file postfix')
pose.add_argument('-cp', '--cache_pose', action='store_true',
                  help='Toggle whether to cache pose')
pose.add_argument('-cf', '--cache_frontal', action='store_true',
                  help='Toggle whether to cache frontal images for each sequence')
pose.add_argument('-spo', '--smooth_poses', default=5, type=int, metavar='N',
                  help='poses temporal smoothing kernel size')

landmarks = base_parser.add_argument_group('landmarks')
landmarks.add_argument('-lm', '--lms_model', default='../weights/hr18_wflw_landmarks.pth', metavar='PATH',
                       help='landmarks model')
landmarks.add_argument('-lb', '--lms_batch_size', default=64, type=int, metavar='N',
                       help='landmarks batch size')
landmarks.add_argument('-lp', '--landmarks_postfix', default='_lms.npz', metavar='POSTFIX',
                       help='landmarks file postfix')
landmarks.add_argument('-cl', '--cache_landmarks', action='store_true',
                       help='Toggle whether to cache landmarks')
landmarks.add_argument('-sl', '--smooth_landmarks', default=7, type=int, metavar='N',
                       help='landmarks temporal smoothing kernel size')

segmentation = base_parser.add_argument_group('segmentation')
segmentation.add_argument('-sm', '--seg_model', default='../weights/celeba_unet_256_1_2_segmentation_v2.pth',
                          metavar='PATH', help='segmentation model')
segmentation.add_argument('-sb', '--seg_batch_size', default=32, type=int, metavar='N',
                          help='segmentation batch size')
segmentation.add_argument('-sep', '--segmentation_postfix', default='_seg.pkl', metavar='POSTFIX',
                          help='segmentation file postfix')
segmentation.add_argument('-cse', '--cache_segmentation', action='store_true',
                          help='Toggle whether to cache segmentation')
segmentation.add_argument('-sse', '--smooth_segmentation', default=5, type=int, metavar='N',
                          help='segmentation temporal smoothing kernel size')
segmentation.add_argument('-srm', '--seg_remove_mouth', action='store_true',
                          help='if true, the inner part of the mouth will be removed from the segmentation')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 parents=[base_parser])
parser.add_argument('input', metavar='VIDEO', nargs='+',
                    help='path to input video')
parser.add_argument('-o', '--output', metavar='DIR',
                    help='output directory')
d = parser.get_default


class VideoProcessBase(object):
    def __init__(self, resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
         cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
         # Detection arguments:
         detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
         # Sequence arguments:
         iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
         center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
         seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
         # Pose arguments:
         pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
         cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
         # Landmarks arguments:
         lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
         cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
         # Segmentation arguments:
         seg_model=d('seg_model'), seg_batch_size=d('seg_batch_size'), segmentation_postfix=d('segmentation_postfix'),
         cache_segmentation=d('cache_segmentation'), smooth_segmentation=d('smooth_segmentation'),
         seg_remove_mouth=d('seg_remove_mouth')):
        # General
        self.resolution = resolution
        self.crop_scale = crop_scale
        self.display = display
        self.verbose = verbose

        # Detection
        self.face_detector = FaceDetector(det_postfix, detection_model, gpus, det_batch_size, display)
        self.det_postfix = det_postfix

        # Sequences
        self.iou_thresh = iou_thresh
        self.min_length = min_length
        self.min_size = min_size
        self.center_kernel = center_kernel
        self.size_kernel = size_kernel
        self.smooth_det = smooth_det
        self.seq_postfix = seq_postfix
        self.write_empty = write_empty

        # Pose
        self.pose_batch_size = pose_batch_size
        self.pose_postfix = pose_postfix
        self.cache_pose = cache_pose
        self.cache_frontal = cache_frontal
        self.smooth_poses = smooth_poses

        # Landmarks
        self.smooth_landmarks = smooth_landmarks
        self.landmarks_postfix = landmarks_postfix
        self.cache_landmarks = cache_landmarks
        self.lms_batch_size = lms_batch_size

        # Segmentation
        self.smooth_segmentation = smooth_segmentation
        self.segmentation_postfix = segmentation_postfix
        self.cache_segmentation = cache_segmentation
        self.seg_batch_size = seg_batch_size
        self.seg_remove_mouth = seg_remove_mouth and cache_landmarks

        # Initialize device
        torch.set_grad_enabled(False)
        self.device, self.gpus = set_device(gpus, not cpu_only)

        # Load models
        self.face_pose = load_model(pose_model, 'face pose', self.device) if cache_pose else None
        self.L = load_model(lms_model, 'face landmarks', self.device) if cache_landmarks else None
        self.S = load_model(seg_model, 'face segmentation', self.device) if cache_segmentation else None

        # Initialize heatmap encoder
        self.heatmap_encoder = LandmarksHeatMapEncoder().to(self.device)

        # Initialize normalization tensors
        # Note: this is necessary because of the landmarks model
        self.img_mean = torch.as_tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.as_tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.context_mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.context_std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.face_pose = nn.DataParallel(self.face_pose, self.gpus) if self.face_pose is not None else None
            self.L = nn.DataParallel(self.L, self.gpus) if self.L is not None else None
            self.S = nn.DataParallel(self.S, self.gpus) if self.S is not None else None

        # Initialize temportal smoothing
        if smooth_segmentation > 0:
            self.smooth_seg = TemporalSmoothing(3, smooth_segmentation).to(self.device)
        else:
            self.smooth_seg = None

        # Initialize output videos format
        self.encoder_codec = encoder_codec
        self.fourcc = cv2.VideoWriter_fourcc(*encoder_codec)

    def process_pose(self, input_path, output_dir, seq_file_path):
        if not self.cache_pose:
            return
        input_path_no_ext, input_ext = os.path.splitext(input_path)

        # Load sequences from file
        with open(seq_file_path, "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # Initialize transforms
        img_transforms = img_landmarks_transforms.Compose([
            Resize(224), ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # For each sequence
        for seq in seq_list:
            curr_vid_name = os.path.basename(input_path_no_ext) + '_seq%02d%s' % (seq.id, input_ext)
            curr_vid_path = os.path.join(output_dir, curr_vid_name)
            curr_pose_path = os.path.splitext(curr_vid_path)[0] + self.pose_postfix

            if os.path.isfile(curr_pose_path):
                continue
            print('=> Computing face poses for video: "%s"...' % curr_vid_name)

            # Initialize input video
            in_vid = VideoInferenceDataset(curr_vid_path, transform=img_transforms)
            in_vid_loader = DataLoader(in_vid, batch_size=self.pose_batch_size, num_workers=1, pin_memory=True,
                                       drop_last=False, shuffle=False)

            # For each batch of frames in the input video
            seq_poses = []
            for i, frame in enumerate(tqdm(in_vid_loader, unit='batches', file=sys.stdout)):
                frame = frame.to(self.device)
                poses = self.face_pose(frame).div_(99.)  # Yaw, Pitch, Roll
                seq_poses.append(poses.cpu().numpy())
            seq_poses = np.concatenate(seq_poses)

            # Save poses to file
            seq_poses_smoothed = smooth_poses(seq_poses, self.smooth_poses)
            np.savez_compressed(curr_pose_path, poses=seq_poses, poses_smoothed=seq_poses_smoothed)

    def extract_frontal_images(self, input_path, output_dir, seq_file_path, out_postfix='.jpg', resolution=None):
        if not self.cache_frontal:
            return

        # Load sequences from file
        with open(seq_file_path, "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # For each sequence
        for seq in seq_list:
            curr_vid_name = os.path.splitext(os.path.basename(input_path))[0] + '_seq%02d.mp4' % seq.id
            curr_vid_path = os.path.join(output_dir, curr_vid_name)
            curr_pose_path = os.path.splitext(curr_vid_path)[0] + self.pose_postfix
            curr_frontal_path = os.path.splitext(curr_vid_path)[0] + out_postfix

            if os.path.isfile(curr_frontal_path):
                continue

            # Open current video file
            vid = cv2.VideoCapture(curr_vid_path)
            if not vid.isOpened():
                raise RuntimeError('Failed to read video: ' + curr_vid_path)

            # Load current sequence poses
            curr_poses = np.load(curr_pose_path)['poses_smoothed']

            # Read frontal image from video
            frontal_index = np.argmin(np.linalg.norm(curr_poses, axis=1))
            vid.set(cv2.CAP_PROP_POS_FRAMES, frontal_index)
            ret, frontal_bgr = vid.read()

            # Resize image
            if resolution is not None:
                frontal_bgr = cv2.resize(frontal_bgr, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

            # Write frontal image to file
            cv2.imwrite(curr_frontal_path, frontal_bgr)

    def process_landmarks(self, input_path, output_dir, seq_file_path):
        if not self.cache_landmarks:
            return
        input_path_no_ext, input_ext = os.path.splitext(input_path)

        # Load sequences from file
        with open(seq_file_path, "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # Initialize transforms
        img_transforms = img_landmarks_transforms.Compose([
            ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # For each sequence
        for seq in seq_list:
            curr_vid_name = os.path.basename(input_path_no_ext) + '_seq%02d%s' % (seq.id, input_ext)
            curr_vid_path = os.path.join(output_dir, curr_vid_name)
            curr_lms_path = os.path.splitext(curr_vid_path)[0] + self.landmarks_postfix

            if os.path.isfile(curr_lms_path):
                continue
            print('=> Computing face landmarks for video: "%s"...' % curr_vid_name)

            # Initialize input video
            in_vid = VideoInferenceDataset(curr_vid_path, transform=img_transforms)
            in_vid_loader = DataLoader(in_vid, batch_size=self.lms_batch_size, num_workers=1, pin_memory=True,
                                       drop_last=False, shuffle=False)

            # For each batch of frames in the input video
            seq_landmarks = []
            for i, frame in enumerate(tqdm(in_vid_loader, unit='batches', file=sys.stdout)):
                frame = frame.to(self.device)
                H = self.L(frame)
                landmarks = self.heatmap_encoder(H)
                seq_landmarks.append(landmarks.cpu().numpy())
            seq_landmarks = np.concatenate(seq_landmarks)

            # Save landmarks to file
            seq_landmarks_smoothed = smooth_landmarks_98pts(seq_landmarks, self.smooth_landmarks)
            np.savez_compressed(curr_lms_path, landmarks=seq_landmarks, landmarks_smoothed=seq_landmarks_smoothed)

    def process_segmentation(self, input_path, output_dir, seq_file_path):
        if not self.cache_segmentation:
            return
        input_path_no_ext, input_ext = os.path.splitext(input_path)

        # Load sequences from file
        with open(seq_file_path, "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # Initialize transforms
        img_transforms = img_landmarks_transforms.Compose([
            ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # For each sequence
        for seq in seq_list:
            curr_vid_name = os.path.basename(input_path_no_ext) + '_seq%02d%s' % (seq.id, input_ext)
            curr_vid_path = os.path.join(output_dir, curr_vid_name)
            curr_seg_path = os.path.splitext(curr_vid_path)[0] + self.segmentation_postfix

            if self.seg_remove_mouth:
                curr_lms_path = os.path.splitext(curr_vid_path)[0] + self.landmarks_postfix
                landmarks = np.load(curr_lms_path)['landmarks_smoothed']
                frame_count = 0

            if os.path.isfile(curr_seg_path):
                continue
            print('=> Computing face segmentation for video: "%s"...' % curr_vid_name)

            # Initialize input video
            in_vid = VideoInferenceDataset(curr_vid_path, transform=img_transforms)
            in_vid_loader = DataLoader(in_vid, batch_size=self.seg_batch_size, num_workers=1, pin_memory=True,
                                       drop_last=False, shuffle=False)

            # For each batch of frames in the input video
            pbar = tqdm(in_vid_loader, unit='batches', file=sys.stdout)
            prev_segmentation = None
            r = self.smooth_seg.kernel_radius
            encoded_segmentations = []
            pad_prev, pad_next = r, r   # This initialization is only relevant if there is a leftover from last batch
            for i, frame in enumerate(pbar):
                frame = frame.to(self.device)

                # Compute segmentation
                raw_segmentation = self.S(frame)
                segmentation = torch.cat((prev_segmentation, raw_segmentation), dim=0) \
                    if prev_segmentation is not None else raw_segmentation
                if segmentation.shape[0] > r:
                    pad_prev, pad_next = r if prev_segmentation is None else 0, min(r, self.seg_batch_size - frame.shape[0])
                    segmentation = self.smooth_seg(segmentation, pad_prev=pad_prev, pad_next=pad_next)

                    # Note: the pad_next value here is only relevant if there is a leftover from last batch
                    prev_segmentation = raw_segmentation[-(r * 2 - pad_next):]

                mask = segmentation.argmax(1) == 1

                # Encode segmentation
                for b in range(mask.shape[0]):
                    curr_mask = mask[b].cpu().numpy()
                    if self.seg_remove_mouth:
                        curr_mask = remove_inner_mouth(curr_mask, landmarks[frame_count])
                        frame_count += 1
                    encoded_segmentations.append(encode_binary_mask(curr_mask))

            # Final iteration if we have leftover unsmoothed segmentations from the last batch
            if pad_next < r:
                # Compute segmentation
                segmentation = self.smooth_seg(prev_segmentation, pad_prev=pad_prev, pad_next=r)
                mask = segmentation.argmax(1) == 1

                # Encode segmentation
                for b in range(mask.shape[0]):
                    curr_mask = mask[b].cpu().numpy()
                    if self.seg_remove_mouth:
                        curr_mask = remove_inner_mouth(curr_mask, landmarks[frame_count])
                        frame_count += 1
                    encoded_segmentations.append(encode_binary_mask(curr_mask))

            # Write to file
            with open(curr_seg_path, "wb") as fp:  # Pickling
                pickle.dump(encoded_segmentations, fp)


    def cache(self, input_path, output_dir=None):
        # Validation
        assert os.path.isfile(input_path), 'Input path "%s" does not exist' % input_path
        assert output_dir is None or os.path.isdir(output_dir), 'Output path "%s" must be a directory' % output_dir
        is_vid = os.path.splitext(input_path)[1] == '.mp4'

        # Set paths
        output_dir = os.path.splitext(input_path)[0] if output_dir is None else output_dir
        det_file_path = os.path.splitext(input_path)[0] + self.det_postfix
        if not os.path.isfile(det_file_path):   # Check if there is a detection file in the same directory as the video
            det_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] +
                                         self.det_postfix)
        seq_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + self.seq_postfix)
        first_cropped_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] +
                                          '_seq00' + os.path.splitext(input_path)[1])
        pose_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + self.pose_postfix)

        # Create directory
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Face detection
        if not os.path.isfile(det_file_path):
            self.face_detector(input_path, det_file_path)

        # Detections to sequences
        if not os.path.isfile(seq_file_path):
            detections2sequences_main(input_path, seq_file_path, det_file_path, self.iou_thresh, self.min_length,
                                      self.min_size, self.crop_scale, self.center_kernel, self.size_kernel,
                                      self.smooth_det, self.display, self.write_empty)

        # Crop video sequences
        if not os.path.isfile(first_cropped_path):
            if is_vid:
                crop_video_sequences_main(input_path, output_dir, seq_file_path, self.seq_postfix, self.resolution,
                                          self.crop_scale, select='all', disable_tqdm=False,
                                          encoder_codec=self.encoder_codec)
            else:
                crop_image_sequences_main(input_path, output_dir, seq_file_path, self.seq_postfix, '.jpg',
                                          self.resolution, self.crop_scale)

        # Face poses
        # if not os.path.isfile(pose_file_path) and is_vid:
        #     self.process_pose(input_path, output_dir, seq_file_path, pose_file_path)
        # if is_vid:
        self.process_pose(input_path, output_dir, seq_file_path)

        # Extract frontal images
        # if self.cache_pose and self.cache_frontal and is_vid:
        #     self.extract_frontal_images(input_path, output_dir, pose_file_path)
        if self.cache_pose and self.cache_frontal and is_vid:
            self.extract_frontal_images(input_path, output_dir, seq_file_path)

        # Cache landmarks
        self.process_landmarks(input_path, output_dir, seq_file_path)

        # Cache segmentation
        self.process_segmentation(input_path, output_dir, seq_file_path)

        return output_dir, seq_file_path, pose_file_path if self.cache_pose and is_vid else None


class VideoProcessCallable(VideoProcessBase):
    def __init__(self, *args, **kwargs):
        super(VideoProcessCallable, self).__init__(*args, **kwargs)

    def __call__(self, input_path, output_dir=None):
        return self.cache(input_path, output_dir)


def smooth_poses(poses, kernel_size=5):
    out_poses = poses.copy() if isinstance(poses, np.ndarray) else np.array(poses)

    # Prepare smoothing kernel
    # w = np.hamming(kernel_size)
    w = np.ones(kernel_size)
    w /= w.sum()

    # Smooth poses
    poses_padded = np.pad(out_poses, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(out_poses.shape[1]):
        out_poses[:, i] = np.convolve(w, poses_padded[:, i], mode='valid')

    return out_poses


def main(input, output=d('output'), resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
         cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
         # Detection arguments:
         detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
         # Sequence arguments:
         iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
         center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
         seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
         # Pose arguments:
         pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
         cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
         # Landmarks arguments:
         lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
         cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
         # Segmentation arguments:
         seg_model=d('seg_model'), seg_batch_size=d('seg_batch_size'), segmentation_postfix=d('segmentation_postfix'),
         cache_segmentation=d('cache_segmentation'), smooth_segmentation=d('smooth_segmentation'),
         seg_remove_mouth=d('seg_remove_mouth')):
    video_process = VideoProcessCallable(
        resolution, crop_scale, gpus, cpu_only, display, verbose, encoder_codec,
        detection_model=detection_model, det_batch_size=det_batch_size, det_postfix=det_postfix,
        iou_thresh=iou_thresh, min_length=min_length, min_size=min_size, center_kernel=center_kernel,
        size_kernel=size_kernel, smooth_det=smooth_det, seq_postfix=seq_postfix, write_empty=write_empty,
        pose_model=pose_model, pose_batch_size=pose_batch_size, pose_postfix=pose_postfix, cache_pose=cache_pose,
        cache_frontal=cache_frontal, smooth_poses=smooth_poses, lms_model=lms_model, lms_batch_size=lms_batch_size,
        landmarks_postfix=landmarks_postfix, cache_landmarks=cache_landmarks, smooth_landmarks=smooth_landmarks,
        seg_model=seg_model, seg_batch_size=seg_batch_size, segmentation_postfix=segmentation_postfix,
        cache_segmentation=cache_segmentation, smooth_segmentation=smooth_segmentation,
        seg_remove_mouth=seg_remove_mouth)
    if len(input) == 1 and os.path.isfile(input[0]):
        video_process.cache(input, output)
    else:
        batch(input, None, output, video_process, postfix='.mp4')


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
