import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from scipy.spatial import cKDTree, Delaunay
import torch
import torch.utils.data as data
from fsgan.utils.seg_utils import decode_binary_mask
from fsgan.utils.video_utils import get_video_info


def fuse_clusters(points, r=0.5):
    """ Select a single point from each cluster of points.

    The clustering is done using a KD-Tree data structure for querying points by radius.

    Args:
        points (np.array): A set of points of shape (N, 2) to fuse
        r (float): The radius for which to fuse the points

    Returns:
        np.array: The indices of remaining points.
    """
    kdt = cKDTree(points)
    indices = kdt.query_ball_point(points, r=r)

    # Build sorted neightbor list
    neighbors = [(i, l) for i, l in enumerate(indices)]
    neighbors.sort(key=lambda t: len(t[1]), reverse=True)

    # Mark remaining indices
    keep = np.ones(points.shape[0], dtype=bool)
    for i, cluster in neighbors:
        if not keep[i]:
            continue
        for j in cluster:
            if i == j:
                continue
            keep[j] = False

    return np.nonzero(keep)[0]


class AppearanceMapDataset(data.Dataset):
    """A dataset representing the appearance map of a video sequence

    Args:
        root (string): Root directory path or file list path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
     Attributes:
        video_paths (list): List of video paths
    """
    def __init__(self, src_vid_seq_path, tgt_vid_seq_path, src_transform=None, tgt_transform=None,
                 landmarks_postfix='_lms.npz', pose_postfix='_pose.npz', seg_postfix='_seg.pkl', min_radius=0.5):
        assert os.path.isfile(src_vid_seq_path), f'src_vid_seq_path is not a path to a file: {src_vid_seq_path}'
        assert os.path.isfile(tgt_vid_seq_path), f'tgt_vid_seq_path is not a path to a file: {tgt_vid_seq_path}'
        self.src_transform = src_transform
        self.tgt_transform = tgt_transform
        self.src_vid_seq_path = src_vid_seq_path
        self.tgt_vid_seq_path = tgt_vid_seq_path
        self.src_vid = None
        self.tgt_vid = None

        # Get target video info
        self.width, self.height, self.total_frames, self.fps = get_video_info(tgt_vid_seq_path)

        # Load landmarks
        src_lms_path = os.path.splitext(src_vid_seq_path)[0] + landmarks_postfix
        self.src_landmarks = np.load(src_lms_path)['landmarks_smoothed']
        tgt_lms_path = os.path.splitext(tgt_vid_seq_path)[0] + landmarks_postfix
        self.tgt_landmarks = np.load(tgt_lms_path)['landmarks_smoothed']

        # Load poses
        src_pose_path = os.path.splitext(src_vid_seq_path)[0] + pose_postfix
        self.src_poses = np.load(src_pose_path)['poses_smoothed']
        tgt_pose_path = os.path.splitext(tgt_vid_seq_path)[0] + pose_postfix
        self.tgt_poses = np.load(tgt_pose_path)['poses_smoothed']

        # Load target segmentations
        tgt_seg_path = os.path.splitext(tgt_vid_seq_path)[0] + seg_postfix
        with open(tgt_seg_path, "rb") as fp:  # Unpickling
            self.tgt_encoded_seg = pickle.load(fp)

        # Initialize appearance map
        self.filtered_indices = fuse_clusters(self.src_poses[:, :2], r=min_radius / 99.)
        self.points = self.src_poses[self.filtered_indices, :2]
        limit_points = np.array([[-75., -75.], [-75., 75.], [75., -75.], [75., 75.]]) / 99.
        self.points = np.concatenate((self.points, limit_points))
        self.tri = Delaunay(self.points)
        self.valid_size = len(self.filtered_indices)

        # Filter source landmarks and poses and handle edge cases
        self.src_landmarks = self.src_landmarks[self.filtered_indices]
        self.src_landmarks = np.vstack((self.src_landmarks, np.zeros_like(self.src_landmarks[-1:])))
        self.src_poses = self.src_poses[self.filtered_indices]
        self.src_poses = np.vstack((self.src_poses, np.zeros_like(self.src_poses[-1:])))

        # Initialize cached frames
        self.src_frames = [None for i in range(len(self.filtered_indices) + 1)]

        # Handle edge cases
        black_rgb = np.zeros((self.height, self.width, 3), dtype='uint8')
        self.src_frames[-1] = black_rgb

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target) where target is True for same identity else False.
        """
        if self.src_vid is None:
            # Open source video on the data loader's process
            self.src_vid = cv2.VideoCapture(self.src_vid_seq_path)
        if self.tgt_vid is None:
            # Open target video on the data loader's process
            self.tgt_vid = cv2.VideoCapture(self.tgt_vid_seq_path)

        # Read next target frame and meta-data
        ret, tgt_frame_bgr = self.tgt_vid.read()
        assert tgt_frame_bgr is not None, 'Failed to read frame from video in index: %d' % index
        tgt_frame = tgt_frame_bgr[:, :, ::-1]
        tgt_landmarks = self.tgt_landmarks[index]
        tgt_pose = self.tgt_poses[index]
        tgt_seg = decode_binary_mask(self.tgt_encoded_seg[index])

        # Query source frames and meta-data given the current target pose
        query_point, tilt_angle = tgt_pose[:2], tgt_pose[2]
        tri_index = self.tri.find_simplex(query_point[:2])
        tri_vertices = self.tri.simplices[tri_index]
        tri_vertices = np.minimum(tri_vertices, self.valid_size)

        # Compute barycentric weights
        b = self.tri.transform[tri_index, :2].dot(query_point[:2] - self.tri.transform[tri_index, 2])
        bw = np.array([b[0], b[1], 1 - b.sum()], dtype='float32')
        bw[tri_vertices >= self.valid_size] = 0.    # Set zero weight for edge points
        bw /= bw.sum()

        # Cache source frames
        for tv in np.sort(tri_vertices):
            if self.src_frames[tv] is None:
                self.src_vid.set(cv2.CAP_PROP_POS_FRAMES, self.filtered_indices[tv])
                ret, frame_bgr = self.src_vid.read()
                assert frame_bgr is not None, 'Failed to read frame from source video in index: %d' % tv
                frame_rgb = frame_bgr[:, :, ::-1]
                self.src_frames[tv] = frame_rgb

        # Get source data from appearance map
        src_frames = [self.src_frames[tv] for tv in tri_vertices]
        src_landmarks = self.src_landmarks[tri_vertices].astype('float32')
        src_poses = self.src_poses[tri_vertices].astype('float32')

        # Apply source transformation
        if self.src_transform is not None:
            src_data = [(src_frames[i], src_landmarks[i], (src_poses[i][2] - tilt_angle) * 99.)
                        for i in range(len(src_frames))]
            src_data = self.src_transform(src_data)
            src_landmarks = torch.stack([src_data[i][1] for i in range(len(src_data))])
            src_frames = [src_data[i][0] for i in range(len(src_data))]
            src_poses[:, 2] = tilt_angle

        # Apply target transformation
        if self.tgt_transform is not None:
            tgt_frame = self.tgt_transform(tgt_frame)

        # Combine pyramids in source frames if they exist
        if isinstance(src_frames[0], (list, tuple)):
            src_frames = [torch.stack([src_frames[f][p] for f in range(len(src_frames))], dim=0)
                          for p in range(len(src_frames[0]))]

        return src_frames, src_landmarks, src_poses, bw, tgt_frame, tgt_landmarks, tgt_pose, tgt_seg

    def __len__(self):
        return self.tgt_poses.shape[0]
