import os
import random
import pickle
import torch.utils.data as data
import numpy as np
import cv2
import torch
from fsgan.utils.video_utils import Sequence, get_video_info


class VideoInferenceDataset(data.Dataset):
    """A dataset for loading video sequences.

    Args:
        root (string): Root directory path or file list path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
     Attributes:
        video_paths (list): List of video paths
    """

    def __init__(self, vid_path, seq=None, transform=None):
        self.vid_path = vid_path
        self.seq = seq
        self.transform = transform
        self.cap = None

        # Get video info
        self.width, self.height, self.total_frames, self.fps = get_video_info(vid_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target) where target is True for same identity else False.
        """
        if self.cap is None:
            # Open video file
            self.cap = cv2.VideoCapture(self.vid_path)
            if self.seq is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seq.start_index)

        ret, frame_bgr = self.cap.read()
        assert frame_bgr is not None, 'Failed to read frame from video in index: %d' % index
        frame_rgb = frame_bgr[:, :, ::-1]
        bbox = self.seq.detections[index] if self.seq is not None else None

        # Apply transformation
        if self.transform is not None:
            if bbox is None:
                frame_rgb = self.transform(frame_rgb)
            else:
                frame_rgb = self.transform(frame_rgb, bbox)

        return frame_rgb

    def __len__(self):
        return self.total_frames if self.seq is None else len(self.seq)
