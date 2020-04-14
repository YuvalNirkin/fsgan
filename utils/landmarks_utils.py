""" Face landmarks utilities. """

from collections.abc import Iterable
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def hflip_face_landmarks_68pts(landmarks, width):
    """ Horizontal flip 68 points landmarks.

    Args:
        landmarks (np.array): Landmarks points of shape (68, 2)
        width (int): The width of the correspondign image

    Returns:
        np.array: Horizontally flipped landmarks.
    """
    assert landmarks.shape[0] == 68
    landmarks = landmarks.copy()

    # Invert X coordinates
    for p in landmarks:
        p[0] = width - p[0]

    # Jaw
    right_jaw, left_jaw = list(range(0, 8)), list(range(16, 8, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    # Eyebrows
    right_brow, left_brow = list(range(17, 22)), list(range(26, 21, -1))
    landmarks[right_brow + left_brow] = landmarks[left_brow + right_brow]

    # Nose
    right_nostril, left_nostril = list(range(31, 33)), list(range(35, 33, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    # Eyes
    right_eye, left_eye = list(range(36, 42)), [45, 44, 43, 42, 47, 46]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    # Mouth outer
    mouth_out_right, mouth_out_left = [48, 49, 50, 59, 58], [54, 53, 52, 55, 56]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    # Mouth inner
    mouth_in_right, mouth_in_left = [60, 61, 67], [64, 63, 65]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks


def hflip_face_landmarks_98pts(landmarks, width=1):
    """ Horizontal flip 98 points landmarks.

    Args:
        landmarks (np.array): Landmarks points of shape (98, 2)
        width (int): The width of the correspondign image

    Returns:
        np.array: Horizontally flipped landmarks.
    """
    assert landmarks.shape[0] == 98
    landmarks = landmarks.copy()

    # Invert X coordinates
    for p in landmarks:
        p[0] = width - p[0]

    # Jaw
    right_jaw, left_jaw = list(range(0, 16)), list(range(32, 16, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    # Eyebrows
    right_brow, left_brow = list(range(33, 42)), list(range(46, 41, -1)) + list(range(50, 46, -1))
    landmarks[right_brow + left_brow] = landmarks[left_brow + right_brow]

    # Nose
    right_nostril, left_nostril = list(range(55, 57)), list(range(59, 57, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    # Eyes
    right_eye, left_eye = list(range(60, 68)) + [96], [72, 71, 70, 69, 68, 75, 74, 73, 97]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    # Mouth outer
    mouth_out_right, mouth_out_left = [76, 77, 78, 87, 86], [82, 81, 80, 83, 84]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    # Mouth inner
    mouth_in_right, mouth_in_left = [88, 89, 95], [92, 91, 93]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks


def filter_landmarks(landmarks, threshold=0.5):
    """ Filter landmarks feature map activations by threshold.

    Args:
        landmarks (torch.Tensor): Landmarks feature map of shape (B, C, H, W)
        threshold (float): Filtering threshold

    Returns:
        torch.Tensor: Filtered landmarks feature map of shape (B, C, H, W)
    """
    landmarks_min = landmarks.view(landmarks.shape[:2] + (-1,)).min(2)[0].view(landmarks.shape[:2] + (1, 1))
    landmarks_max = landmarks.view(landmarks.shape[:2] + (-1,)).max(2)[0].view(landmarks.shape[:2] + (1, 1))
    landmarks = (landmarks - landmarks_min) / (landmarks_max - landmarks_min)
    # landmarks.pow_(2)
    landmarks[landmarks < threshold] = 0.0

    return landmarks


class LandmarksHeatMapEncoder(nn.Module):
    """ Encodes landmarks heatmap into a landmarks vector of points.

    Args:
        size (int or sequence of int): the size of the landmarks heat map (height, width)
    """
    def __init__(self, size=64):
        super(LandmarksHeatMapEncoder, self).__init__()
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        size = (size, size) if isinstance(size, int) else size
        y_indices, x_indices = torch.meshgrid(torch.arange(0., size[1]), torch.arange(0., size[0]))
        self.register_buffer('x_indices', x_indices.add(0.5) / size[1])
        self.register_buffer('y_indices', y_indices.add(0.5) / size[0])

    def __call__(self, landmarks):
        """ Encode landmarks heatmap to landmarks points.

        Args:
            landmarks (torch.Tensor): Landmarks heatmap of shape (B, C, H, W)

        Returns:
            torch.Tensor: Encoded landmarks points of shape (B, C, 2).
        """
        landmarks = filter_landmarks(landmarks)
        w = landmarks.div(landmarks.view(landmarks.shape[:2] + (-1,)).sum(dim=2).view(landmarks.shape[:2] + (1, 1)))
        x = w * self.x_indices
        y = w * self.y_indices
        x = x.view(x.shape[:2] + (-1,)).sum(dim=2).unsqueeze(2)
        y = y.view(y.shape[:2] + (-1,)).sum(dim=2).unsqueeze(2)
        landmarks = torch.cat((x, y), dim=2)

        return landmarks


class LandmarksHeatMapDecoder(nn.Module):
    """ Decodes a landmarks vector of points into a landmarks heatmap.

    Args:
        size (int or sequence of int): the size of the landmarks heat map (height, width)
        threshold (float): A threshold that controls the size of the heatmap's disks
    """
    def __init__(self, size=256, threshold=0.8):
        super(LandmarksHeatMapDecoder, self).__init__()
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = (size, size) if isinstance(size, int) else size
        self.threshold = threshold
        self.max_dist = np.sqrt(2)

        y_indices, x_indices = torch.meshgrid(torch.arange(0., self.size[1]), torch.arange(0., self.size[0]))
        self.register_buffer('x_indices', x_indices.add(0.5) / self.size[1])
        self.register_buffer('y_indices', y_indices.add(0.5) / self.size[0])

    def __call__(self, landmarks):
        """ Decode landmarks points to landmarks heatmap.

        Args:
            landmarks (torch.Tensor): Landmarks points of shape (B, C, 2)

        Returns:
            torch.Tensor: Decoded landmarks heatmap of shape (B, C, H, W).
        """
        x = landmarks[:, :, 0].view(landmarks.shape[:2] + (1, 1))
        y = landmarks[:, :, 1].view(landmarks.shape[:2] + (1, 1))
        x_indices = self.x_indices.view((1, 1) + self.x_indices.shape).repeat(landmarks.shape[:2] + (1, 1))
        y_indices = self.y_indices.view((1, 1) + self.y_indices.shape).repeat(landmarks.shape[:2] + (1, 1))
        landmarks = self.max_dist - torch.sqrt(torch.pow(x_indices - x, 2) + torch.pow(y_indices - y, 2))
        landmarks.div_(self.max_dist)

        # Decrease disc size
        landmarks.pow_(8)
        landmarks.sub_(self.threshold).div_(1 - self.threshold)
        landmarks[landmarks < 0] = 0.0

        return landmarks


# TODO: Remove this function
def encode_landmarks_98pts(context, resolution=256):
    for i in range(np.log2(context.shape[2]).astype(int) + 1, np.log2(resolution).astype(int) + 1):
        curr_res = np.power(2, i)
        context = F.interpolate(context, size=(curr_res, curr_res), mode='bilinear', align_corners=False)
    out = torch.zeros(context.shape[0], 3, *context.shape[2:], dtype=context.dtype, device=context.device)

    # Jaw
    activation = context[:, :33].sum(axis=1).clamp(max=1.)
    out[:, 0] += activation  # White [255, 255, 255]
    out[:, 1] += activation  # White [255, 255, 255]
    out[:, 2] += activation  # White [255, 255, 255]

    # Eyebrows
    activation = context[:, 33:51].sum(axis=1).clamp(max=1.)
    out[:, 0] += activation  # Purple [255, 0, 255]
    out[:, 2] += activation  # Purple [255, 0, 255]

    # Nose
    activation = context[:, 51:60].sum(axis=1).clamp(max=1.)
    out[:, 1] += activation  # Cyan [0, 255, 255]
    out[:, 2] += activation  # Cyan [0, 255, 255]

    # Eyes
    activation = context[:, 60:76].sum(axis=1).clamp(max=1.)
    out[:, 0] += activation  # Yellow [255, 255, 0]
    out[:, 1] += activation  # Yellow [255, 255, 0]

    # Iris
    out[:, 2] += context[:, 96:].sum(axis=1).clamp(max=1.)  # Blue

    # Mouth outer
    out[:, 0] += context[:, 76:88].sum(axis=1).clamp(max=1.)  # Red

    # Mouth inner
    out[:, 1] += context[:, 88:96].sum(axis=1).clamp(max=1.)  # Green

    out.clamp_(max=1.).mul_(2.0).sub_(1.0)

    return out


def smooth_landmarks(landmarks, kernel_size=5):
    """ Temporally smooth a series of face landmarks by averaging.

    Args:
        landmarks (np.array): A sequence of face landmarks of shape (N, C, 2) where N is the length of the sequence
            and C is the number of landmarks points
        kernel_size (int): The temporal kernel size

    Returns:
        np.array: The smoothed face landmarks of shape (N, C, 2).
    """
    out_landmarks = landmarks.copy()

    # Prepare smoothing kernel
    w = np.ones(kernel_size)
    w /= w.sum()

    # Smooth landmarks by applying convolution kernel
    orig_shape = landmarks.shape
    out_landmarks = out_landmarks.reshape(out_landmarks.shape[0], -1)
    landmarks_padded = np.pad(out_landmarks, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(out_landmarks.shape[1]):
        out_landmarks[:, i] = np.convolve(w, landmarks_padded[:, i], mode='valid')
    out_landmarks = out_landmarks.reshape(orig_shape)

    return out_landmarks


def estimate_motion(landmarks, kernel_size=5):
    """ Estimate motion of a temporal sequence of face landmarks.

    Args:
        landmarks (np.array): A sequence of face landmarks of shape (N, C, 2) where N is the length of the sequence
            and C is the number of landmarks points
        kernel_size (int): The temporal kernel size

    Returns:
        motion (np.array): Array of scalars of shape (N,) representing the amount of motion.
    """
    deltas = np.zeros(landmarks.shape)
    deltas[1:] = landmarks[1:] - landmarks[:-1]

    # Prepare smoothing kernel
    w = np.ones(kernel_size)
    w /= w.sum()

    # Smooth landmarks by applying convolution kernel
    orig_shape = deltas.shape
    deltas = deltas.reshape(deltas.shape[0], -1)
    deltas_padded = np.pad(deltas, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(deltas.shape[1]):
        deltas[:, i] = np.convolve(w, deltas_padded[:, i], mode='valid')
    deltas = deltas.reshape(orig_shape)

    motion = np.linalg.norm(deltas, axis=2)

    return motion


def smooth_landmarks_98pts(landmarks, smooth_kernel_size=7, motion_kernel_size=5, max_motion=0.01):
    """ Temporally smooth a series of face landmarks by motion estimate.

    The motion is estimate for each group of specific face part separately.
    Based on the idea of the one Euro filter described in the paper:
    `"1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems"
    <https://dl.acm.org/doi/pdf/10.1145/2207676.2208639>`_

    Args:
        landmarks (np.array): A sequence of face landmarks of shape (N, C, 2) where N is the length of the sequence
            and C is the number of landmarks points
        smooth_kernel_size (int): Average smoothing kernel size
        motion_kernel_size (int): Motion estimate kernel size
        max_motion (float): The maximum allowed motion (for normalization)

    Returns:
        np.array: The smoothed face landmarks of shape (N, C, 2).
    """
    landmarks_out = landmarks.copy()
    landmarks_avg = smooth_landmarks(landmarks, kernel_size=smooth_kernel_size)
    motion = estimate_motion(landmarks, motion_kernel_size)

    landmarks_parts_indices = [
        list(range(0, 33)),                                         # jaw
        list(range(51, 60)),                                        # nose
        list(range(77, 82)) + list(range(89, 92)),                  # mouth_upper
        [76] + list(range(82, 88)) + [88] + list(range(92, 96)),    # mouth_lower
        list(range(33, 42)),                                        # eyebrow_right
        list(range(42, 51)),                                        # eyebrow_left
        list(range(60, 68)),                                        # eye_right
        list(range(68, 76)),                                        # eye_left
        [96],                                                       # iris_right
        [97]                                                        # iris_left
    ]

    for part_indices in landmarks_parts_indices:
        part_motion = motion[:, part_indices].mean(axis=1)
        a = np.minimum(part_motion / max_motion, 1.)[..., np.newaxis, np.newaxis]
        landmarks_out[:, part_indices] = landmarks[:, part_indices] * a + landmarks_avg[:, part_indices] * (1 - a)

    return landmarks_out


def blend_landmarks_heatmap(img, heatmap, alpha=0.25, color='red'):
    """ Blend images with landmarks heatmaps.

    Args:
        img (torch.Tensor): A batch of image tensors of shape (B, 3, H, W)
        heatmap (torch.Tensor): A batch of landmarks heatmaps of shape (B, C, H, W)
        alpha (float): Opacity value for the landmarks heatmap in the range [0, 1] where 0 is completely transparent
            and 1 is completely opaque
        color (str): The landmarks heatmap color. Can be 'red', 'green', or 'blue'

    Returns:
        torch.Tensor: The blended image tensors.
    """
    color_mask = -torch.ones_like(img)
    if color == 'red':
        color_mask[:, 0, :, :] = 1
    elif color == 'green':
        color_mask[:, 1, :, :] = 1
    elif color == 'blue':
        color_mask[:, 2, :, :] = 1

    alpha_map = 1 - torch.clamp(heatmap.sum(dim=1), max=1.0) * alpha
    alpha_map = alpha_map.unsqueeze(1).repeat(1, 3, 1, 1)

    return img * alpha_map + color_mask * (1 - alpha_map)
