""" Video utilities. """

import numpy as np
from itertools import count
import ffmpeg
from fsgan.utils.one_euro_filter import OneEuroFilter


class Sequence(object):
    """ Represents a sequence of detected faces in a video.

    Args:
        start_index (int): The frame index in the video from which the sequence starts
        det (np.array): Frame face detections bounding boxes of shape (N, 4), in the format [left, top, right, bottom]
    """
    _ids = count(0)

    def __init__(self, start_index, det=None):
        self.start_index = start_index
        self.size_sum = 0.
        self.size_avg = 0.
        self.id = next(self._ids)
        self.obj_id = -1
        self.detections = []

        if det is not None:
            self.add(det)

    def add(self, det):
        """ Add new frame detection bounding boxes.

        Args:
            det (np.array): Frame face detections bounding boxes of shape (N, 4), in the format
                [left, top, right, bottom]
        """
        self.detections.append(det)
        size = det[3] - det[1]
        self.size_sum += size
        self.size_avg = self.size_sum / len(self.detections)

    def smooth(self, kernel_size=7):
        """ Temporally smooth the detection bounding boxes.

        Args:
            kernel_size (int): The temporal kernel size
        """
        # Prepare smoothing kernel
        w = np.hamming(kernel_size)
        w /= w.sum()

        # Smooth bounding boxes
        bboxes = np.array(self.detections)
        bboxes_padded = np.pad(bboxes, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
        for i in range(bboxes.shape[1]):
            bboxes[:, i] = np.convolve(w, bboxes_padded[:, i], mode='valid')

        self.detections = bboxes

    def finalize(self):
        """ Packs all list of added detections into a single numpy array.

        Should be called after all detections were added if smooth was not called.
        """
        self.detections = np.array(self.detections)

    def __getitem__(self, index):
        return self.detections[index]

    def __len__(self):
        return len(self.detections)


# TODO: Remove this
def estimate_motion(detections, min_cutoff=0.0, beta=3.0, d_cutoff=5.0, fps=30.0):
    one_euro_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, t_e=(1.0 / fps))
    detections_n = np.array(detections)
    center = np.mean((detections_n[:, 2:] + detections_n[:, :2])*0.5, axis=0)
    size = np.mean(detections_n[:, 2:] - detections_n[:, :2], axis=0)
    detections_n = (detections_n - np.concatenate((center, center))) / np.concatenate((size, size))

    motion = []
    for det in detections_n:
        det_s, a = one_euro_filter(det)
        motion.append(a)

    return np.array(motion)


# TODO: Remove this
def smooth_detections_avg(detections, kernel_size=7):
    # Prepare smoothing kernel
    # w = np.hamming(kernel_size)
    w = np.ones(kernel_size)
    w /= w.sum()

    # Smooth bounding boxes
    bboxes = np.array(detections)
    bboxes_padded = np.pad(bboxes, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(bboxes.shape[1]):
        bboxes[:, i] = np.convolve(w, bboxes_padded[:, i], mode='valid')

    return bboxes


# TODO: Remove this
def smooth_detections_1euro(detections, kernel_size=7, min_cutoff=0.0, beta=3.0, d_cutoff=5.0, fps=30.0):
    detections_np = np.array(detections)
    detections_avg = smooth_detections_avg(detections, kernel_size)
    motion = np.expand_dims(estimate_motion(detections, min_cutoff, beta, d_cutoff, fps), 1).astype('float32')
    out_detections = detections_np * motion + detections_avg * (1 - motion)

    return out_detections


# TODO: Remove this
def smooth_detections_avg_center(detections, center_kernel=11, size_kernel=21):
    # Prepare smoothing kernel
    center_w = np.ones(center_kernel)
    center_w /= center_w.sum()
    size_w = np.ones(size_kernel)
    size_w /= size_w.sum()

    # Convert bounding boxes to center and size format
    bboxes = np.array(detections)
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    sizes = bboxes[:, 2:] - bboxes[:, :2]

    # Smooth bounding boxes
    centers_padded = np.pad(centers, ((center_kernel // 2, center_kernel // 2), (0, 0)), 'reflect')
    sizes_padded = np.pad(sizes, ((size_kernel // 2, size_kernel // 2), (0, 0)), 'reflect')
    for i in range(centers.shape[1]):
        centers[:, i] = np.convolve(center_w, centers_padded[:, i], mode='valid')
        sizes[:, i] = np.convolve(size_w, sizes_padded[:, i], mode='valid')

    # Change back to detections format
    sizes /= 2.0
    bboxes = np.concatenate((centers - sizes, centers + sizes), axis=1)

    return bboxes


def get_main_sequence(seq_list, frame_size):
    """ Return the main sequence in a list of sequences according to their size and how central they are.

    Args:
        seq_list (list of Sequence): List of sequences
        frame_size (tuple of int): The corresponding sequence video's frame size of shape (H, W)

    Returns:
        Sequence: The main sequence.
    """
    if len(seq_list) == 0:
        return None

    # Calculate frame max distance and size
    img_center = np.array([frame_size[1], frame_size[0]]) * 0.5
    max_dist = 0.25 * np.linalg.norm(frame_size)
    max_size = 0.25 * (frame_size[0] + frame_size[1])

    # For each sequence
    seq_scores = []
    for seq in seq_list:

        # For each detection in the sequence
        det_scores = []
        for det in seq:
            bbox = np.concatenate((det[:2], det[2:] - det[:2]))

            # Calculate center distance
            bbox_center = bbox[:2] + bbox[2:] * 0.5
            bbox_dist = np.linalg.norm(bbox_center - img_center)

            # Calculate bbox size
            bbox_size = bbox[2:].mean()

            # Calculate central ratio
            central_ratio = 1.0 if max_size < 1e-6 else (1.0 - bbox_dist / max_dist)
            central_ratio = np.clip(central_ratio, 0.0, 1.0)

            # Calculate size ratio
            size_ratio = 1.0 if max_size < 1e-6 else (bbox_size / max_size)
            size_ratio = np.clip(size_ratio, 0.0, 1.0)

            # Add score
            score = (central_ratio + size_ratio) * 0.5
            det_scores.append(score)

        seq_scores.append(np.array(det_scores).mean())

    return seq_list[np.argmax(seq_scores)]


def get_media_info(media_path):
    """ Return media information.

    Args:
        media_path (str): Path to media file

    Returns:
        (int, int, int, float): Tuple containing:
            - width (int): Frame width
            - height (int): Frame height
            - total_frames (int): Total number of frames (will be 1 for images)
            - fps (float): Frames per second (irrelevant for images)
    """
    probe = ffmpeg.probe(media_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    total_frames = int(video_stream['nb_frames']) if 'nb_frames' in video_stream else 1
    fps_part1, fps_part2 = video_stream['r_frame_rate'].split(sep='/')
    fps = float(fps_part1) / float(fps_part2)

    return width, height, total_frames, fps


def get_media_resolution(media_path):
    return get_media_info(media_path)[:2]


# TODO: Remove this
def get_video_info(vid_path):
    return get_media_info(vid_path)
