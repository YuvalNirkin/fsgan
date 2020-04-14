""" Bounding box utilities. """

import numpy as np
import cv2


# Adapted from: http://ronny.rest/tutorials/module/localization_001/iou/
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


# Adapted from: http://ronny.rest/tutorials/module/localization_001/iou/
def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def scale_bbox(bbox, scale=1.2, square=True):
    """ Scale bounding box by the specified scale and optionally make it square.

    Args:
        bbox (np.array): Input bounding box in the format [left, top, width, height]
        scale (float): Multiply the bounding box by this scale
        square (bool): If True, make the shorter edges of the bounding box equal the length as the longer edges

    Returns:
        np.array. The scaled bounding box
    """
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def crop_img(img, bbox, landmarks=None, border=cv2.BORDER_CONSTANT, value=None):
    """ Crop image and corresponding landmarks by bounding box.

    If the bounding box is out the image bounds, the image will be padded in the corresponding regions.

    Args:
        img (np.array): An image of shape (H, W, 3)
        landmarks (np.array): Face landmarks points of shape (68, 2)
        bbox (np.array): Bounding box in the format [left, top, width, height]
        border (int): OpenCV's border code
        value (int, optional): Border value if border==cv2.BORDER_CONSTANT

    Returns:
        (np.array, np.array (optional)): A tuple of numpy arrays containing:
            - Cropped image (np.array)
            - Cropped landmarks (np.array): Will be returned if the landmarks parameter is not None
    """
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    img_bbox = bbox.copy()
    if any((left, top, right, bottom)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, border, value=value)
        img_bbox[0] += left
        img_bbox[1] += top

    if landmarks is not None:
        # Adjust landmarks
        new_landmarks = landmarks.copy()
        new_landmarks[:, :2] += (np.array([left, top]) - img_bbox[:2])
        return img[img_bbox[1]:img_bbox[1] + img_bbox[3], img_bbox[0]:img_bbox[0] + img_bbox[2]], new_landmarks
    else:
        return img[img_bbox[1]:img_bbox[1] + img_bbox[3], img_bbox[0]:img_bbox[0] + img_bbox[2]]


def crop2img(img, crop, bbox):
    """ Writes cropped image into another image corresponding to the specified bounding box.

    Args:
        img (np.array): The image to write into of shape (H, W, 3)
        crop (np.array): The cropped image of shape (H, W, 3)
        bbox (np.array): Bounding box in the format [left, top, width, height]

    Returns:
        np.array: Result image.
    """
    scaled_bbox = bbox
    scaled_crop = cv2.resize(crop, (scaled_bbox[3], scaled_bbox[2]), interpolation=cv2.INTER_CUBIC)
    left = -scaled_bbox[0] if scaled_bbox[0] < 0 else 0
    top = -scaled_bbox[1] if scaled_bbox[1] < 0 else 0
    right = scaled_bbox[0] + scaled_bbox[2] - img.shape[1] if (scaled_bbox[0] + scaled_bbox[2] - img.shape[1]) > 0 else 0
    bottom = scaled_bbox[1] + scaled_bbox[3] - img.shape[0] if (scaled_bbox[1] + scaled_bbox[3] - img.shape[0]) > 0 else 0
    crop_bbox = np.array([left, top, scaled_bbox[2] - left - right, scaled_bbox[3] - top - bottom])
    scaled_bbox += np.array([left, top, -left - right, -top - bottom])

    out_img = img.copy()
    out_img[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]] = \
        scaled_crop[crop_bbox[1]:crop_bbox[1] + crop_bbox[3], crop_bbox[0]:crop_bbox[0] + crop_bbox[2]]

    return out_img


def get_main_bbox(bboxes, img_size):
    """ Returns the main bounding box in a list of bounding boxes according to their size and how central they are.

    Args:
        bboxes (list of np.array): A list of bounding boxes in the format [left, top, width, height]
        img_size (tuple of int): The size of the corresponding image in the format [height, width]

    Returns:
        np.array: The main bounding box.
    """
    if len(bboxes) == 0:
        return None

    # Calculate frame max distance and size
    img_center = np.array([img_size[1], img_size[0]]) * 0.5
    max_dist = 0.25 * np.linalg.norm(img_size)
    max_size = 0.25 * (img_size[0] + img_size[1])

    # For each bounding box
    scores = []
    for bbox in bboxes:
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
        scores.append(score)

    return bboxes[np.argmax(scores)]


def estimate_motion(points, kernel_size=5):
    """ Estimate motion of temporally sampled points.

    Args:
        points (np.array): An array of temporally sampled points of shape (N, 2)
        kernel_size (int): The temporal kernel size

    Returns:
        motion (np.array): Array of scalars of shape (N,) representing the amount of motion.
    """
    deltas = np.zeros(points.shape)
    deltas[1:] = points[1:] - points[:-1]

    # Prepare smoothing kernel
    w = np.ones(kernel_size)
    w /= w.sum()

    # Smooth points
    deltas_padded = np.pad(deltas, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(points.shape[1]):
        deltas[:, i] = np.convolve(w, deltas_padded[:, i], mode='valid')

    motion = np.linalg.norm(deltas, axis=1)

    return motion


def smooth_bboxes(detections, center_kernel=25, size_kernel=51, max_motion=0.01):
    """ Temporally smooth a series of bounding boxes by motion estimate.

    Based on the idea of the one Euro filter described in the paper:
    `"1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems"
    <https://dl.acm.org/doi/pdf/10.1145/2207676.2208639>`_

    Args:
        detections (list of np.array): A list of detection bounding boxes in the format [left, top, bottom, right]
        center_kernel (int): The temporal kernel size for smoothing the bounding box centers
        size_kernel (int): The temporal kernel size for smoothing the bounding box sizes
        max_motion (float): The maximum allowed motion (for normalization)

    Returns:
        (list of np.array): The smoothed bounding boxes.
    """
    # Prepare smoothing kernel
    center_w = np.ones(center_kernel)
    center_w /= center_w.sum()
    size_w = np.ones(size_kernel)
    size_w /= size_w.sum()

    # Convert bounding boxes to center and size format
    bboxes = np.array(detections)
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    sizes = bboxes[:, 2:] - bboxes[:, :2]

    # Smooth sizes
    sizes_padded = np.pad(sizes, ((size_kernel // 2, size_kernel // 2), (0, 0)), 'reflect')
    for i in range(centers.shape[1]):
        sizes[:, i] = np.convolve(size_w, sizes_padded[:, i], mode='valid')

    # Estimate motion
    centers_normalized = centers / sizes[:, 1:]
    motion = estimate_motion(centers_normalized, center_kernel)

    # Average smooth centers
    centers_padded = np.pad(centers, ((center_kernel // 2, center_kernel // 2), (0, 0)), 'reflect')
    centers_avg = centers.copy()
    for i in range(centers.shape[1]):
        centers_avg[:, i] = np.convolve(center_w, centers_padded[:, i], mode='valid')

    # Smooth centers by motion
    a = np.minimum(motion / max_motion, 1.)[..., np.newaxis]
    centers_smoothed = centers * a + centers_avg * (1 - a)

    # Change back to detections format
    sizes /= 2.0
    bboxes = np.concatenate((centers_smoothed - sizes, centers_smoothed + sizes), axis=1)

    return bboxes
