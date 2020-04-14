""" Bounding box utilities. """

import numpy as np
import cv2


def scale_bbox(bbox, scale=2., square=True):
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


def crop_img_with_padding(img, bbox):
    """ Crop image by bounding box.

    If the bounding box is out the image bounds, the image will be padded with a black border in the corresponding
    regions.

    Args:
        img (np.array): An image of shape (H, W, 3)
        bbox (np.array): Bounding box in the format [left, top, width, height]

    Returns:
        np.array: Cropped image
    """
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    img_bbox = bbox.copy()
    if any((left, top, right, bottom)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        img_bbox[0] += left
        img_bbox[1] += top

    return img[img_bbox[1]:img_bbox[1] + img_bbox[3], img_bbox[0]:img_bbox[0] + img_bbox[2]]


def crop_img(img, landmarks, bbox, border=cv2.BORDER_CONSTANT):
    """ Crop image and corresponding landmarks by bounding box.

    If the bounding box is out the image bounds, the image will be padded in the corresponding regions.

    Args:
        img (np.array): An image of shape (H, W, 3)
        landmarks (np.array): Face landmarks points of shape (68, 2)
        bbox (np.array): Bounding box in the format [left, top, width, height]
        border (int): OpenCV's border code

    Returns:
        (np.array, np.array): A tuple of numpy arrays containing:
            - Cropped image (np.array)
            - Cropped landmarks (np.array)
    """
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    img_bbox = bbox.copy()
    if any((left, top, right, bottom)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, border)
        img_bbox[0] += left
        img_bbox[1] += top

    # Adjust landmarks
    new_landmarks = landmarks.copy()
    new_landmarks[:, :2] += (np.array([left, top]) - img_bbox[:2])

    return img[img_bbox[1]:img_bbox[1] + img_bbox[3], img_bbox[0]:img_bbox[0] + img_bbox[2]], new_landmarks


def crop_landmarks(img_size, landmarks, bbox):
    """ Crop landmarks by bounding box.

    Args:
        img_size (tuple of int): The size of the corresponding image in the format [height, width]
        landmarks (np.array): Face landmarks points of shape (68, 2)
        bbox (np.array): Bounding box in the format [left, top, width, height]

    Returns:
        np.array: Cropped face landmarks.
    """
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img_size[1] if (bbox[0] + bbox[2] - img_size[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img_size[0] if (bbox[1] + bbox[3] - img_size[0]) > 0 else 0
    img_bbox = bbox.copy()
    if any((left, top, right, bottom)):
        img_bbox[0] += left
        img_bbox[1] += top

    # Adjust landmarks
    new_landmarks = landmarks.copy()
    new_landmarks[:, :2] += (np.array([left, top]) - img_bbox[:2])

    return new_landmarks


def hflip_bbox(bbox, width):
    """ Horizontal flip bounding box.

    Args:
        bbox (np.array): Bounding box in the format [left, top, width, height]
        width (int): The width of the correspondign image

    Returns:
        np.array: The horizontally flipped bounding box.
    """
    out_bbox = bbox.copy()
    out_bbox[0] = width - out_bbox[2] - out_bbox[0]

    return out_bbox


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
