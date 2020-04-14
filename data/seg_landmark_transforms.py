import collections
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from fsgan.utils.bbox_utils import scale_bbox, crop_img, hflip_bbox
from fsgan.utils.landmark_utils import generate_heatmaps, hflip_face_landmarks, align_crop


class SegmentationLandmarksTransform(object):
    def __call__(self, img, landmarks, bbox, seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)
            seg (PIL Image or numpy.ndarray): Segmentation to transform.

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img, landmarks, bbox, seg


class SegmentationLandmarksPairTransform(object):
    def __call__(self, img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2):
        """
        Args:
            img1 (PIL Image or numpy.ndarray): First image to transform.
            landmarks1 (numpy.ndarray): First face landmarks (68 X 2)
            bbox1 (numpy.ndarray): First face bounding box (4,)
            seg1 (PIL Image or numpy.ndarray): First segmentation to transform.
            img2 (PIL Image or numpy.ndarray): Second image to transform.
            landmarks2 (numpy.ndarray): Second face landmarks (68 X 2)
            bbox2 (numpy.ndarray): Second face bounding box (4,)
            seg2 (PIL Image or numpy.ndarray): Second segmentation to transform.

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2


class Compose(SegmentationLandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, landmarks=None, bboxes=None, seg=None):
        for t in self.transforms:
            if isinstance(t, SegmentationLandmarksTransform):
                img, landmarks, bboxes, seg = t(img, landmarks, bboxes, seg)
            else:
                img = t(img)

        return img, landmarks, bboxes, seg

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposePair(SegmentationLandmarksPairTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.ComposePair([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        for t in self.transforms:
            assert isinstance(t, SegmentationLandmarksPairTransform)

    def __call__(self, img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2):
        for t in self.transforms:
            img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2 = t(
                img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2)

        return img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(SegmentationLandmarksTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` with landmarks to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] with landmarks in numpy.ndarray format (68 x 2) to a torch.FloatTensor of shape
    ((C + 68) x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, landmarks, bbox, seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        img = F.to_tensor(img)
        landmarks = torch.from_numpy(landmarks)
        bbox = torch.from_numpy(bbox)
        # seg = torch.from_numpy(np.array(seg, dtype='int64')).unsqueeze(0) if seg is not None else None
        seg = torch.from_numpy(np.array(seg, dtype='int64')) if seg is not None else None
        return img, landmarks, bbox, seg

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(SegmentationLandmarksTransform):
    """Resize the input PIL Image and it's corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, img_interp=Image.BICUBIC, seg_interp=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.img_interp = img_interp
        self.seg_interp = seg_interp

    def __call__(self, img, landmarks, bbox, seg=None):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to resize.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2) or (68 X 3) to resize.
            bbox (numpy.ndarray): Face bounding box (4,)
            seg (PIL Image or numpy.ndarray): Segmentation to resize.

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        orig_size = np.array(img.size)
        img = F.resize(img, self.size, self.img_interp)
        axes_scale = (np.array(img.size) / orig_size)

        # 3D landmarks case
        if landmarks.shape[1] == 3:
            axes_scale = np.append(axes_scale, axes_scale.mean())

        landmarks *= axes_scale
        seg = F.resize(seg, self.size, self.seg_interp) if seg is not None else None

        return img, landmarks, bbox, seg

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class LandmarksToHeatmaps(SegmentationLandmarksTransform):
    """Convert a ``numpy.ndarray`` landmarks vector to heatmaps.

    Converts a numpy.ndarray landmarks vector of (68 X 2) to heatmaps (H x W x 68) in the range [0, 255].
    """
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, img, landmarks, bbox, seg):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)
            seg (PIL Image or numpy.ndarray): Segmentation to transform.

        Returns:
            Tensor: Converted image.
        """
        landmarks = generate_heatmaps(img.size[1], img.size[0], landmarks, sigma=self.sigma)
        return img, landmarks, bbox, seg

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class FaceAlignCrop(SegmentationLandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, bbox_scale=2.0, bbox_square=True, align=False, border='constant'):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.align = align
        if border == 'repeat':
            self.border = cv2.BORDER_REPLICATE
        elif border == 'reflect':
            self.border = cv2.BORDER_REFLECT_101
        else:
            self.border = cv2.BORDER_CONSTANT

    def __call__(self, img, landmarks, bbox, seg):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box
            seg (PIL Image or numpy.ndarray): Segmentation to align and crop.

        Returns:
            PIL Image: Rescaled image.
        """
        img = np.array(img).copy()
        seg = np.array(seg).copy() if seg is not None else None
        if self.align:
            img, landmarks = align_crop(img, landmarks, bbox, self.bbox_scale, self.bbox_square)
            seg, _ = align_crop(seg, landmarks, bbox, self.bbox_scale, self.bbox_square)
        else:
            bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
            img, landmarks = crop_img(img, landmarks, bbox_scaled, border=self.border)
            seg, _ = crop_img(seg, landmarks, bbox_scaled) if seg is not None else (None, landmarks)

        img = Image.fromarray(img)
        seg = Image.fromarray(seg) if seg is not None else None

        return img, landmarks, bbox, seg

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class RandomHorizontalFlipPair(SegmentationLandmarksPairTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2):
        """
        Args:
            img1 (PIL Image or numpy.ndarray): First image to be flipped.
            landmarks1 (numpy.ndarray): First face landmarks to be flipped (68 X 2)
            bbox1 (numpy.ndarray): First face bounding box to be flipped (4,)
            seg1 (PIL Image or numpy.ndarray): First segmentation to be flipped.
            img2 (PIL Image or numpy.ndarray): Second image to be flipped.
            landmarks2 (numpy.ndarray): Second face landmarks to be flipped (68 X 2)
            bbox2 (numpy.ndarray): Second face bounding box to be flipped (4,)
            seg2 (PIL Image or numpy.ndarray): Second segmentation to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
            landmarks1 = hflip_face_landmarks(landmarks1, img1.size[0])
            landmarks2 = hflip_face_landmarks(landmarks2, img2.size[0])
            bbox1 = hflip_bbox(bbox1, img1.size[0])
            bbox2 = hflip_bbox(bbox2, img2.size[0])
            seg1 = F.hflip(seg1) if seg1 is not None else None
            seg2 = F.hflip(seg2) if seg2 is not None else None

        return img1, landmarks1, bbox1, seg1, img2, landmarks2, bbox2, seg2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Pyramids(SegmentationLandmarksTransform):
    """Generate pyramids from an image and it's corresponding landmarks and bounding box

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img, landmarks, bbox, seg):
        """
        Args:
            img (PIL Image): Face image to transform.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box
            seg (PIL Image or numpy.ndarray): Segmentation to transform.

        Returns:
            Image list (PIL Image list): List of scaled images.
            Landmarks list (numpy array list): List of scaled landmarks.
            Bounding boxes list (numpy array list): List of scaled boudning boxes.
            Segmentation list (PIL Image list): List of scaled segmentations.
        """
        img_pyd = [img]
        landmarks_pyd = [landmarks]
        bbox_pyd = [bbox]
        seg_pyd = [seg]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
            landmarks_pyd.append(landmarks_pyd[-1] / 2)
            bbox_pyd.append(bbox_pyd[-1] / 2)
            # seg_pyd.append(Image.fromarray(cv2.pyrDown(np.array(seg_pyd[-1]))))
            seg_pyd.append(F.resize(seg_pyd[-1], (img_pyd[-1].size[1], img_pyd[-1].size[0]), Image.NEAREST))

        return img_pyd, landmarks_pyd, bbox_pyd, seg_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


class ComposePyramids(SegmentationLandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> landmark_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, landmarks=None, bboxes=None, seg=None):
        for t in self.transforms:
            if isinstance(t, SegmentationLandmarksTransform):
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i], landmarks[i], bboxes[i], seg[i] = t(img[i], landmarks[i], bboxes[i], seg[i])
                else:
                    img, landmarks, bboxes, seg = t(img, landmarks, bboxes, seg)
            else:
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i] = t(img[i])
                else:
                    img = t(img)

        return img, landmarks, bboxes, seg

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string