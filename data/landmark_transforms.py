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


class LandmarksTransform(object):
    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img, landmarks, bbox


class LandmarksPairTransform(object):
    def __call__(self, img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        """
        Args:
            img1 (PIL Image or numpy.ndarray): First image to transform.
            landmarks1 (numpy.ndarray): First face landmarks (68 X 2)
            bbox1 (numpy.ndarray): First face bounding box (4,)
            img2 (PIL Image or numpy.ndarray): Second image to transform.
            landmarks2 (numpy.ndarray): Second face landmarks (68 X 2)
            bbox2 (numpy.ndarray): Second face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img1, landmarks1, bbox1, img2, landmarks2, bbox2


class Compose(LandmarksTransform):
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

    def __call__(self, img, landmarks=None, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LandmarksTransform):
                img, landmarks, bboxes = t(img, landmarks, bboxes)
            else:
                img = t(img)

        return img, landmarks, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposePair(LandmarksPairTransform):
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
            assert isinstance(t, LandmarksPairTransform)

    def __call__(self,  img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        for t in self.transforms:
            img1, landmarks1, bbox1, img2, landmarks2, bbox2 = t(img1, landmarks1, bbox1, img2, landmarks2, bbox2)

        return img1, landmarks1, bbox1, img2, landmarks2, bbox2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(LandmarksTransform):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` with landmarks to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] with landmarks in numpy.ndarray format (68 x 2) to a torch.FloatTensor of shape
    ((C + 68) x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, landmarks, bbox):
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
        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(LandmarksTransform):
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

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to resize.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2) or (68 X 3) to resize.
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        orig_size = np.array(img.size)
        img = F.resize(img, self.size, self.interpolation)
        axes_scale = (np.array(img.size) / orig_size)

        # 3D landmarks case
        if landmarks.shape[1] == 3:
            axes_scale = np.append(axes_scale, axes_scale.mean())

        landmarks *= axes_scale
        return img, landmarks, bbox

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class LandmarksToHeatmaps(LandmarksTransform):
    """Convert a ``numpy.ndarray`` landmarks vector to heatmaps.

    Converts a numpy.ndarray landmarks vector of (68 X 2) to heatmaps (H x W x 68) in the range [0, 255].
    """
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image.
        """
        landmarks = generate_heatmaps(img.size[1], img.size[0], landmarks, sigma=self.sigma)
        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class FaceAlignCrop(LandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, bbox_scale=2.0, bbox_square=True, align=False):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.align = align

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            PIL Image: Rescaled image.
        """
        img = np.array(img).copy()
        if self.align:
            img, landmarks = align_crop(img, landmarks, bbox, self.bbox_scale, self.bbox_square)
        else:
            bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
            img, landmarks = crop_img(img, landmarks, bbox_scaled)

        img = Image.fromarray(img)

        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class RandomHorizontalFlipPair(LandmarksPairTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, landmarks1, bbox1, img2, landmarks2, bbox2):
        """
        Args:
            img (PIL Image): Image to be flipped.

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

        return img1, landmarks1, bbox1, img2, landmarks2, bbox2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Pyramids(LandmarksTransform):
    """Generate pyramids from an image and it's corresponding landmarks and bounding box

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            Image list (PIL Image list): List of scaled images.
            Landmarks list (numpy array list): List of scaled landmarks.
            Bounding boxes list (numpy array list): List of scaled boudning boxes.
        """
        img_pyd = [img]
        landmarks_pyd = [landmarks]
        bbox_pyd = [bbox]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
            landmarks_pyd.append(landmarks_pyd[-1] / 2)
            bbox_pyd.append(bbox_pyd[-1] / 2)

        return img_pyd, landmarks_pyd, bbox_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


class ComposePyramids(LandmarksTransform):
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

    def __call__(self, img, landmarks=None, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LandmarksTransform):
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i], landmarks[i], bboxes[i] = t(img[i], landmarks[i], bboxes[i])
                else:
                    img, landmarks, bboxes = t(img, landmarks, bboxes)
            else:
                if isinstance(img, list):
                    for i in range(len(img)):
                        img[i] = t(img[i])
                else:
                    img = t(img)

        return img, landmarks, bboxes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string