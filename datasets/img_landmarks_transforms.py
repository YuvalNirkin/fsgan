import collections
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from fsgan.utils.landmarks_utils import hflip_face_landmarks_68pts


def interpolation_str2int(interpolation):
    if isinstance(interpolation, (list, tuple)):
        return [interpolation_str2int(i) for i in interpolation]
    if interpolation == 'cubic':
        return cv2.INTER_CUBIC
    elif interpolation == 'linear':
        return cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        return cv2.INTER_NEAREST
    else:
        raise RuntimeError('Unknown interpolation type: "%s"' % interpolation)


def call_recursive(f, x):
    return [call_recursive(f, y) for y in x] if isinstance(x, (list, tuple)) else f(x)


class ImgLandmarksTransform(object):
    def process(self, img_list, landmarks_list=None):
        return img_list, landmarks_list

    def __call__(self, img, landmarks=None):
        """
        Args:
            img (numpy.ndarray or list of numpy.ndarray): Image to transform (H x W x C)
            landmarks (numpy.ndarray or list of numpy.ndarray): Array of landmarks (N X D)

        Returns:
            Tensor or list of Tensor: Transformed images
            Tensor or list of Tensor: Transformed landmarks
        """
        # Validate input
        img_list = img if isinstance(img, (list, tuple)) else [img]
        if landmarks is None:
            landmarks_list = None
        else:
            landmarks_list = landmarks if isinstance(landmarks, (list, tuple)) else [landmarks]
            assert len(img_list) == len(landmarks_list)

        # Process validated image and landmarks lists
        img_list, landmarks_list = self.process(img_list, landmarks_list)

        # Return transformed output
        if isinstance(img, (list, tuple)) or isinstance(landmarks, (list, tuple)):
            return img_list, landmarks_list
        else:
            return img_list[0], landmarks_list[0] if landmarks_list is not None else None


class Compose(ImgLandmarksTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> img_landmarks_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox=None, landmarks=None):
        """
        Args:
            img (numpy.ndarray or list of numpy.ndarray): Image to transform (H x W x C)
            bbox (numpy.ndarray or list of numpy.ndarray): Bounding box (4,)
            landmarks (numpy.ndarray or list of numpy.ndarray): Array of landmarks (N X D)
        Returns:
            Tensor or list of Tensor: Transformed images
            Tensor or list of Tensor: Transformed landmarks
        """
        for t in self.transforms:
            if isinstance(t, (Crop, RandomRotation)):
                img, landmarks = t(img, bbox, landmarks)
            elif isinstance(t, ImgLandmarksTransform):
                img, landmarks = t(img, landmarks)
            else:
                img = call_recursive(t, img)
                # img = t(img)
        if landmarks is None:
            return img
        else:
            return img, landmarks

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Crop(ImgLandmarksTransform):
    """ Crop images and landmarks by bounding boxes.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
    """

    def __init__(self, bbox_scale=1.2, bbox_square=True, det_format=True, border='constant', value=None):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.det_format = det_format
        if border == 'repeat':
            self.border = cv2.BORDER_REPLICATE
        elif border == 'reflect':
            self.border = cv2.BORDER_REFLECT_101
        else:
            self.border = cv2.BORDER_CONSTANT
        self.value = value

    def process(self, img_list, bbox_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], _ = self.process(img_list[i], bbox_list[i])
                else:
                    img_list[i], landmarks_list[i] = self.process(img_list[i], bbox_list[i], landmarks_list[i])
            else:
                if self.det_format:
                    bbox = np.concatenate((bbox_list[i][:2], bbox_list[i][2:] - bbox_list[i][:2]))
                else:
                    bbox = bbox_list[i]
                bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
                if landmarks_list is None:
                    img_list[i] = crop_img(img_list[i], bbox_scaled, border=self.border, value=self.value)
                else:
                    img_list[i], landmarks_list[i] = crop_img(img_list[i], bbox_scaled, landmarks_list[i], self.border,
                                                              self.value)

        return img_list, landmarks_list

    def __call__(self, img, landmarks=None):
        raise RuntimeError('Bounding box must be specified!')

    def __call__(self, img, bbox, landmarks=None):
        """
        Args:
            img (numpy.ndarray or list of numpy.ndarray): Image to transform (H x W x C)
            bbox (numpy.ndarray or list of numpy.ndarray): Bounding box (4,)
            landmarks (numpy.ndarray or list of numpy.ndarray): Array of landmarks (N X D)
        Returns:
            Tensor or list of Tensor: Transformed images
            Tensor or list of Tensor: Transformed landmarks
        """
        # Validate input
        img_list = img if isinstance(img, (list, tuple)) else [img]
        bbox_list = bbox if isinstance(bbox, (list, tuple)) else [bbox]
        if landmarks is None:
            landmarks_list = None
            assert len(img_list) == len(bbox_list)
        else:
            landmarks_list = landmarks if isinstance(landmarks, (list, tuple)) else [landmarks]
            assert len(img_list) == len(bbox_list) == len(landmarks_list)

        # Process validated image and landmarks lists
        img_list, landmarks_list = self.process(img_list, bbox_list, landmarks_list)

        # Return transformed output
        if isinstance(img, (list, tuple)) or isinstance(landmarks, (list, tuple)):
            return img_list, landmarks_list
        else:
            return img_list[0], landmarks_list[0] if landmarks_list is not None else None

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1})'.format(self.bbox_scale, self.bbox_square)


class ToTensor(ImgLandmarksTransform):
    """ Convert an image and landmarks in numpy.ndarray format to Tensor.

    Convert a numpy.ndarray image (H x W x C) in the range [0, 255] and numpy.ndarray landmarks (N X D)
    to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] and torch.FloatTensor of shape (N X D)
    correspondingly.
    """

    def process(self, img_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], _ = self.process(img_list[i])
                else:
                    img_list[i], landmarks_list[i] = self.process(img_list[i], landmarks_list[i])
            else:
                # img_list[i] = torch.from_numpy(img_list[i].transpose((2, 0, 1))).float().mul_(1 / 255)
                # Note: The copy is only necessary when the crop and resize transfroms are absent
                img_list[i] = torch.from_numpy(img_list[i].copy().transpose((2, 0, 1))).float().mul_(1 / 255)
                if landmarks_list is not None:
                    landmarks_list[i] = torch.from_numpy(landmarks_list[i]) if landmarks_list[i] is not None else None

        return img_list, landmarks_list

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(ImgLandmarksTransform):
    """Resize the input image and its corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size.
        interpolation (str, optional): Desired interpolation. Default is ``cubic``
    """

    def __init__(self, size, interpolation='cubic'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size if isinstance(size, collections.Iterable) else (size, size)
        self.interpolation = interpolation_str2int(interpolation)

    def process(self, img_list, landmarks_list=None):
        return self._resize_recursive(img_list, landmarks_list, self.interpolation)

    def _resize_recursive(self, img_list, landmarks_list=None, interpolation=cv2.INTER_CUBIC):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            curr_interpolation = interpolation[i] if isinstance(interpolation, (list, tuple)) else interpolation
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], _ = self._resize_recursive(img_list[i], interpolation=curr_interpolation)
                else:
                    img_list[i], landmarks_list[i] = self._resize_recursive(
                        img_list[i], landmarks_list[i], curr_interpolation)
            else:
                orig_size = np.array((img_list[i].shape[1], img_list[i].shape[0]))
                img_list[i] = cv2.resize(img_list[i], (self.size[1], self.size[0]), interpolation=curr_interpolation)
                if landmarks_list is not None:
                    axes_scale = (np.array((img_list[i].shape[1], img_list[i].shape[0])) / orig_size)

                    # 3D landmarks case
                    if landmarks_list[i].shape[1] == 3:
                        axes_scale = np.append(axes_scale, axes_scale.mean())

                    landmarks_list[i] *= axes_scale

        return img_list, landmarks_list

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class RandomHorizontalFlip(ImgLandmarksTransform):
    """Horizontally flip the given image and its corresponding landmarks randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def process(self, img_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        if random.random() < self.p:
            return self._flip_recursive(img_list, landmarks_list)

        return img_list, landmarks_list

    def _flip_recursive(self, img_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], _ = self._flip_recursive(img_list[i])
                else:
                    img_list[i], landmarks_list[i] = self._flip_recursive(img_list[i], landmarks_list[i])
            else:
                img_list[i] = cv2.flip(img_list[i], 1)
                if landmarks_list is not None and landmarks_list[i] is not None:
                    landmarks_list[i] = hflip_face_landmarks_68pts(landmarks_list[i], img_list[i].shape[1])

        return img_list, landmarks_list

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Pyramids(ImgLandmarksTransform):
    """Generate pyramids from an image and its corresponding landmarks and bounding box.

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def process(self, img_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], _ = self.process(img_list[i])
                else:
                    img_list[i], landmarks_list[i] = self.process(img_list[i], landmarks_list[i])
            else:
                img_pyd = [img_list[i]]
                landmarks_pyd = [landmarks_list[i]] if landmarks_list is not None else None
                for j in range(self.levels - 1):
                    img_pyd.append(cv2.pyrDown(img_pyd[-1]))
                    if landmarks_list is not None and landmarks_list[i] is not None:
                        landmarks_pyd.append(landmarks_pyd[-1] / 2)

                img_list[i] = img_pyd
                if landmarks_list is not None and landmarks_list[i] is not None:
                    landmarks_list[i] = landmarks_pyd

        return img_list, landmarks_list

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


def rotate_img_landmarks(angle, bbox, img, landmarks=None, interpolation=cv2.INTER_CUBIC, border=cv2.BORDER_CONSTANT,
                         value=None, det_format=True):
    if det_format:
        center = (bbox[:2] + bbox[2:]) * 0.5
    else:
        center = bbox[:2] + bbox[2:] * 0.5
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    out_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interpolation, borderMode=border,
                             borderValue=value)

    # if out_img.dtype == 'float32':
    #     out_img = np.clip(out_img, 0., 255.)

    # Adjust landmarks
    if landmarks is not None:
        out_landmarks = np.concatenate((landmarks, np.ones((68, 1))), axis=1)
        out_landmarks = out_landmarks.dot(M.transpose())
    else:
        out_landmarks = None

    return out_img, out_landmarks


class RandomRotation(ImgLandmarksTransform):
    def __init__(self, max_degrees=30.0, interpolation='cubic', det_format=True):
        assert max_degrees > 0.0
        self.max_degrees = max_degrees
        self.interpolation = interpolation_str2int(interpolation)
        self.det_format = det_format

    def process(self, img_list, bbox_list, landmarks_list=None):
        # For each input image and corresponding landmarks
        angle = (random.random() * 2.0 - 1.0) * self.max_degrees
        return self._rotate_recursive(img_list, bbox_list, landmarks_list, angle, self.interpolation)

    def _rotate_recursive(self, img_list, bbox_list, landmarks_list=None, angle=0.0, interpolation=cv2.INTER_CUBIC):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            curr_interpolation = interpolation[i] if isinstance(interpolation, (list, tuple)) else interpolation
            if isinstance(img_list[i], (list, tuple)):
                if landmarks_list is None:
                    img_list[i], bbox_list[i], _ = self._rotate_recursive(
                        img_list[i], bbox_list[i], angle=angle, interpolation=curr_interpolation)
                else:
                    img_list[i], bbox_list[i], landmarks_list[i] = self._rotate_recursive(
                        img_list[i], bbox_list[i], landmarks_list[i], angle=angle, interpolation=curr_interpolation)
            else:
                if landmarks_list is not None and landmarks_list[i] is not None:
                    img_list[i], landmarks_list[i] = rotate_img_landmarks(
                        angle, bbox_list[i], img_list[i], landmarks_list[i], curr_interpolation)
                else:
                    img_list[i], _ = rotate_img_landmarks(
                        angle, bbox_list[i], img_list[i], interpolation=curr_interpolation)

        return img_list, bbox_list, landmarks_list

    def __call__(self, img, landmarks=None):
        raise RuntimeError('Bounding box must be specified!')

    def __call__(self, img, bbox, landmarks=None):
        """
        Args:
            img (numpy.ndarray or list of numpy.ndarray): Image to transform (H x W x C)
            bbox (numpy.ndarray or list of numpy.ndarray): Bounding box (4,)
            landmarks (numpy.ndarray or list of numpy.ndarray): Array of landmarks (N X D)
        Returns:
            Tensor or list of Tensor: Transformed images
            Tensor or list of Tensor: Transformed landmarks
        """
        # Validate input
        img_list = img if isinstance(img, (list, tuple)) else [img]
        bbox_list = bbox if isinstance(bbox, (list, tuple)) else [bbox]
        if landmarks is None:
            landmarks_list = None
            assert len(img_list) == len(bbox_list)
        else:
            landmarks_list = landmarks if isinstance(landmarks, (list, tuple)) else [landmarks]
            assert len(img_list) == len(bbox_list) == len(landmarks_list)

        # Process validated image and landmarks lists
        img_list, _, landmarks_list = self.process(img_list, bbox_list, landmarks_list)

        # Return transformed output
        if isinstance(img, (list, tuple)) or isinstance(landmarks, (list, tuple)):
            return img_list, landmarks_list
        else:
            return img_list[0], landmarks_list[0] if landmarks_list is not None else None

    def __repr__(self):
        return self.__class__.__name__ + '(max_degrees={0})'.format(self.max_degrees)


class RandomGaussianBlur(ImgLandmarksTransform):
    """Applies Gaussian blur filter on the given image.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, kernel_size=5, sigma=0, filter=None):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.filter = filter

    def process(self, img_list, landmarks_list=None):
        # For each input image
        if random.random() < self.p:
            img_list = self._blur_recursive(img_list, self.filter)

        return img_list, landmarks_list

    def _blur_recursive(self, img_list, filter=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                img_list[i] = self._blur_recursive(img_list[i])
            elif filter is None or filter[i]:
                img_list[i] = cv2.GaussianBlur(img_list[i], (self.kernel_size, self.kernel_size), self.sigma)

        return img_list

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ColorJitter(ImgLandmarksTransform):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, filter=None):
        self._transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.filter = filter

    def process(self, img_list, landmarks_list=None):
        # For each input image
        img_list = self._apply_recursive(img_list, self.filter)

        return img_list, landmarks_list

    def _apply_recursive(self, img_list, filter=None):
        # For each input image and corresponding landmarks
        for i in range(len(img_list)):
            if isinstance(img_list[i], (list, tuple)):
                img_list[i] = self._apply_recursive(img_list[i])
            elif filter is None or filter[i]:
                img_list[i] = np.array(self._transform(Image.fromarray(img_list[i])))

        return img_list
