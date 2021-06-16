""" The concept is that any 1D numpy array is considered pose and anything else is considered an image.
"""
from typing import Tuple, List, Optional
import collections
import random
import numbers
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from fsgan.utils.landmarks_utils import hflip_face_landmarks_98pts


def interpolation_str2int(interpolation):
    """ Convert interpolation type string to its corresponding OpencCV's id.

    Args:
        interpolation (str or list): Interpolation type ['cubic' | 'linear' | 'nearest']. If a list is provided,
            it will be processed recursively

    Returns:
        interpolation_id (int or list): The matching OpenCV's id or a list of ids.

    Raises:
        RuntimeError: If an unknown interpolation type was provided.
    """
    if isinstance(interpolation, (list, tuple)):
        return [interpolation_str2int(i) for i in interpolation]
    if interpolation == 'cubic':
        return cv2.INTER_CUBIC
    elif interpolation == 'linear':
        return cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        return cv2.INTER_NEAREST
    else:
        raise RuntimeError(f'Unknown interpolation type: "{interpolation}"')


def border_str2int(border):
    """ Convert border type string to its corresponding OpencCV's id.

    Args:
        border (str or list): Border type ['repeat' | 'reflect' | 'constant']. If a list is provided,
            it will be processed recursively

    Returns:
        border_id (int or list): The matching OpenCV's id or a list of ids.

    Raises:
        RuntimeError: If an unknown border type was provided.
    """
    if isinstance(border, (list, tuple)):
        return [border_str2int(b) for b in border]
    if border == 'repeat':
        return cv2.BORDER_REPLICATE
    elif border == 'reflect':
        return cv2.BORDER_REFLECT_101
    elif border == 'constant':
        return cv2.BORDER_CONSTANT
    else:
        raise RuntimeError(f'Unknown border type: "{border}"')


def call_recursive(f, x):
    return [call_recursive(f, y) for y in x] if isinstance(x, (list, tuple)) else f(x)


def is_img(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 3


def is_landmarks(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 2 and x.shape[1] == 2


def is_pose(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 1 and x.size == 3


def is_bbox(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 1 and x.size == 4


def is_binary_mask(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 2 and x.dtype == bool


class RecursiveTransform(object):
    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        return x


class Compose(object):
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

    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)
        Returns:
            Tensor or list of Tensor: Transformed images or poses
        """
        for t in self.transforms:
            if isinstance(t, RecursiveTransform):
                x = t(x)
            else:
                x = call_recursive(t, x)

        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Crop(RecursiveTransform):
    """ Crop images by bounding boxes.
    Input must be specified as tuples (or lists) of images and corresponding bounding boxes.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
    """

    def __init__(self, bbox_scale=1.2, bbox_square=True, det_format=True, border='constant', value=None):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.det_format = det_format
        self.border = border
        self.border_id = border_str2int(border)
        self.value = value

    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and is_img(x[0]) and is_bbox(x[1]):
                # Found image and bounding box pair
                img, bbox = x
                if self.det_format:
                    bbox = np.concatenate((bbox[:2], bbox[2:] - bbox[:2]))
                bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
                return crop_img(img, bbox_scaled, border=self.border_id, value=self.value)
            else:
                return [self.__call__(a) for a in x]

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, det_format={2}, border={3})'.format(
            self.bbox_scale, self.bbox_square, self.det_format, self.border)


class Resize(RecursiveTransform):
    """Resize the input image and its corresponding landmarks to the given size.

    Args:
        size (sequence or int): Desired output size.
        interpolation (str, optional): Desired interpolation. Default is ``cubic``
    """

    def __init__(self, size, interpolation='cubic'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size if isinstance(size, collections.Iterable) else (size, size)
        self.interpolation = interpolation
        self.interpolation_id = interpolation_str2int(interpolation)

    def __call__(self, x, interpolation=None):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        interpolation = self.interpolation_id if interpolation is None else interpolation
        if isinstance(x, (list, tuple)):
            if isinstance(interpolation, list):
                assert len(x) == len(interpolation)
                return [self.__call__(a, interpolation[i]) for i, a in enumerate(x)]
            else:
                return [self.__call__(a, interpolation) for a in x]
        elif is_img(x):  # x is an image
            interpolation = interpolation[0] if isinstance(interpolation, list) else interpolation
            x = cv2.resize(x, (self.size[1], self.size[0]), interpolation=interpolation)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class ToTensor(object):
    """ Convert an image and pose in numpy.ndarray format to Tensor.

    Convert a numpy.ndarray image (H x W x C) in the range [0, 255] and numpy.ndarray pose (3)
    to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] and torch.FloatTensor of shape (3)
    correspondingly.
    """

    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1 or len(x.shape) == 2:   # x is a pose vector or landmarks points
                x = torch.from_numpy(x)
            else:   # x is an image
                x = torch.from_numpy(x.copy().transpose((2, 0, 1))).float().mul_(1 / 255)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:     # x is an image
            x = F.normalize(x, self.mean, self.std, self.inplace)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(RecursiveTransform):
    """Horizontally flip the given image and its corresponding landmarks randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, flip=None):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        flip = random.random() < self.p if flip is None else flip
        if not flip:
            return x

        if isinstance(x, (list, tuple)):
            x = [self.__call__(a, flip) for a in x]
        elif is_pose(x):            # x is a pose vector
            x[0] *= -1.     # Flip the yaw angle
        elif is_landmarks(x):       # x is landmarks points
            x = hflip_face_landmarks_98pts(x)
        elif is_img(x):             # x is an image
            x = cv2.flip(x, 1)
        elif is_binary_mask(x):     # x is a binary mask
            x = cv2.flip(x.astype('uint8'), 1).astype(bool)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Pyramids(object):
    """Generate pyramids from an image.

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """
    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        if is_img(x):     # x is an image
            x = [x]
            for i in range(self.levels - 1):
                x.append(cv2.pyrDown(x[-1]))

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


def rotate_img_landmarks(angle, img, bbox=None, landmarks=None, interpolation=cv2.INTER_CUBIC,
                         border=cv2.BORDER_CONSTANT, value=None, det_format=True):
    if bbox is None:
        center = np.array(img.shape[1::-1]) * 0.5
    else:
        center = (bbox[:2] + bbox[2:]) * 0.5 if det_format else bbox[:2] + bbox[2:] * 0.5
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    out_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interpolation, borderMode=border,
                             borderValue=value)

    # Adjust landmarks
    if landmarks is not None:
        scale = np.array(img.shape[1::-1])
        out_landmarks = np.concatenate((landmarks * scale, np.ones((landmarks.shape[0], 1))), axis=1)
        out_landmarks = out_landmarks.dot(M.transpose())
        out_landmarks /= scale
        out_landmarks = out_landmarks.astype('float32')
    else:
        out_landmarks = None

    if out_landmarks is None:
        return out_img
    else:
        return out_img, out_landmarks


def rotate_img_landmarks_mask(angle, img, bbox=None, landmarks=None, mask=None, interpolation=cv2.INTER_CUBIC,
                              border=cv2.BORDER_CONSTANT, value=None, det_format=True):
    if bbox is None:
        center = np.array(img.shape[1::-1]) * 0.5
    else:
        center = (bbox[:2] + bbox[2:]) * 0.5 if det_format else bbox[:2] + bbox[2:] * 0.5
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    out_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interpolation, borderMode=border,
                             borderValue=value)

    # Adjust landmarks
    if landmarks is not None:
        scale = np.array(img.shape[1::-1])
        out_landmarks = np.concatenate((landmarks * scale, np.ones((landmarks.shape[0], 1))), axis=1)
        out_landmarks = out_landmarks.dot(M.transpose())
        out_landmarks /= scale
        out_landmarks = out_landmarks.astype('float32')
    else:
        out_landmarks = None

    # Rotate mask
    if mask is not None:
        out_mask = cv2.warpAffine(mask.astype('uint8'), M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                                  borderMode=border, borderValue=value).astype(bool)
    else:
        out_mask = None

    if out_landmarks is None:
        if out_mask is None:
            return out_img
        else:
            return out_img, out_mask
    elif out_mask is None:
        return out_img, out_landmarks

    return out_img, out_landmarks, out_mask


class Rotate(RecursiveTransform):
    """ Rotate images to target angle.
    Input must be specified as tuples (or lists) of images, optional landmarks and corresponding angles.

    Args:
        interpolation (str, optional): Desired interpolation type. Default is ``cubic``
        border (str, optional): Desired border type. Default is ``constant``
    """

    def __init__(self, interpolation='cubic', border='constant', value=None):
        self.interpolation = interpolation
        self.interpolation_id = interpolation_str2int(interpolation)
        self.border = border
        self.border_id = border_str2int(border)
        self.value = value

    def __call__(self, x):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and is_img(x[0]) and isinstance(x[1], float):
                # Found image and angle pair
                img, angle = x
                return rotate_img_landmarks(angle, img, interpolation=self.interpolation_id, border=self.border_id,
                                            value=self.value)
            elif len(x) == 3 and is_img(x[0]) and is_landmarks(x[1]) and isinstance(x[2], float):
                # Found image, landmarks, and angle triplet
                img, landmarks, angle = x
                return rotate_img_landmarks(angle, img, landmarks=landmarks, interpolation=self.interpolation_id,
                                            border=self.border_id, value=self.value)
            else:
                return [self.__call__(a) for a in x]

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(interpolation={0}, border={1})'.format(self.interpolation, self.border)


class RandomRotation(RecursiveTransform):
    """ Randomly rotate images and landmarks.

    Args:
        max_degrees (float): Maximum rotation angle [degrees]
        interpolation (str, optional): Desired interpolation type. Default is ``cubic``
        border (str, optional): Desired border type. Default is ``constant``
        value (float, optional): Value used in case of a constant border. By default, it is 0
        randomize_per_image (bool): If True, the jittering parameters will be randomized per image else they will be
            randomized once per call
    """
    def __init__(self, max_degrees=30.0, interpolation='cubic', border='constant', value=None,
                 randomize_per_image=False):
        assert max_degrees > 0.0
        self.max_degrees = max_degrees
        self.interpolation = interpolation
        self.interpolation_id = interpolation_str2int(interpolation)
        self.border = border
        self.border_id = border_str2int(border)
        self.value = value
        self.randomize_per_image = randomize_per_image

    def __call__(self, x, angle=None):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or landmarks
        """
        if angle is None or self.randomize_per_image:
            angle = (random.random() * 2.0 - 1.0) * self.max_degrees
        if isinstance(x, (list, tuple)):
            x = list(x) if isinstance(x, tuple) else x
            for i in range(len(x)):
                if is_img(x[i]):
                    angle = (random.random() * 2.0 - 1.0) * self.max_degrees if self.randomize_per_image else angle
                    if (i + 2) < len(x) and is_landmarks(x[i + 1]) and is_binary_mask(x[i + 2]):
                        # Found image, landmarks, and mask triplet
                        x[i], x[i + 1], x[i + 2] = rotate_img_landmarks_mask(
                            angle, x[i], landmarks=x[i + 1], mask=x[i + 2], interpolation=self.interpolation_id,
                            border=self.border_id, value=self.value)
                    elif (i + 1) < len(x) and is_landmarks(x[i + 1]):     # Found image and landmarks pair
                        x[i], x[i + 1] = rotate_img_landmarks(
                            angle, x[i], landmarks=x[i + 1], interpolation=self.interpolation_id,
                            border=self.border_id, value=self.value)
                    else:   # Found image
                        x[i] = rotate_img_landmarks(angle, x[i], interpolation=self.interpolation_id,
                                                    border=self.border_id, value=self.value)
                elif isinstance(x[i], (list, tuple)):
                    x[i] = [self.__call__(a, angle) for a in x]
        elif is_img(x):
            x = rotate_img_landmarks(angle, x, interpolation=self.interpolation_id, border=self.border_id,
                                     value=self.value)

        return x


class RandomGaussianBlur(RecursiveTransform):
    """Applies Gaussian blur filter on the given image.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, kernel_size=5, sigma=0, randomize_per_image=False):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.randomize_per_image = randomize_per_image

    def __call__(self, x, blur=None):
        blur = random.random() < self.p if blur is None else blur
        if is_img(x):     # x is an image
            return cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), self.sigma) if blur else x
        elif isinstance(x, (list, tuple)):
            return [self.__call__(a, None if self.randomize_per_image else blur) for a in x]

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p}, kernel_size={self.kernel_size}, sigma={self.sigma}, ' \
               f'randomize_per_image={self.randomize_per_image})'


# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
class ColorJitter(RecursiveTransform):
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
        randomize_per_image (bool): If True, the jittering parameters will be randomized per image else they will be
            randomized once per call
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, randomize_per_image=False):
        super(ColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.randomize_per_image = randomize_per_image

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, x, params=None):
        """
        Args:
            x (numpy.ndarray or list): Image (H x W x C) or pose (3) or bounding box (4)

        Returns:
            numpy.ndarray or list: Transformed images.
        """
        if params is None:
            params = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        if is_img(x):     # x is an image
            x = Image.fromarray(x)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    x = F.adjust_brightness(x, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    x = F.adjust_contrast(x, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    x = F.adjust_saturation(x, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    x = F.adjust_hue(x, hue_factor)

            return np.array(x)
        elif isinstance(x, (list, tuple)):
            return [self.__call__(a, None if self.randomize_per_image else params) for a in x]

        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', randomize_per_image={0})'.format(self.randomize_per_image)
        return format_string


def main(input, np_transforms=None, tensor_transforms=None, batch_size=4):
    from torchvision.transforms import Compose
    from fsgan.utils.obj_factory import obj_factory

    np_transforms = obj_factory(np_transforms) if np_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = Compose(np_transforms + tensor_transforms)

    img = cv2.imread(input)
    pose = np.array([1., 2., 3.])

    x = img_transforms((img, pose))
    pass


if __name__ == "__main__":
    # Parse program arguments
    import os
    import argparse
    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument('input', metavar='PATH',
                        help='path to input image')
    parser.add_argument('-nt', '--np_transforms', nargs='+', help='Numpy transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    main(**vars(parser.parse_args()))
