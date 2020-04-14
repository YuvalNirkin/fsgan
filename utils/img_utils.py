""" Image utilities. """

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms.functional as F


def rgb2tensor(img, normalize=True):
    """ Converts a RGB image to tensor.

    Args:
        img (np.array or list of np.array): RGB image of shape (H, W, 3) or a list of images
        normalize (bool): If True, the tensor will be normalized to the range [-1, 1]

    Returns:
        torch.Tensor or list of torch.Tensor: The converted image tensor or a list of converted tensors.
    """
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def bgr2tensor(img, normalize=True):
    """ Converts a BGR image to tensor.

    Args:
        img (np.array or list of np.array): BGR image of shape (H, W, 3) or a list of images
        normalize (bool): If True, the tensor will be normalized to the range [-1, 1]

    Returns:
        torch.Tensor or list of torch.Tensor: The converted image tensor or a list of converted tensors.
    """
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(o, normalize) for o in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2rgb(img_tensor):
    """ Convert an image tensor to a numpy RGB image.

    Args:
        img_tensor (torch.Tensor): Tensor image of shape (3, H, W)

    Returns:
        np.array: RGB image of shape (H, W, 3)
    """
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def tensor2bgr(img_tensor):
    """ Convert an image tensor to a numpy BGR image.

    Args:
        img_tensor (torch.Tensor): Tensor image of shape (3, H, W)

    Returns:
        np.array: BGR image of shape (H, W, 3)
    """
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]

    return output_img


def make_grid(*args, cols=8):
    """ Create an image grid from a batch of images.

    Args:
        *args: (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size
        cols: The maximum number of columns in the grid

    Returns:
        torch.Tensor: The grid of images.
    """
    assert len(args) > 0, 'At least one input tensor must be given!'
    imgs = torch.cat([a.cpu() for a in args], dim=2)

    return torchvision.utils.make_grid(imgs, nrow=cols, normalize=True, scale_each=False)


def create_pyramid(img, n=1):
    """ Create an image pyramid.

    Args:
        img (torch.Tensor): An image tensor of shape (B, C, H, W)
        n (int): The number of pyramids to create

    Returns:
        list of torch.Tensor: The computed image pyramid.
    """
    # If input is a list or tuple return it as it is (probably already a pyramid)
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd
