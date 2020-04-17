""" Face segmentation utilities. """

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def blend_seg_pred(img, seg, alpha=0.5):
    """ Blend images with their corresponding segmentation prediction.

    Args:
        img (torch.Tensor): A batch of image tensors of shape (B, 3, H, W) where B is the batch size,
            H is the images height and W is the images width
        seg (torch.Tensor): A batch of segmentation predictions of shape (B, C, H, W) where B is the batch size,
            C is the number of segmentation classes, H is the images height and W is the images width
        alpha: alpha (float): Opacity value for the segmentation in the range [0, 1] where 0 is completely transparent
            and 1 is completely opaque

    Returns:
        torch.Tensor: The blended image.
    """
    pred = seg.argmax(1)
    pred = pred.view(pred.shape[0], 1, pred.shape[1], pred.shape[2]).repeat(1, 3, 1, 1)
    blend = img

    # For each segmentation class except the background (label 0)
    for i in range(1, seg.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend


def blend_seg_label(img, seg, alpha=0.5):
    """ Blend images with their corresponding segmentation labels.

    Args:
        img (torch.Tensor): A batch of image tensors of shape (B, 3, H, W) where B is the batch size,
            H is the images height and W is the images width
        seg (torch.Tensor): A batch of segmentation labels of shape (B, H, W) where B is the batch size,
            H is the images height and W is the images width
        alpha: alpha (float): Opacity value for the segmentation in the range [0, 1] where 0 is completely transparent
            and 1 is completely opaque

    Returns:
        torch.Tensor: The blended image.
    """
    pred = seg.unsqueeze(1).repeat(1, 3, 1, 1)
    blend = img

    # For each segmentation class except the background (label 0)
    for i in range(1, pred.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend


# TODO: Move this somewhere else later
def random_hair_inpainting_mask(face_mask):
    """ Simulate random hair occlusions on face mask.

    The algorithm works as follows:
        1. The method first randomly choose a y coordinate of the face mask
        2. x coordinate is chosen randomly: Either minimum or maximum x value of the selected line
        3. A random ellipse is rendered with its center in (x, y)
        4. The inpainting map is the intersection of the face mask with the ellipse.

    Args:
        face_mask (np.array): A binary mask tensor of shape (H, W) where `1` means face region
            and `0` means background

    Returns:
        np.array: Result mask.
    """
    mask = face_mask == 1
    inpainting_mask = np.zeros(mask.shape, 'uint8')
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return inpainting_mask
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return inpainting_mask

    # Select a random point on the mask borders
    try:
        y_coords = np.unique(a[0])
        y_ind = np.random.randint(len(y_coords))
        y = y_coords[y_ind]
        x_ind = np.where(a[0] == y)
        x_coords = a[1][x_ind[0]]
        x = x_coords.min() if np.random.rand() > 0.5 else x_coords.max()
    except:
        print(y_coords)
        print(x_coords)

    # Draw inpainting shape
    width = a[1].max() - a[1].min() + 1
    # height = a[0].max() - a[0].min() + 1
    scale = (np.random.randint(width // 4, width // 2), np.random.randint(width // 4, width // 2))
    rotation_angle = np.random.randint(0, 180)
    cv2.ellipse(inpainting_mask, (x, y), scale, rotation_angle, 0, 360, (255, 255, 255), -1, 8)

    # Get inpainting mask by intersection with face mask
    inpainting_mask *= mask
    inpainting_mask = inpainting_mask > 0

    ### Debug ###
    # cv2.imshow('face_mask', inpainting_mask)
    # cv2.waitKey(0)
    #############

    return inpainting_mask


def random_hair_inpainting_mask_tensor(face_mask):
    """ Simulate random hair occlusions on face mask.

    The algorithm works as follows:
        1. The method first randomly choose a y coordinate of the face mask
        2. x coordinate is chosen randomly: Either minimum or maximum x value of the selected line
        3. A random ellipse is rendered with its center in (x, y)
        4. The inpainting map is the intersection of the face mask with the ellipse.

    Args:
        face_mask (torch.Tensor): A binary mask tensor of shape (B, H, W) where `1` means face region
            and `0` means background

    Returns:
        torch.Tensor: Result mask.
    """
    out_tensors = []
    for b in range(face_mask.shape[0]):
        curr_face_mask = face_mask[b]
        inpainting_mask = random_hair_inpainting_mask(curr_face_mask.cpu().numpy())
        out_tensors.append(torch.from_numpy(inpainting_mask.astype(float)).unsqueeze(0))

    return torch.cat(out_tensors, dim=0)


# TODO: Remove this later
def encode_segmentation(segmentation):
    seg_min, seg_max = segmentation.min(), segmentation.max()
    segmentation = segmentation.sub(seg_min).div_((seg_max - seg_min) * 0.5).sub_(1.0)

    return segmentation


def encode_binary_mask(mask):
    """ Encode binary mask using binary PNG encoding.

    Args:
        mask (np.array): Binary mask of shape (H, W)

    Returns:
        bytes: Encoded binary mask.
    """
    mask_pil = Image.fromarray(mask.astype('uint8') * 255, mode='L').convert('1')
    in_mem_file = io.BytesIO()
    mask_pil.save(in_mem_file, format='png')
    in_mem_file.seek(0)

    return in_mem_file.read()


def decode_binary_mask(bytes):
    """ Decode an encoded binary mask.

    Args:
        bytes: Encoded binary mask of shape (H, W)

    Returns:
        np.array: Decoded binary mask.
    """
    return np.array(Image.open(io.BytesIO(bytes)))
    # return np.array(Image.open(io.BytesIO(bytes)).convert('L'))


class SoftErosion(nn.Module):
    """ Applies *soft erosion* on a binary mask, that is similar to the
    `erosion morphology operation <https://en.wikipedia.org/wiki/Erosion_(morphology)>`_,
    returning both a soft mask and a hard binary mask.

    All values greater or equal to the the specified threshold will be set to 1 in both the soft and hard masks,
    the other values will be 0 in the hard mask and will be gradually reduced to 0 in the soft mask.

    Args:
        kernel_size (int): The size of the erosion kernel size
        threshold (float): The erosion threshold
        iterations (int) The number of times to apply the erosion kernel
    """
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        """ Apply the soft erosion operation.

        Args:
            x (torch.Tensor): A binary mask of shape (1, H, W)

        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing:
                - soft_mask (torch.Tensor): The soft mask of shape (1, H, W)
                - hard_mask (torch.Tensor): The hard mask of shape (1, H, W)
        """
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def remove_inner_mouth(seg, landmarks):
    """ Removes the inner part of the mouth, corresponding to the face landmarks, from a binary mask.

    Args:
        seg (np.array): A binary mask of shape (H, W)
        landmarks (np.array): Face landmarks of shape (98, 2)

    Returns:
        np.array: The binary mask with the inner part of the mouth removed.
    """
    size = np.array(seg.shape[::-1])
    mouth_pts = landmarks[88:96] * size
    mouth_pts = np.round(mouth_pts).astype(int)
    out_seg = cv2.fillPoly(seg.astype('uint8'), [mouth_pts], (0, 0, 0))

    return out_seg.astype(seg.dtype)


def main(input_path):
    from PIL import Image
    seg = np.array(Image.open(input_path))
    # while True:
    #     random_hair_inpainting_mask(seg)


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('seg_utils')
    parser.add_argument('input', help='input path')
    args = parser.parse_args()
    main(args.input)