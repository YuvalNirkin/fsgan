""" Face segmentation utilities. """

import torch
import numpy as np
import cv2


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


def main(input_path):
    from PIL import Image
    seg = np.array(Image.open(input_path))
    while True:
        random_hair_inpainting_mask(seg)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('seg_utils')
    parser.add_argument('input', help='input path')
    args = parser.parse_args()
    main(args.input)
