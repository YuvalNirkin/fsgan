import os
import random
import numpy as np
import cv2
from PIL import Image
from fsgan.data.image_list_dataset import ImageListDataset
from torchvision.datasets.folder import default_loader


def read_landmarks(landmarks_file):
    if landmarks_file.lower().endswith('.npy'):
        return np.load(landmarks_file)
    data = np.loadtxt(landmarks_file, dtype=int, skiprows=1, usecols=range(1, 11), delimiter=',')
    return data.reshape(data.shape[0], -1, 2)


def read_bboxes(bboxes_file):
    if bboxes_file.lower().endswith('.npy'):
        return np.load(bboxes_file)
    data = np.loadtxt(bboxes_file, dtype=int, skiprows=1, usecols=range(1, 5), delimiter=',')
    return data


def crop_img(img, bbox):
    bbox_max = bbox[:2] + bbox[2:] - 1
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    if any((left, top, right, bottom)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        bbox[0] += left
        bbox[1] += top

    return img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]


def scale_bbox(bbox, scale=1.35, square=True):
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def align_crop(img, landmarks, bbox, scale=1.35, square=True):
    # Rotate image for horizontal eyes
    right_eye_center = landmarks[0]
    left_eye_center = landmarks[1]
    eye_center = np.round(np.mean(landmarks[:2], axis=0)).astype(int)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.)
    output = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Scale bounding box
    bbox_scaled = scale_bbox(bbox, scale, square)

    # Crop image
    output = crop_img(output, bbox_scaled)

    return output


class FaceAlignCrop(object):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, bbox_scale=1.35, bbox_square=True, align=True):
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
            img = align_crop(img, landmarks, bbox, self.bbox_scale, self.bbox_square)
        else:
            bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
            img = crop_img(img, bbox_scaled)

        img = Image.fromarray(img)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class FaceListDataset(ImageListDataset):
    """A face specific data loader for loading aligned faces where the images are arranged in this way:

        root/id1/xxx.png
        root/id1/xxy.png
        root/id1/xxz.png

        root/id2/123.png
        root/id2/nsdf3.png
        root/id2/asd932_.png

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, landmarks_list=None, bboxes_list=None, crop_scale=1.35, crop_square=True,
                 align=True, transform=None, target_transform=None, loader=default_loader):
        super(FaceListDataset, self).__init__(root, img_list, transform, target_transform, loader)
        landmarks_list_path = landmarks_list if os.path.exists(landmarks_list) else os.path.join(root, landmarks_list)
        bboxes_list_path = bboxes_list if os.path.exists(bboxes_list) else os.path.join(root, bboxes_list)
        self.landmarks = read_landmarks(landmarks_list_path)
        self.bboxes = read_bboxes(bboxes_list_path)
        if landmarks_list is None or bboxes_list is None:
            self.face_transform = None
        else:
            self.face_transform = FaceAlignCrop(crop_scale, crop_square, align)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.face_transform is not None:
            landmarks = self.landmarks[index]
            bbox = self.bboxes[index]
            img = self.face_transform(img, landmarks, bbox)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FacePairListDataset(FaceListDataset):
    """A face specific data loader for loading aligned face pairs where the images are arranged in this way:

        root/id1/xxx.png
        root/id1/xxy.png
        root/id1/xxz.png

        root/id2/123.png
        root/id2/nsdf3.png
        root/id2/asd932_.png

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, landmarks_list=None, bboxes_list=None, crop_scale=1.35, crop_square=True,
                 align=True, same_prob=0.5, transform=None, target_transform=None, loader=default_loader):
        super(FacePairListDataset, self).__init__(root, img_list, landmarks_list, bboxes_list,
                                                  crop_scale, crop_square, align,
                                                  transform, target_transform, loader)
        self.same_prob = same_prob

        paths, labels = zip(*self.imgs)
        self.label_ranges = [len(self.imgs)] * (len(self.class_to_idx) + 1)
        for i, img in enumerate(self.imgs):
            self.label_ranges[img[1]] = min(self.label_ranges[img[1]], i)

    """
   Args:
       index (int): Index
   Returns:
       tuple: (image1, image2, target1, target2) where target is True for same identity else False.
   """
    def __getitem__(self, index):
        label1 = self.imgs[index][1]
        total_imgs1 = self.label_ranges[label1 + 1] - self.label_ranges[label1]
        if total_imgs1 > 1 and random.random() < self.same_prob:
            # Select another image from the same identity
            index2 = random.randint(self.label_ranges[label1], self.label_ranges[label1 + 1] - 2)
            index2 = index2 + 1 if index2 >= index else index2
        else:
            # Select another image from a different identity
            label2 = random.randint(0, len(self.class_to_idx) - 2)
            label2 = label2 + 1 if label2 >= label1 else label2
            index2 = random.randint(self.label_ranges[label2], self.label_ranges[label2 + 1] - 1)

        img1, label1 = super(FacePairListDataset, self).__getitem__(index)
        img2, label2 = super(FacePairListDataset, self).__getitem__(index2)

        return img1, img2, label1, label2


def main(root_path, img_list, landmarks_list, bboxes_list, bbox_scale=1.0, align=True, pil_transforms=None,
         dataset_type='singles'):
    import torchvision.transforms as transforms
    from fsgan.utils.obj_factory import obj_factory
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    pil_transforms = transforms.Compose(pil_transforms)

    if dataset_type == 'singles':
        dataset = FaceListDataset(root_path, img_list, landmarks_list, bboxes_list, bbox_scale, crop_square=True,
                                  align=align, transform=pil_transforms)
        for img, label in dataset:
            img = np.array(img)[:, :, ::-1].copy()
            cv2.imshow('img', img)
            print('label = ' + str(label))
            cv2.waitKey(0)
    elif dataset_type == 'pairs':
        dataset = FacePairListDataset(root_path, img_list, landmarks_list, bboxes_list, bbox_scale, crop_square=True,
                                      align=align, same_prob=0, transform=pil_transforms)
        for img1, img2, label1, label2 in dataset:
            img1 = np.array(img1)[:, :, ::-1].copy()
            img2 = np.array(img2)[:, :, ::-1].copy()
            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            print('label1 = %d, label2 = %d' % (label1, label2))
            cv2.waitKey(0)
    else:
        raise RuntimeError('Dataset type must be either "singles" or "pairs"!')


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('face_list_dataset')
    parser.add_argument('root_path', help='paths to dataset root directory')
    parser.add_argument('img_list', help='image list file path')
    parser.add_argument('-l', '--landmarks', default=None, help='landmarks file')
    parser.add_argument('-b', '--bboxes', default=None, help='bounding boxes file')
    parser.add_argument('-s', '--scale', default=1.0, type=float, help='crop bounding boxes scale')
    parser.add_argument('-a', '--align', action='store_true', help='align faces')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-d', '--dataset', choices=['singles', 'pairs'], help='dataset type')
    args = parser.parse_args()

    main(args.root_path, args.img_list, args.landmarks, args.bboxes, args.scale, args.align, args.pil_transforms,
         args.dataset)
