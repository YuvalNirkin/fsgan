import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.datasets.folder import default_loader
from fsgan.data.image_list_dataset import ImageListDataset
from fsgan.data.landmark_transforms import LandmarksTransform
from fsgan.utils.obj_factory import obj_factory
import fsgan.data.landmark_transforms as landmark_transforms


def read_landmarks(landmarks_file):
    if landmarks_file is None:
        return None
    if landmarks_file.lower().endswith('.npy'):
        return np.load(landmarks_file)
    data = np.loadtxt(landmarks_file, dtype=int, skiprows=1, usecols=range(1, 11), delimiter=',')
    return data.reshape(data.shape[0], -1, 2)


def read_bboxes(bboxes_file):
    if bboxes_file is None:
        return None
    if bboxes_file.lower().endswith('.npy'):
        return np.load(bboxes_file)
    data = np.loadtxt(bboxes_file, dtype=int, skiprows=1, usecols=range(1, 5), delimiter=',')
    return data


class FaceLandmarksDataset(ImageListDataset):
    """A face list dataset with landmarks and bounding boxes.

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path, absolute or relative to root (.txt).
        landmarks_list (string): Landmarks list file path, absolute or relative to root (.txt r .npy).
        bboxes_list (string): Bounding boxes list file path, absolute or relative to root (.txt or .npy).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
        landmarks (numpy.ndarray): Array of landmarks per image.
        bboxes (numpy.ndarray): Array of bounding boxes per image.
    """
    def __init__(self, root, img_list, landmarks_list=None, bboxes_list=None,
                 transform=None, target_transform=None, loader=default_loader):
        super(FaceLandmarksDataset, self).__init__(root, img_list, transform, target_transform, loader)
        landmarks_list_path = landmarks_list if os.path.exists(landmarks_list) else os.path.join(root, landmarks_list)
        bboxes_list_path = bboxes_list if os.path.exists(bboxes_list) else os.path.join(root, bboxes_list)
        self.landmarks = read_landmarks(landmarks_list_path)
        self.bboxes = read_bboxes(bboxes_list_path)
        if self.transform is not None:
            assert isinstance(self.transform, landmark_transforms.LandmarksTransform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        landmarks = self.landmarks[index] if self.landmarks is not None else None
        bbox = self.bboxes[index] if self.bboxes is not None else None

        if self.transform is not None:
            img, landmarks, bbox = self.transform(img, landmarks, bbox)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, landmarks, bbox, target


class FacePairLandmarksDataset(FaceLandmarksDataset):
    """A face list dataset for loading face pairs with landmarks and bounding boxes.

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path, absolute or relative to root (.txt).
        landmarks_list (string): Landmarks list file path, absolute or relative to root (.txt or .npy).
        bboxes_list (string): Bounding boxes list file path, absolute or relative to root (.txt or .npy).
        pair_transform (callable, optional): A function/transform that  takes in the first PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        transform1 (callable, optional): A function/transform that  takes in the first PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        transform2 (callable, optional): A function/transform that  takes in the second PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        same_prob (float): The probability that the selected pair will be from the same identity [0..1].
        face_target (bool): If true the target will be the second face else the targets will be the face labels.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
        landmarks (numpy.ndarray): Array of landmarks per image.
        bboxes (numpy.ndarray): Array of bounding boxes per image.
    """
    def __init__(self, root, img_list, landmarks_list=None, bboxes_list=None, pair_transform=None,
                 transform1=None, transform2=None, target_transform=None, loader=default_loader, same_prob=0.5):
        super(FacePairLandmarksDataset, self).__init__(root, img_list, landmarks_list, bboxes_list,
                                                       None, target_transform, loader)
        self.pair_transform = pair_transform
        self.transform1 = transform1
        self.transform2 = transform2
        self.same_prob = same_prob

        # Validate transforms
        if self.pair_transform is not None:
            assert isinstance(self.pair_transform, landmark_transforms.LandmarksPairTransform)
        if self.transform1 is not None:
            assert isinstance(self.transform1, landmark_transforms.LandmarksTransform)
        if self.transform2 is not None:
            assert isinstance(self.transform2, landmark_transforms.LandmarksTransform)

        # Calculate label ranges
        paths, labels = zip(*self.imgs)
        self.label_ranges = [len(self.imgs)] * (len(self.class_to_idx) + 1)
        for i, img in enumerate(self.imgs):
            self.label_ranges[img[1]] = min(self.label_ranges[img[1]], i)

    def select_second_index(self, index):
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

        return index2

    def read_data(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        landmarks = self.landmarks[index] if self.landmarks is not None else None
        bbox = self.bboxes[index] if self.bboxes is not None else None

        return img, landmarks, bbox, label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img1, landmarks1, bbox1, label1, img2, landmarks2, bbox2, label2) where
            - landmarks# is the face landmarks corresponding to img#
            - bbox# is the boundnig box corresponding to img#
            - label# is class_index of the target class corresponding to img#.
        """
        index2 = self.select_second_index(index)
        img1, landmarks1, bbox1, label1 = self.read_data(index)
        img2, landmarks2, bbox2, label2 = self.read_data(index2)
        if self.pair_transform is not None:
            img1, landmarks1, bbox1, img2, landmarks2, bbox2 = self.pair_transform(
                img1, landmarks1, bbox1, img2, landmarks2, bbox2)
        if self.transform1 is not None:
            img1, landmarks1, bbox1 = self.transform1(img1, landmarks1, bbox1)
        if self.transform2 is not None:
            img2, landmarks2, bbox2 = self.transform2(img2, landmarks2, bbox2)
        if self.target_transform is not None:
            label1, label2 = self.target_transform(label1), self.target_transform(label2)

        return img1, landmarks1, bbox1, label1, img2, landmarks2, bbox2, label2


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


def main(dataset, pair_transforms=None, pil_transforms1=None, pil_transforms2=None,
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])')):
    if 'FaceLandmarksDataset' in str(dataset):
        # Initialize dataset
        pil_transforms = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
        tensor_transforms = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
        img_transforms = landmark_transforms.ComposePyramids(pil_transforms + tensor_transforms)
        dataset = obj_factory(dataset, transform=img_transforms)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, drop_last=True, shuffle=False)

        for img, landmarks, bbox, target in dataset:
            if isinstance(img, list):
                for i in range(len(img)):
                    curr_img = img[i]
                    curr_landmarks = landmarks[i]

                    # Render landmark heatmaps on image
                    landmarks_rgb = landmark_transforms.heatmap2rgb(curr_landmarks.numpy())
                    img_red = np.zeros((landmarks_rgb.shape[0], landmarks_rgb.shape[1], 3))
                    img_red[:, :, 0] = 1.0
                    curr_img = unnormalize(curr_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    curr_img = curr_img.permute(1, 2, 0).numpy()
                    blend = curr_img * (1. - landmarks_rgb) + img_red * landmarks_rgb
                    blend_bgr = blend[:, :, ::-1].copy()
                    cv2.imshow('blend_bgr', blend_bgr)
                    cv2.waitKey(0)

            # img_landmarks = torch.cat((img, landmarks), dim=0)

            # img = np.array(img)[:, :, ::-1].copy()
            # cv2.imshow('img', img)
            # print('target = ' + str(target))
            # cv2.waitKey(0)
    elif 'FacePairLandmarksDataset' in str(dataset):
        # Initialize dataset
        pair_transforms = obj_factory(pair_transforms)
        pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
        pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
        tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
        tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
        img_pair_transforms = landmark_transforms.ComposePair(pair_transforms)
        img_transforms1 = landmark_transforms.Compose(pil_transforms1 + tensor_transforms1)
        img_transforms2 = landmark_transforms.Compose(pil_transforms2 + tensor_transforms2)
        dataset = obj_factory(dataset, pair_transform=img_pair_transforms,
                              transform1=img_transforms1, transform2=img_transforms2)

        for i, (img1, landmarks1, bbox1, target1, img2, landmarks2, bbox2, target2) in enumerate(dataset):
            # Render landmark heatmaps on image
            landmarks2_rgb = landmark_transforms.heatmap2rgb(landmarks2.numpy())
            img_red = np.zeros((landmarks2_rgb.shape[0], landmarks2_rgb.shape[1], 3))
            img_red[:, :, 0] = 1.0
            img2 = unnormalize(img2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            img2 = img2.permute(1, 2, 0).numpy()
            blend = img2 * (1. - landmarks2_rgb) + img_red * landmarks2_rgb
            blend_bgr = blend[:, :, ::-1].copy()
            cv2.imshow('blend_bgr', blend_bgr)
            cv2.waitKey(0)

            # landmarks2_rgb = landmark_transforms.heatmap2rgb(landmarks2.numpy())
            # landmarks2_rgb[:, :, 1:] = 0    # make red
            # img2 = img2.permute(1, 2, 0).numpy()
            # blend = img2*0.5 + landmarks2_rgb*0.5
            # blend_bgr = blend[:, :, ::-1].copy()
            # cv2.imshow('blend_bgr', blend_bgr)
            # landmarks2_bgr = landmarks2_rgb[:, :, ::-1].copy()
            # cv2.imshow('landmarks2_bgr', landmarks2_bgr)
            # cv2.waitKey(0)

            # img_landmarks1 = torch.cat((img1, landmarks2), dim=0)

            # img1 = unnormalize(img1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # img2 = unnormalize(img2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # img1 = transforms.ToPILImage()(img1)
            # img2 = transforms.ToPILImage()(img2)
            # img1 = np.array(img1)[:, :, ::-1].copy()
            # img2 = np.array(img2)[:, :, ::-1].copy()
            # cv2.imshow('img1', img1)
            # cv2.imshow('img2', img2)
            # print('target1 = %d, target2 = %d' % (target1, target2))
            # cv2.waitKey(0)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('face_landmarks_dataset')
    parser.add_argument('-d', '--dataset', help='dataset object')
    parser.add_argument('-pt', '--pair_transforms', default=None, nargs='+', help='pair PIL transforms')
    parser.add_argument('-pt1', '--pil_transforms1', default=None, nargs='+', help='first PIL transforms')
    parser.add_argument('-pt2', '--pil_transforms2', default=None, nargs='+', help='second PIL transforms')
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    args = parser.parse_args()

    main(args.dataset, args.pair_transforms, args.pil_transforms1, args.pil_transforms2,
         args.tensor_transforms1, args.tensor_transforms2)
