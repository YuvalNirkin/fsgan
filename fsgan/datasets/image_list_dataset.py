import os
import random
from itertools import groupby
import numpy as np
import cv2
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import fsgan.datasets.img_landmarks_transforms as img_landmarks_transforms


def find_classes(img_rel_paths):
    classes = [key for key, group in groupby(enumerate(img_rel_paths), lambda x: os.path.split(x[1])[0])]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def calc_weights_for_balanced_classes(targets):
    _, class_weights = np.unique(targets, return_counts=True)
    class_weights = np.sum(class_weights) / class_weights
    weights = np.array([class_weights[i] for i in targets])
    weights = weights / np.sum(weights)

    return weights


def read_bboxes(bboxes_file):
    if bboxes_file.lower().endswith('.npy'):
        return np.load(bboxes_file)
    data = np.loadtxt(bboxes_file, dtype=int, skiprows=1, usecols=range(1, 5), delimiter=',')
    return data


def opencv_loader(path):
    return cv2.imread(path)[:, :, ::-1]


def get_loader(backend='opencv'):
    if backend == 'opencv':
        return opencv_loader
    else:
        return default_loader


class ImageListDataset(VisionDataset):
    """An image list datset with corresponding bounding boxes and targets.

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path.
        bboxes_list (string): Bounding boxes list file path
        targets_list (string): Targets list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (string, optional): 'opencv', 'accimage', or 'pil'

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, bboxes_list=None, targets_list=None, transform=None, target_transform=None,
                 loader='opencv'):
        super(ImageListDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = get_loader(loader)
        self.transform = transform
        self.target_transform = target_transform

        # Parse image paths
        img_list_path = img_list if os.path.exists(img_list) else os.path.join(root, img_list)
        if not os.path.exists(img_list_path):
            raise (RuntimeError('Could not find image list file: ' + img_list))
        with open(img_list_path, 'r') as f:
            img_rel_paths = f.read().splitlines()
        self.imgs = [os.path.join(root, p) for p in img_rel_paths]

        # Load bounding boxes
        if bboxes_list is not None:
            bboxes_list_path = bboxes_list if os.path.exists(bboxes_list) else os.path.join(root, bboxes_list)
            self.bboxes = read_bboxes(bboxes_list_path) if bboxes_list is not None else bboxes_list
            assert len(self.imgs) == self.bboxes.shape[0]
        else:
            self.bboxes = None

        # Parse classes
        if targets_list is None:
            self.classes, class_to_idx = find_classes(img_rel_paths)
            self.targets = np.array([class_to_idx[os.path.split(p)[0]] for p in img_rel_paths], dtype='int64')
        else:
            targets_list_path = targets_list if os.path.exists(targets_list) else os.path.join(root, targets_list)
            if not os.path.exists(targets_list_path):
                raise (RuntimeError('Could not find targets list file: ' + targets_list))
            self.targets = np.loadtxt(targets_list_path, dtype=int)
            self.classes = np.unique(self.targets)
        assert len(self.imgs) == len(self.targets)

        self.weights = calc_weights_for_balanced_classes(self.targets)

    def get_data(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        target = self.targets[index]
        bbox = self.bboxes[index] if self.bboxes is not None else None

        return img, target, bbox

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img, target, bbox = self.get_data(index)
        if self.transform is not None:
            img = self.transform(img) if bbox is None else self.transform(img, bbox)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImagePairListDataset(ImageListDataset):
    """ An image dataset for loading pairs from a list.

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path
        bboxes_list (string): Bounding boxes list file path
        targets_list (string): Targets list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (string, optional): 'opencv', 'accimage', or 'pil'
        same_prob (float): The probability to return images of the same class
        return_targets (bool): If True, return the targets together with the images

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, bboxes_list=None, targets_list=None, transform=None, target_transform=None,
                 loader='opencv', same_prob=0.5, return_targets=False):
        super(ImagePairListDataset, self).__init__(root, img_list, bboxes_list, targets_list, transform,
                                                   target_transform, loader)
        self.same_prob = same_prob
        self.return_targets = return_targets

        # paths, labels = zip(*self.imgs)
        self.label_ranges = [len(self.imgs)] * (len(self.classes) + 1)
        for i, target in enumerate(self.targets):
            self.label_ranges[target] = min(self.label_ranges[target], i)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target1, target2) if return_targets is True else (image1, image2, same)
        """
        # Get pair indices
        label1 = self.targets[index]
        total_imgs1 = self.label_ranges[label1 + 1] - self.label_ranges[label1]
        if total_imgs1 > 1 and random.random() < self.same_prob:
            # Select another image from the same identity
            index2 = random.randint(self.label_ranges[label1], self.label_ranges[label1 + 1] - 2)
            index2 = index2 + 1 if index2 >= index else index2
        else:
            # Select another image from a different identity
            label2 = random.randint(0, len(self.classes) - 2)
            label2 = label2 + 1 if label2 >= label1 else label2
            index2 = random.randint(self.label_ranges[label2], self.label_ranges[label2 + 1] - 1)

        # Get pair data
        img1, target1, bbox1 = self.get_data(index)
        img2, target2, bbox2 = self.get_data(index2)

        # Apply transformations
        if self.transform is not None:
            if bbox1 is None or bbox2 is None:
                img1, img2 = tuple(self.transform([img1, img2]))
            else:
                img1, img2 = tuple(self.transform([img1, img2], [bbox1, bbox2]))
        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        if self.return_targets:
            return (img1, img2), target1, target2
        else:
            # return (img1, img2), np.array(target1 == target2, dtype='float32')
            return (img1, img2), target1 == target2


class ImageTripletListDataset(VisionDataset):
    """ An image dataset for loading triplets from a list.

    Args:
        root (string): Root directory path.
        img_list (string): Image list file path.
        bboxes_list (string): Bounding boxes list file path
        targets_list (string): Targets list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (string, optional): 'opencv', 'accimage', or 'pil'

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, bboxes_list=None, targets_list=None, transform=None, target_transform=None,
                 loader='opencv'):
        super(ImageTripletListDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = get_loader(loader)

        # Load image paths from list
        img_list_path = img_list if os.path.exists(img_list) else os.path.join(root, img_list)
        assert os.path.isfile(img_list_path), f'Could not find image list file: "{img_list}"'
        with open(img_list_path, 'r') as f:
            img_rel_path_triplets = f.read().splitlines()
        self.imgs = []
        for img_rel_path_triplet in img_rel_path_triplets:
            self.imgs.append([os.path.join(root, p) for p in img_rel_path_triplet.split()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target1, target2) if return_targets is True else (image1, image2, same)
        """
        img_triplet_paths = self.imgs[index]
        img_triplet = [self.loader(p) for p in img_triplet_paths]
        if self.transform is not None:
            img_triplet = self.transform(img_triplet)

        return tuple(img_triplet)

    def __len__(self):
        return len(self.imgs)


def main(dataset='fake_detection.datasets.image_list_dataset.ImageListDataset', np_transforms=None,
         tensor_transforms=('img_landmarks_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         workers=4, batch_size=4):
    import time
    import fsgan
    from fsgan.utils.obj_factory import obj_factory
    from fsgan.utils.img_utils import tensor2bgr

    np_transforms = obj_factory(np_transforms) if np_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = img_landmarks_transforms.Compose(np_transforms + tensor_transforms)
    dataset = obj_factory(dataset, transform=img_transforms)
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=workers, pin_memory=True, drop_last=True,
                                 shuffle=True)

    start = time.time()
    if isinstance(dataset, fsgan.datasets.image_list_dataset.ImageListDataset):
        for img, target in dataloader:
            print(img.shape)
            print(target)

            # For each batch
            for b in range(img.shape[0]):
                render_img = tensor2bgr(img[b]).copy()
                cv2.imshow('render_img', render_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
    elif isinstance(dataset, fsgan.datasets.image_list_dataset.ImagePairListDataset):
        for img1, img2, target in dataloader:
            print(img1.shape)
            print(img2.shape)
            print(target)

            # For each batch
            for b in range(target.shape[0]):
                left_img = tensor2bgr(img1[b]).copy()
                right_img = tensor2bgr(img2[b]).copy()
                render_img = np.concatenate((left_img, right_img), axis=1)
                cv2.imshow('render_img', render_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
    elif isinstance(dataset, fsgan.datasets.image_list_dataset.ImageTripletListDataset):
        for img1, img2, img3 in dataloader:
            print(img1.shape)
            print(img2.shape)
            print(img3.shape)
    end = time.time()
    print('elapsed time: %f[s]' % (end - start))

    return 0


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('image_list_dataset')
    parser.add_argument('dataset', metavar='OBJ', default='fake_detection.datasets.image_list_dataset.ImageListDataset',
                        help='dataset object')
    parser.add_argument('-nt', '--np_transforms', default=None, nargs='+', help='Numpy transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('img_landmarks_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    args = parser.parse_args()
    main(args.dataset, args.np_transforms, args.tensor_transforms, args.workers, args.batch_size)