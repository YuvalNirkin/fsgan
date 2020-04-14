import os
import numpy as np
from itertools import groupby
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader


def find_classes(img_rel_paths):
    classes = [key for key, group in groupby(enumerate(img_rel_paths), lambda x: os.path.split(x[1])[0])]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, img_rel_paths, class_to_idx):
    images = []
    for img_rel_path in img_rel_paths:
        target = os.path.split(img_rel_path)[0]
        img_abs_path = os.path.join(root, img_rel_path)
        images.append((img_abs_path, class_to_idx[target]))

    return images


def calc_weights_for_balanced_classes(samples, num_classes):
    paths, labels = zip(*samples)
    _, class_weights = np.unique(labels, return_counts=True)
    class_weights = np.sum(class_weights) / class_weights
    weights = np.array([class_weights[i] for i in labels])
    weights = weights / np.sum(weights)

    return weights


class ImageListDataset(DatasetFolder):
    """A generic image data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

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
        weights (numpy.ndarray) Sample weights for balanced classes.
    """
    def __init__(self, root, img_list, transform=None, target_transform=None, loader=default_loader):
        img_list_path = img_list if os.path.exists(img_list) else os.path.join(root, img_list)
        if not os.path.exists(img_list_path):
            raise (RuntimeError('Could not find image list file: ' + img_list))
        with open(img_list_path, 'r') as f:
            img_rel_paths = f.read().splitlines()
        img_rel_paths = sorted(img_rel_paths)   # Make sure the paths are in the right order

        classes, class_to_idx = find_classes(img_rel_paths)
        samples = make_dataset(root, img_rel_paths, class_to_idx)
        if len(samples) == 0:
            raise(RuntimeError('Found 0 files in image list: ' + img_list))

        self.root = root
        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = self.samples = samples
        self.targets = [s[1] for s in samples]
        self.weights = calc_weights_for_balanced_classes(self.samples, len(self.classes))

        self.transform = transform
        self.target_transform = target_transform


def main(root_path, img_list, pil_transforms):
    import cv2
    import torchvision.transforms as transforms
    from fsgan.utils.obj_factory import obj_factory
    import fsgan.utils as utils
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    pil_transforms = transforms.Compose(pil_transforms)
    dataset = ImageListDataset(root_path, img_list, transform=pil_transforms)
    for img, label in dataset:
        img = np.array(img)[:, :, ::-1].copy()
        cv2.imshow('img', img)
        print('label = ' + str(label))
        cv2.waitKey(0)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('image_list_dataset')
    parser.add_argument('root_path', help='paths to dataset root directory')
    parser.add_argument('img_list', help='image list file path')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    args = parser.parse_args()

    main(args.root_path, args.img_list, args.pil_transforms)