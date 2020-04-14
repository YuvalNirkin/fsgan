import os
import random
from itertools import groupby
import numpy as np
import cv2
from PIL import Image
import torch
import torch.utils.data as data
from fsgan.datasets.image_list_dataset import ImageListDataset
import fsgan.datasets.img_landmarks_transforms as img_landmarks_transforms


def seg_label2img(seg, classes=3):
    out_seg = np.zeros(seg.shape + (classes,), dtype=seg.dtype)
    # out_seg = np.full(seg.shape + (classes,), 127.5, dtype='float32')
    for i in range(classes):
        out_seg[:, :, i][seg == i] = 255

    return out_seg


class ImageSegDataset(ImageListDataset):
    """An image list datset with corresponding bounding boxes where the images can be arranged in this way:

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
        loader (string, optional): 'opencv', 'accimage', or 'pil'

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, img_list, bboxes_list=None, transform=None, target_transform=None, loader='opencv',
                 seg_postfix='_mask.png', seg_classes=3, mask_root=None, classification=False):
        super(ImageSegDataset, self).__init__(root, img_list, bboxes_list, None, transform, target_transform, loader)
        self.seg_classes = seg_classes
        if not classification:
            self.classes = list(range(seg_classes))
        self.classification = classification
        if mask_root is None:
            self.segs = [os.path.splitext(p)[0] + seg_postfix for p in self.imgs]
        else:
            self.segs = [os.path.join(mask_root, os.path.splitext(os.path.relpath(p, root))[0] + seg_postfix)
                         for p in self.imgs]

        # Validate that all mask files exist
        for seg_path in self.segs:
            assert os.path.isfile(seg_path), 'Could not find mask file: "%s"' % seg_path

    def get_data(self, index):
        img, target, bbox = super(ImageSegDataset, self).get_data(index)
        seg = np.array(Image.open(self.segs[index]))

        # Convert segmentation format to a channel for each class
        seg = seg_label2img(seg, self.seg_classes)

        return img, seg, bbox, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, segmentation)
        """
        img, seg, bbox, target = self.get_data(index)
        if self.transform is not None:
            seg_scale = seg.shape[0] / img.shape[0]
            bboxes = [bbox, bbox * seg_scale]
            img, seg = tuple(self.transform([img, seg], bboxes) if bbox is None else self.transform([img, seg], bboxes))
            # seg[(seg[:, :, 0] < 0) & (seg[:, :, 1] < 0) & (seg[:, :, 2] < 0)] = 1.0
            # seg = torch.clamp(seg, min=0.0)
        # if self.transform is not None:
        #     img = self.transform(img) if bbox is None else self.transform(img, bbox)
        # if self.target_transform is not None:
        #     seg_scale = seg.shape[0] / img.shape[0]
        #     seg = self.target_transform(seg, bbox * seg_scale)

        # Postprocess segmentation
        seg[0, :, :][(seg[0, :, :] <= 0) & (seg[1, :, :] <= 0) & (seg[0, :, :] <= 0)] = 1.
        seg = torch.clamp(seg, min=0.0)

        if self.classification:
            return img, seg, target
        else:
            return img, seg

    def __len__(self):
        return len(self.imgs)


def main(dataset='fsgan.datasets.image_seg_dataset.ImageSegDataset',
         np_transforms1=None, np_transforms2=None,
         tensor_transforms1=('img_landmarks_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('img_landmarks_transforms.ToTensor()',),
         workers=4, batch_size=4):
    import time
    from fsgan.utils.obj_factory import obj_factory
    from fsgan.utils.seg_utils import blend_seg_pred, blend_seg_label
    from fsgan.utils.img_utils import tensor2bgr

    np_transforms1 = obj_factory(np_transforms1) if np_transforms1 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    img_transforms1 = img_landmarks_transforms.Compose(np_transforms1 + tensor_transforms1)
    np_transforms2 = obj_factory(np_transforms2) if np_transforms2 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_transforms2 = img_landmarks_transforms.Compose(np_transforms2 + tensor_transforms2)
    dataset = obj_factory(dataset, transform=img_transforms1, target_transform=img_transforms2)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, drop_last=True,
                                 shuffle=True)

    start = time.time()
    for img, seg in dataloader:
        # For each batch
        for b in range(img.shape[0]):
            blend_tensor = blend_seg_pred(img, seg)
            render_img = tensor2bgr(blend_tensor[b])
            # render_img = tensor2bgr(img[b])
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    end = time.time()
    print('elapsed time: %f[s]' % (end - start))


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('image_seg_dataset')
    parser.add_argument('dataset', metavar='OBJ', default='fsgan.datasets.image_seg_dataset.ImageSegDataset',
                        help='dataset object')
    parser.add_argument('-nt1', '--np_transforms1', nargs='+', help='Numpy transforms')
    parser.add_argument('-nt2', '--np_transforms2', nargs='+', help='Numpy transforms')
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='tensor transforms',
                        default=('img_landmarks_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='tensor transforms',
                        default=('img_landmarks_transforms.ToTensor()',))
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    main(**vars(parser.parse_args()))
