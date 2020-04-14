import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils as tutils
import fsgan.data.landmark_transforms as landmark_transforms
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils import utils
from fsgan.utils.seg_utils import blend_seg_pred
from fsgan.utils.img_utils import tensor2bgr
from fsgan.utils.bbox_utils import scale_bbox


def main(img_list, root=None, out_dir=None, seg_model_path=None, workers=4, iterations=None, batch_size=8, gpus=None,
         test_dataset='image_list_dataset.ImageListDataset', pil_transforms=None,
         tensor_transforms=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         seg_threshold=0.15):
    root = os.path.split(img_list)[0] if root is None else root
    out_dir = os.path.join(root, 'masks') if out_dir is None else out_dir
    torch.set_grad_enabled(False)

    # Validation
    if not os.path.isfile(img_list):
        raise RuntimeError('Image list file was not found: \'' + img_list + '\'')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)

    # Initialize datasets
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    img_transforms = landmark_transforms.Compose(pil_transforms + tensor_transforms)
    test_dataset = obj_factory(test_dataset, root, transform=img_transforms)

    # Initialize data loaders
    test_loader = tutils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=None,
                                         num_workers=workers, pin_memory=True, drop_last=False, shuffle=False)

    # Load face segmentation model
    if seg_model_path is not None:
        print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
        if seg_model_path.endswith('.pth'):
            checkpoint = torch.load(seg_model_path)
            Gs = obj_factory(checkpoint['arch']).to(device)
            Gs.load_state_dict(checkpoint['state_dict'])
        else:
            Gs = torch.jit.load(seg_model_path, map_location=device)
        if Gs is None:
            raise RuntimeError('Failed to load face segmentation model!')
            Gs.eval()
    else:
        Gs = None

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        Gs = nn.DataParallel(Gs, gpus)

    # For each image in the list
    img_rel_paths = np.loadtxt(img_list, dtype=str)
    img_paths = [os.path.join(root, p) for p in img_rel_paths]
    img_count = 0
    seg_rel_paths = []
    for i, (input, landmarks, bbox, target) in enumerate(tqdm(test_loader, unit='batches')):

        input = input.to(device)

        # compute output and loss
        pred = Gs(input)

        # Get segmentation prediction as numpy array
        seg = pred.max(1)[1].cpu().numpy()

        # seg_blend = blend_seg_pred(input, pred)

        # For each image in the batch
        for j in range(seg.shape[0]):
            full_img_bgr = cv2.imread(img_paths[img_count])

            curr_bbox = bbox[j].cpu().numpy()
            full_seg = np.zeros((full_img_bgr.shape[0], full_img_bgr.shape[1]), dtype=full_img_bgr.dtype)
            full_seg = crop2img(full_seg, seg[j], curr_bbox)

            # Check against threshold
            curr_bbox = np.round(curr_bbox).astype(int)
            tight_seg = full_seg[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            score = np.count_nonzero(tight_seg == 1) / (curr_bbox[2] * curr_bbox[3])
            if score > seg_threshold:
                # Write output to file
                curr_file_name = os.path.splitext(os.path.basename(img_rel_paths[img_count]))[0] + '.png'
                curr_rel_dir = os.path.join(out_dir, os.path.split(os.path.split(img_rel_paths[img_count])[0])[1])
                curr_out_path = os.path.join(curr_rel_dir, curr_file_name)
                if not os.path.isdir(curr_rel_dir):
                    os.mkdir(curr_rel_dir)
                cv2.imwrite(curr_out_path, full_seg)

                ### Debug ###
                # cropped_seg_blend_bgr = tensor2bgr(seg_blend[j])
                # full_seg_blend_bgr = crop2img(full_img_bgr, cropped_seg_blend_bgr, curr_bbox, interp=cv2.INTER_CUBIC)
                # cv2.imshow('seg_blend', full_seg_blend_bgr)
                # cv2.waitKey(0)
                #############

                curr_seg_path = os.path.relpath(curr_out_path, root).replace(os.sep, '/')
                seg_rel_paths.append(curr_seg_path)

            img_count += 1

    # Write seg list file
    np.savetxt(os.path.join(root, 'seg_list.txt'), seg_rel_paths, fmt='%s')


def crop2img(img, crop, bbox, interp=cv2.INTER_NEAREST):
    scaled_bbox = scale_bbox(bbox)
    scaled_crop = cv2.resize(crop, (scaled_bbox[3], scaled_bbox[2]), interpolation=interp)
    left = -scaled_bbox[0] if scaled_bbox[0] < 0 else 0
    top = -scaled_bbox[1] if scaled_bbox[1] < 0 else 0
    right = scaled_bbox[0] + scaled_bbox[2] - img.shape[1] if (scaled_bbox[0] + scaled_bbox[2] - img.shape[1]) > 0 else 0
    bottom = scaled_bbox[1] + scaled_bbox[3] - img.shape[0] if (scaled_bbox[1] + scaled_bbox[3] - img.shape[0]) > 0 else 0
    crop_bbox = np.array([left, top, scaled_bbox[2] - left - right, scaled_bbox[3] - top - bottom])
    scaled_bbox += np.array([left, top, -left - right, -top - bottom])

    out_img = img.copy()
    out_img[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]] = \
        scaled_crop[crop_bbox[1]:crop_bbox[1] + crop_bbox[3], crop_bbox[0]:crop_bbox[0] + crop_bbox[2]]

    return out_img


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('create_masks')
    parser.add_argument('img_list',
                        help='path to image list file')
    parser.add_argument('-r', '--root', metavar='DIR',
                        help='paths to the root directory')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output video path')
    parser.add_argument('-sm', '--seg_model', metavar='PATH',
                        help='path to face segmentation model')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-td', '--test_dataset', default='image_list_dataset.ImageListDataset', type=str,
                        help='test dataset object')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-st', '--seg_threshold', default=0.15, type=float, metavar='F',
                        help='segmentation threshold')
    args = parser.parse_args()
    main(args.img_list, args.root, args.output, args.seg_model, workers=args.workers,
         batch_size=args.batch_size, gpus=args.gpus, test_dataset=args.test_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms, seg_threshold=args.seg_threshold)