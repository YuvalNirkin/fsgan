""" Test script for the face segmentation model. """

import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils as tutils
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils import utils
from PIL import Image
import scipy.misc as m


def main(exp_dir, test_dir, workers=4, iterations=None, batch_size=64, gpus=None,
         test_dataset='image_list_dataset.ImageListDataset', pil_transforms=None,
         tensor_transforms=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         arch='resnet18'):
    # Validation
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')

    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)

    # Initialize datasets
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    img_transforms = transforms.Compose(pil_transforms + tensor_transforms)
    test_dataset = obj_factory(test_dataset, test_dir, transform=img_transforms)

    # Initialize data loaders
    if iterations is None:
        test_sampler = tutils.data.sampler.WeightedRandomSampler(test_dataset.weights, len(test_dataset))
    else:
        test_sampler = tutils.data.sampler.WeightedRandomSampler(test_dataset.weights, iterations,
                                                                 replacement=(iterations > len(test_dataset)))
    test_loader = tutils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                         num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)

    # Create model
    model = obj_factory(arch).to(device)

    # Load model weights
    model_path = os.path.join(exp_dir, 'model_best.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(exp_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(exp_dir))

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    # evaluate on validation set
    output_dir = os.path.join(exp_dir, 'test')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    validate(test_loader, model, device, output_dir)


def validate(val_loader, model, device, output_dir):
    batch_time = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, unit='batches')
        img_count = 0
        for i, (input, target) in enumerate(pbar):

            input = input.to(device)

            # compute output and loss
            output = model(input)

            # getting images and predictions as numpy arrays
            # images = input.cpu().numpy()
            pred = output.max(1)[1].cpu().numpy()

            for j in range(pred.shape[0]):
                # recovering mask of predictions color
                rgb_mask = decode_segmap(pred[j], model.n_classes)

                # using blending to concatenate mask and image
                img_with_mask = tensor2rgb(input[j]) * 0.5 + rgb_mask * 0.5
                img_with_mask = Image.fromarray(np.uint8(img_with_mask))
                m.imsave(os.path.join(output_dir, 'frame_%05d.jpg' % img_count), img_with_mask)
                img_count += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(
                'VALIDATION: '
                'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '.format( batch_time=batch_time))


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
    output_img = unnormalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def get_labels():
    """Load the mapping that associates pascal classes with label colors

        Array values could be changed according to the task and classes

    Returns:
        np.ndarray with dimensions (N, 3)
    """
    return np.asarray(
        [
            [0, 0, 255],  # background
            [0, 255, 0],  # face
            [255, 0, 0],  # hair

        ]
    )


def decode_segmap(label_mask, n_classes, label_colours=get_labels()):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        n_classes (int): number of classes in the dataset

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(1, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r  # / 255.0
    rgb[:, :, 1] = g  # / 255.0
    rgb[:, :, 2] = b  # / 255.0

    return rgb


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Segmentation Testing')
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('-t', '--test', type=str, metavar='DIR',
                        help='paths to test dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-td', '--test_dataset', default='image_list_dataset.ImageListDataset', type=str,
                        help='test dataset object')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-a', '--arch', default='generators.GlobalGenerator',
                        help='network architecture object')
    args = parser.parse_args()
    main(args.exp_dir, args.test, workers=args.workers, iterations=args.iterations, batch_size=args.batch_size,
         gpus=args.gpus, test_dataset=args.test_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms, arch=args.arch)


