""" Generic classifier test script. """

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
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils import utils
import pandas as pd


class ConfusionMatrix(object):
    """Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not
    """

    def __init__(self, k, normalized=False):
        super(ConfusionMatrix, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


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
    if arch in models.__dict__:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](num_classes=len(test_dataset.classes))
        model.to(device)
    else:
        model = obj_factory(arch, num_classes=len(test_dataset.classes)).to(device)

    # Load model weights
    model_path = os.path.join(exp_dir, 'model_latest.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(exp_dir))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(exp_dir))

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features, gpus)
        else:
            model = nn.DataParallel(model, gpus)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    # evaluate on validation set
    val_loss, prec1, conf = validate(test_loader, model, criterion, device)
    print('prec@1 = %f' % prec1)

    # Print confusion matrix
    conf_df = pd.DataFrame(conf, columns=test_dataset.classes, index=test_dataset.classes)
    conf_df.to_csv(os.path.join(exp_dir, 'conf.csv'))
    print(conf)


def validate(test_loader, model, criterion, device):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    conf = ConfusionMatrix(len(test_loader.dataset.classes))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(test_loader, unit='batches')
        for i, (input, target) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            if len(output.shape) > 2:
                output = output.view(output.size(0), -1)
            loss = criterion(output, target)

            conf.add(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(
                'VALIDATION: '
                'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '
                'Loss {loss.val:.3f} ({loss.avg:.3f}); '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}); '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, conf.value()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Classifier Testing')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                    choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    args = parser.parse_args()
    main(args.exp_dir, args.test, workers=args.workers, iterations=args.iterations, batch_size=args.batch_size,
         gpus=args.gpus, test_dataset=args.test_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms, arch=args.arch)


