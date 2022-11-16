import os
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils as tutils
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.tensorboard_logger import TensorBoardLogger
from fsgan.utils import utils, img_utils
from fsgan.utils.seg_utils import blend_seg_pred, blend_seg_label
from fsgan.utils.iou_metric import IOUMetric
from fsgan.datasets import img_landmarks_transforms


class IOUBenchmark(IOUMetric):
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super(IOUBenchmark, self).__init__(num_classes, normalized, ignore_index)

    def to(self, device):
        return self

    def __call__(self, pred, target):
        self.add(pred, target)
        _, miou = self.value()

        return {'iou': miou}


def main(
    # General arguments
    exp_dir, resume_dir=None, start_epoch=None, epochs=(90,), iterations=None, resolutions=(128, 256),
    learning_rate=(1e-1,), gpus=None, workers=4, batch_size=(64,), seed=None, log_freq=20,

    # Data arguments
    train_dataset='fsgan.image_seg_dataset.ImageSegDataset', val_dataset=None, numpy_transforms=None,
    tensor_transforms=('img_landmarks_transforms.ToTensor()',
                       'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),

    # Training arguments
    optimizer='optim.SGD(momentum=0.9,weight_decay=1e-4)', scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
    criterion='nn.CrossEntropyLoss', model='fsgan.models.simple_unet.UNet(n_classes=3,feature_scale=1)',
    pretrained=False, benchmark='fsgan.train_segmentation.IOUBenchmark(3)'
):
    def proces_epoch(dataset_loader, train=True):
        stage = 'TRAINING' if train else 'VALIDATION'
        total_iter = len(dataset_loader) * dataset_loader.batch_size * epoch
        pbar = tqdm(dataset_loader, unit='batches')

        # Set networks training mode
        model.train(train)

        # Reset logger
        logger.reset(prefix='{} {}X{}: Epoch: {} / {}; LR: {:.0e}; '.format(
            stage, res, res, epoch + 1, res_epochs, scheduler.get_lr()[0]))

        # For each batch in the training data
        for i, (input, target) in enumerate(pbar):
            # Prepare input
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                target = target.argmax(dim=1)

            # Execute model
            pred = model(input)

            # Calculate loss
            loss_total = criterion(pred, target)

            # Run benchmark
            benchmark_res = benchmark(pred, target) if benchmark is not None else {}

            if train:
                # Update generator weights
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            logger.update('losses', total=loss_total)
            logger.update('bench', **benchmark_res)
            total_iter += dataset_loader.batch_size

            # Batch logs
            pbar.set_description(str(logger))
            if train and i % log_freq == 0:
                logger.log_scalars_val('%dx%d/batch' % (res, res), total_iter)

        # Epoch logs
        logger.log_scalars_avg('%dx%d/epoch/%s' % (res, res, 'train' if train else 'val'), epoch)
        if not train:
            # Log images
            seg_pred = blend_seg_pred(input, pred)
            seg_gt = blend_seg_label(input, target)
            grid = img_utils.make_grid(input, seg_pred, seg_gt)
            logger.log_image('%dx%d/vis' % (res, res), grid, epoch)

        return logger.log_dict['losses']['total'].avg

    #################
    # Main pipeline #
    #################

    # Validation
    resolutions = resolutions if isinstance(resolutions, (list, tuple)) else [resolutions]
    learning_rate = learning_rate if isinstance(learning_rate, (list, tuple)) else [learning_rate]
    epochs = epochs if isinstance(epochs, (list, tuple)) else [epochs]
    batch_size = batch_size if isinstance(batch_size, (list, tuple)) else [batch_size]
    iterations = iterations if iterations is None or isinstance(iterations, (list, tuple)) else [iterations]

    learning_rate = learning_rate * len(resolutions) if len(learning_rate) == 1 else learning_rate
    epochs = epochs * len(resolutions) if len(epochs) == 1 else epochs
    batch_size = batch_size * len(resolutions) if len(batch_size) == 1 else batch_size
    if iterations is not None:
        iterations = iterations * len(resolutions) if len(iterations) == 1 else iterations
        iterations = utils.str2int(iterations)

    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')
    assert len(learning_rate) == len(resolutions)
    assert len(epochs) == len(resolutions)
    assert len(batch_size) == len(resolutions)
    assert iterations is None or len(iterations) == len(resolutions)

    # Seed
    utils.set_seed(seed)

    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)

    # Initialize loggers
    logger = TensorBoardLogger(log_dir=exp_dir)

    # Initialize datasets
    numpy_transforms = obj_factory(numpy_transforms) if numpy_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = img_landmarks_transforms.Compose(numpy_transforms + tensor_transforms)

    train_dataset = obj_factory(train_dataset, transform=img_transforms)
    if val_dataset is not None:
        val_dataset = obj_factory(val_dataset, transform=img_transforms)

    # Create networks
    arch = utils.get_arch(model, num_classes=len(train_dataset.classes))
    model = obj_factory(model, num_classes=len(train_dataset.classes)).to(device)

    # Resume from a checkpoint or initialize the networks weights randomly
    checkpoint_dir = exp_dir if resume_dir is None else resume_dir
    model_path = os.path.join(checkpoint_dir, 'model_latest.pth')
    best_loss = 1e6
    curr_res = resolutions[0]
    optimizer_state = None
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        # model
        checkpoint = torch.load(model_path)
        if 'resolution' in checkpoint:
            curr_res = checkpoint['resolution']
            start_epoch = checkpoint['epoch'] if start_epoch is None else start_epoch
        # else:
        #     curr_res = resolutions[1] if len(resolutions) > 1 else resolutions[0]
        best_loss_key = 'best_loss_%d' % curr_res
        best_loss = checkpoint[best_loss_key] if best_loss_key in checkpoint else best_loss
        model.apply(utils.init_weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer_state = checkpoint['optimizer']
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        if not pretrained:
            print("=> randomly initializing networks...")
            model.apply(utils.init_weights)

    # Lossess
    criterion = obj_factory(criterion).to(device)

    # Benchmark
    benchmark = obj_factory(benchmark).to(device)

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # For each resolution
    start_res_ind = int(np.log2(curr_res)) - int(np.log2(resolutions[0]))
    start_epoch = 0 if start_epoch is None else start_epoch
    for ri in range(start_res_ind, len(resolutions)):
        res = resolutions[ri]
        res_lr = learning_rate[ri]
        res_epochs = epochs[ri]
        res_iterations = iterations[ri] if iterations is not None else None
        res_batch_size = batch_size[ri]

        # Optimizer and scheduler
        optimizer = obj_factory(optimizer, model.parameters(), lr=res_lr)
        scheduler = obj_factory(scheduler, optimizer)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        # Initialize data loaders
        if res_iterations is None:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, len(train_dataset))
        else:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, res_iterations)
        train_loader = tutils.data.DataLoader(train_dataset, batch_size=res_batch_size, sampler=train_sampler,
                                              num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)
        if val_dataset is not None:
            if res_iterations is None:
                val_sampler = tutils.data.sampler.WeightedRandomSampler(val_dataset.weights, len(val_dataset))
            else:
                val_iterations = (res_iterations * len(val_dataset)) // len(train_dataset)
                val_sampler = tutils.data.sampler.WeightedRandomSampler(val_dataset.weights, val_iterations)
            val_loader = tutils.data.DataLoader(val_dataset, batch_size=res_batch_size, sampler=val_sampler,
                                                num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)
        else:
            val_loader = None

        # For each epoch
        for epoch in range(start_epoch, res_epochs):
            total_loss = proces_epoch(train_loader, train=True)
            if val_loader is not None:
                with torch.no_grad():
                    total_loss = proces_epoch(val_loader, train=False)
            if hasattr(benchmark, 'reset'):
                benchmark.reset()

            # Schedulers step (in PyTorch 1.1.0+ it must follow after the epoch training and validation steps)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss)
            else:
                scheduler.step()

            # Save models checkpoints
            is_best = total_loss < best_loss
            best_loss = min(best_loss, total_loss)
            utils.save_checkpoint(exp_dir, 'model', {
                'resolution': res,
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if gpus and len(gpus) > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss_%d' % res: best_loss,
                'arch': arch,
            }, is_best)

        # Reset start epoch to 0 because it's should only effect the first training resolution
        start_epoch = 0
        best_loss = 1e6


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('train_segmentation_ces')
    general = parser.add_argument_group('general')
    general.add_argument('exp_dir', metavar='DIR',
                         help='path to experiment directory')
    general.add_argument('-rd', '--resume_dir', metavar='DIR',
                         help='path to resume directory (default: None)')
    general.add_argument('-se', '--start-epoch', metavar='N',
                         help='manual epoch number (useful on restarts)')
    general.add_argument('-e', '--epochs', default=90, type=int, nargs='+', metavar='N',
                         help='number of total epochs to run')
    general.add_argument('-i', '--iterations', nargs='+', metavar='N',
                         help='number of iterations per resolution to run')
    general.add_argument('-r', '--resolutions', default=(128, 256), type=int, nargs='+', metavar='N',
                         help='the training resolutions list (must be power of 2)')
    general.add_argument('-lr', '--learning-rate', default=(1e-1,), type=float, nargs='+', metavar='F',
                         help='initial learning rate per resolution')
    general.add_argument('--gpus', nargs='+', type=int, metavar='N',
                         help='list of gpu ids to use (default: all)')
    general.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                         help='number of data loading workers (default: 4)')
    general.add_argument('-b', '--batch-size', default=(64,), type=int, nargs='+', metavar='N',
                         help='mini-batch size (default: 64)')
    general.add_argument('--seed', type=int, metavar='N',
                         help='random seed')
    general.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                         help='number of steps between each loss plot')

    data = parser.add_argument_group('data')
    data.add_argument('-td', '--train_dataset', default='fsgan.image_seg_dataset.ImageSegDataset',
                      help='train dataset object')
    data.add_argument('-vd', '--val_dataset',
                      help='val dataset object')
    data.add_argument('-nt', '--numpy_transforms', nargs='+',
                      help='Numpy transforms')
    data.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                      default=('img_landmarks_transforms.ToTensor()',
                               'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))

    training = parser.add_argument_group('training')
    training.add_argument('-o', '--optimizer', default='optim.SGD(momentum=0.9,weight_decay=1e-4)',
                          help='network\'s optimizer object')
    training.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
                          help='scheduler object')
    training.add_argument('-c', '--criterion', default='nn.CrossEntropyLoss',
                          help='criterion object')
    training.add_argument('-m', '--model', default='fsgan.models.simple_unet.UNet(n_classes=3,feature_scale=1)',
                          help='model object')
    training.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                          help='use pre-trained model')
    training.add_argument('-be', '--benchmark', default='fsgan.train_segmentation.IOUBenchmark(3)',
                          help='benchmark object')

    main(**vars(parser.parse_args()))
