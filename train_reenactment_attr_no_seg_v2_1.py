import os
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils as tutils
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.tensorboard_logger import TensorBoardLogger
from fsgan.utils import utils, img_utils, landmarks_utils
from fsgan.datasets import img_lms_pose_transforms
from fsgan.models.hrnet import hrnet_wlfw


def main(
    # General arguments
    exp_dir, resume_dir=None, start_epoch=None, epochs=(90,), iterations=None, resolutions=(128, 256),
    lr_gen=(1e-4,), lr_dis=(1e-4,), gpus=None, workers=4, batch_size=(64,), seed=None, log_freq=20,

    # Data arguments
    train_dataset='opencv_video_seq_dataset.VideoSeqDataset', val_dataset=None, numpy_transforms=None,
    tensor_transforms=('img_lms_pose_transforms.ToTensor()', 'img_lms_pose_transforms.Normalize()'),

    # Training arguments
    optimizer='optim.SGD(momentum=0.9,weight_decay=1e-4)', scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
    pretrained=False, criterion_pixelwise='nn.L1Loss', criterion_id='vgg_loss.VGGLoss',
    criterion_attr='vgg_loss.VGGLoss', criterion_gan='gan_loss.GANLoss(use_lsgan=True)',
    generator='res_unet.MultiScaleResUNet(in_nc=101,out_nc=3)',
    discriminator='discriminators_pix2pix.MultiscaleDiscriminator',
    rec_weight=1.0, gan_weight=0.001
):
    def proces_epoch(dataset_loader, train=True):
        stage = 'TRAINING' if train else 'VALIDATION'
        total_iter = len(dataset_loader) * dataset_loader.batch_size * epoch
        pbar = tqdm(dataset_loader, unit='batches')

        # Set networks training mode
        G.train(train)
        D.train(train)

        # Reset logger
        logger.reset(prefix='{} {}X{}: Epoch: {} / {}; LR: {:.0e}; '.format(
            stage, res, res, epoch + 1, res_epochs,  optimizer_G.param_groups[0]['lr']))

        # For each batch in the training data
        for i, (img, landmarks, target) in enumerate(pbar):
            # Prepare input
            with torch.no_grad():
                # For each view images and landmarks
                landmarks[1] = landmarks[1].to(device)
                for j in range(len(img)):
                    # landmarks[j] = landmarks[j].to(device)

                    # For each pyramid image: push to device
                    for p in range(len(img[j])):
                        img[j][p] = img[j][p].to(device)

                # Remove unnecessary pyramids
                for j in range(len(img)):
                    img[j] = img[j][-ri - 1:]

                # Concatenate pyramid images with context to derive the final input
                input = []
                for p in range(len(img[0])):
                    context = res_landmarks_decoders[p](landmarks[1])
                    input.append(torch.cat((img[0][p], context), dim=1))

            # Reenactment
            img_pred = G(input)

            # Fake Detection and Loss
            img_pred_pyd = img_utils.create_pyramid(img_pred, len(img[0]))
            pred_fake_pool = D([x.detach() for x in img_pred_pyd])
            loss_D_fake = criterion_gan(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = D(img[1])
            loss_D_real = criterion_gan(pred_real, True)

            loss_D_total = (loss_D_fake + loss_D_real) * 0.5

            # GAN loss (Fake Passability Loss)
            pred_fake = D(img_pred_pyd)
            loss_G_GAN = criterion_gan(pred_fake, True)

            # Reconstruction and segmentation loss
            loss_pixelwise = criterion_pixelwise(img_pred, img[1][0])
            loss_id = criterion_id(img_pred, img[1][0])
            loss_attr = criterion_attr(img_pred, img[1][0])
            loss_rec = 0.1 * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr

            loss_G_total = rec_weight * loss_rec + gan_weight * loss_G_GAN

            if train:
                # Update generator weights
                optimizer_G.zero_grad()
                loss_G_total.backward()
                optimizer_G.step()

                # Update discriminator weights
                optimizer_D.zero_grad()
                loss_D_total.backward()
                optimizer_D.step()

            logger.update('losses', pixelwise=loss_pixelwise, id=loss_id, attr=loss_attr, rec=loss_rec,
                          g_gan=loss_G_GAN, d_gan=loss_D_total)
            total_iter += dataset_loader.batch_size

            # Batch logs
            pbar.set_description(str(logger))
            if train and i % log_freq == 0:
                logger.log_scalars_val('%dx%d/batch' % (res, res), total_iter)

        # Epoch logs
        logger.log_scalars_avg('%dx%d/epoch/%s' % (res, res, 'train' if train else 'val'), epoch)
        if not train:
            # Log images
            grid = img_utils.make_grid(img[0][0], img_pred, img[1][0])
            logger.log_image('%dx%d/vis' % (res, res), grid, epoch)

        return logger.log_dict['losses']['rec'].avg

    #################
    # Main pipeline #
    #################

    # Validation
    resolutions = resolutions if isinstance(resolutions, (list, tuple)) else [resolutions]
    lr_gen = lr_gen if isinstance(lr_gen, (list, tuple)) else [lr_gen]
    lr_dis = lr_dis if isinstance(lr_dis, (list, tuple)) else [lr_dis]
    epochs = epochs if isinstance(epochs, (list, tuple)) else [epochs]
    batch_size = batch_size if isinstance(batch_size, (list, tuple)) else [batch_size]
    iterations = iterations if iterations is None or isinstance(iterations, (list, tuple)) else [iterations]

    lr_gen = lr_gen * len(resolutions) if len(lr_gen) == 1 else lr_gen
    lr_dis = lr_dis * len(resolutions) if len(lr_dis) == 1 else lr_dis
    epochs = epochs * len(resolutions) if len(epochs) == 1 else epochs
    batch_size = batch_size * len(resolutions) if len(batch_size) == 1 else batch_size
    if iterations is not None:
        iterations = iterations * len(resolutions) if len(iterations) == 1 else iterations
        iterations = utils.str2int(iterations)

    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')
    assert len(lr_gen) == len(resolutions)
    assert len(lr_dis) == len(resolutions)
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
    img_transforms = img_lms_pose_transforms.Compose(numpy_transforms + tensor_transforms)

    train_dataset = obj_factory(train_dataset, transform=img_transforms)
    if val_dataset is not None:
        val_dataset = obj_factory(val_dataset, transform=img_transforms)

    # Create networks
    G_arch = utils.get_arch(generator)
    D_arch = utils.get_arch(discriminator)
    G = obj_factory(generator).to(device)
    D = obj_factory(discriminator).to(device)

    # Resume from a checkpoint or initialize the networks weights randomly
    checkpoint_dir = exp_dir if resume_dir is None else resume_dir
    G_path = os.path.join(checkpoint_dir, 'G_latest.pth')
    D_path = os.path.join(checkpoint_dir, 'D_latest.pth')
    best_loss = 1e6
    curr_res = resolutions[0]
    optimizer_G_state, optimizer_D_state = None, None
    if os.path.isfile(G_path) and os.path.isfile(D_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        # G
        checkpoint = torch.load(G_path)
        if 'resolution' in checkpoint:
            curr_res = checkpoint['resolution']
            start_epoch = checkpoint['epoch'] if start_epoch is None else start_epoch
        # else:
        #     curr_res = resolutions[1] if len(resolutions) > 1 else resolutions[0]
        best_loss_key = 'best_loss_%d' % curr_res
        best_loss = checkpoint[best_loss_key] if best_loss_key in checkpoint else best_loss
        G.apply(utils.init_weights)
        G.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer_G_state = checkpoint['optimizer']

        # D
        D.apply(utils.init_weights)
        if os.path.isfile(D_path):
            checkpoint = torch.load(D_path)
            D.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer_D_state = checkpoint['optimizer']
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        if not pretrained:
            print("=> randomly initializing networks...")
            G.apply(utils.init_weights)
            D.apply(utils.init_weights)

    # Initialize landmarks decoders
    landmarks_decoders = []
    for res in resolutions:
        landmarks_decoders.insert(0, landmarks_utils.LandmarksHeatMapDecoder(res).to(device))

    # Lossess
    criterion_pixelwise = obj_factory(criterion_pixelwise).to(device)
    criterion_id = obj_factory(criterion_id).to(device)
    criterion_attr = obj_factory(criterion_attr).to(device)
    criterion_gan = obj_factory(criterion_gan).to(device)

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        G = nn.DataParallel(G, gpus)
        D = nn.DataParallel(D, gpus)
        criterion_id.vgg = nn.DataParallel(criterion_id.vgg, gpus)
        criterion_attr.vgg = nn.DataParallel(criterion_attr.vgg, gpus)
        landmarks_decoders = [nn.DataParallel(ld, gpus) for ld in landmarks_decoders]

    # For each resolution
    start_res_ind = int(np.log2(curr_res)) - int(np.log2(resolutions[0]))
    start_epoch = 0 if start_epoch is None else start_epoch
    for ri in range(start_res_ind, len(resolutions)):
        res = resolutions[ri]
        res_lr_gen = lr_gen[ri]
        res_lr_dis = lr_dis[ri]
        res_epochs = epochs[ri]
        res_iterations = iterations[ri] if iterations is not None else None
        res_batch_size = batch_size[ri]
        res_landmarks_decoders = landmarks_decoders[-ri - 1:]

        # Optimizer and scheduler
        optimizer_G = obj_factory(optimizer, G.parameters(), lr=res_lr_gen)
        optimizer_D = obj_factory(optimizer, D.parameters(), lr=res_lr_dis)
        scheduler_G = obj_factory(scheduler, optimizer_G)
        scheduler_D = obj_factory(scheduler, optimizer_D)
        if optimizer_G_state is not None:
            optimizer_G.load_state_dict(optimizer_G_state)
            optimizer_G_state = None
        if optimizer_D_state is not None:
            optimizer_D.load_state_dict(optimizer_D_state)
            optimizer_D_state = None

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
                val_iterations = (res_iterations * len(val_dataset.classes)) // len(train_dataset.classes)
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

            # Schedulers step (in PyTorch 1.1.0+ it must follow after the epoch training and validation steps)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_G.step(total_loss)
                scheduler_D.step(total_loss)
            else:
                scheduler_G.step()
                scheduler_D.step()

            # Save models checkpoints
            is_best = total_loss < best_loss
            best_loss = min(best_loss, total_loss)
            utils.save_checkpoint(exp_dir, 'G', {
                'resolution': res,
                'epoch': epoch + 1,
                'state_dict': G.module.state_dict() if gpus and len(gpus) > 1 else G.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'best_loss_%d' % res: best_loss,
                'arch': G_arch,
            }, is_best)
            utils.save_checkpoint(exp_dir, 'D', {
                'resolution': res,
                'epoch': epoch + 1,
                'state_dict': D.module.state_dict() if gpus and len(gpus) > 1 else D.state_dict(),
                'optimizer': optimizer_D.state_dict(),
                'best_loss_%d' % res: best_loss,
                'arch': D_arch,
            }, is_best)

        # Reset start epoch to 0 because it's should only effect the first training resolution
        start_epoch = 0
        best_loss = 1e6


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    general = parser.add_argument_group('general')
    general.add_argument('exp_dir', metavar='DIR',
                         help='path to experiment directory')
    general.add_argument('-re', '--resume', metavar='DIR',
                         help='path to latest checkpoint (default: None)')
    general.add_argument('-se', '--start-epoch', metavar='N',
                         help='manual epoch number (useful on restarts)')
    general.add_argument('-e', '--epochs', default=90, type=int, nargs='+', metavar='N',
                         help='number of total epochs to run')
    general.add_argument('-i', '--iterations', nargs='+', metavar='N',
                         help='number of iterations per resolution to run')
    general.add_argument('-r', '--resolutions', default=(128, 256), type=int, nargs='+', metavar='N',
                         help='the training resolutions list (must be power of 2)')
    parser.add_argument('-lrg', '--lr_gen', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial generator learning rate per resolution')
    parser.add_argument('-lrd', '--lr_dis', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial discriminator learning rate per resolution')
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
    data.add_argument('-td', '--train_dataset', default='opencv_video_seq_dataset.VideoSeqDataset',
                      help='train dataset object')
    data.add_argument('-vd', '--val_dataset',
                      help='val dataset object')
    data.add_argument('-nt', '--numpy_transforms', nargs='+',
                      help='Numpy transforms')
    data.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                      default=('img_lms_pose_transforms.ToTensor()', 'img_lms_pose_transforms.Normalize()'))

    training = parser.add_argument_group('training')
    training.add_argument('-o', '--optimizer', default='optim.SGD(momentum=0.9,weight_decay=1e-4)',
                          help='network\'s optimizer object')
    training.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
                          help='scheduler object')
    training.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                          help='use pre-trained model')
    training.add_argument('-cp', '--criterion_pixelwise', default='nn.L1Loss',
                          help='pixelwise criterion object')
    training.add_argument('-ci', '--criterion_id', default='vgg_loss.VGGLoss',
                          help='id criterion object')
    training.add_argument('-ca', '--criterion_attr', default='vgg_loss.VGGLoss',
                          help='attributes criterion object')
    training.add_argument('-cg', '--criterion_gan', default='gan_loss.GANLoss(use_lsgan=True)',
                          help='GAN criterion object')
    training.add_argument('-g', '--generator', default='res_unet.MultiScaleResUNet(in_nc=101,out_nc=3)',
                          help='generator object')
    training.add_argument('-d', '--discriminator', default='discriminators_pix2pix.MultiscaleDiscriminator',
                          help='discriminator object')
    training.add_argument('-rw', '--rec_weight', default=1.0, type=float, metavar='F',
                          help='reconstruction loss weight')
    training.add_argument('-gw', '--gan_weight', default=0.001, type=float, metavar='F',
                          help='GAN loss weight')
    main(**vars(parser.parse_args()))
