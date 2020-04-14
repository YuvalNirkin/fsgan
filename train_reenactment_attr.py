""" Training script for the face reenactment model using an additional attribute based perceptial loss. """

import os
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils as tutils
import fsgan.data.landmark_transforms as landmark_transforms
import numpy as np
from tqdm import tqdm
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils import utils
from fsgan.utils import img_utils
from fsgan.utils import seg_utils
from fsgan.loggers.tensorboard_logger import TensorBoardLogger


def main(exp_dir, train_dir, val_dir=None, workers=4, iterations=None, epochs=(90,), start_epoch=None,
         lr_gen=(1e-4,), lr_dis=(1e-4,), batch_size=(64,), resolutions=(128, 256), resume_dir=None, seed=None,
         gpus=None, tensorboard=False,
         train_dataset='fsgan.data.face_landmarks_dataset.FacePairLandmarksDataset', val_dataset=None,
         optimizer='optim.Adam(lr=1e-4,betas=(0.5,0.999))',
         scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
         criterion_pixelwise='nn.L1Loss', criterion_id='vgg_loss.VGGLoss', criterion_attr='vgg_loss.VGGLoss',
         criterion_gan='gan_loss.GANLoss(use_lsgan=False)',
         log_freq=20, pair_transforms=None, pil_transforms1=None, pil_transforms2=None,
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         generator='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3))',
         discriminator='discriminators_pix2pix.MultiscaleDiscriminator',
         seg_model=None, seg_weight=0.1, rec_weight=1.0, gan_weight=0.001):
    iterations = iterations * len(resolutions) if len(iterations) == 1 else iterations
    epochs = epochs * len(resolutions) if len(epochs) == 1 else epochs
    batch_size = batch_size * len(resolutions) if len(batch_size) == 1 else batch_size
    iterations = utils.str2int(iterations)

    # Validation
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')
    assert len(iterations) == len(resolutions)
    assert len(epochs) == len(resolutions)
    assert len(batch_size) == len(resolutions)

    # Seed
    utils.set_seed(seed)

    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)

    # Initialize loggers
    logger = TensorBoardLogger(log_dir=exp_dir if tensorboard else None)

    # Initialize datasets
    pair_transforms = obj_factory(pair_transforms)
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_pair_transforms = landmark_transforms.ComposePair(pair_transforms)
    img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    val_dataset = train_dataset if val_dataset is None else val_dataset
    train_dataset = obj_factory(train_dataset, train_dir, pair_transform=img_pair_transforms,
                                transform1=img_transforms1, transform2=img_transforms2)
    if val_dir:
        val_dataset = obj_factory(val_dataset, val_dir, pair_transform=img_pair_transforms,
                                  transform1=img_transforms1, transform2=img_transforms2)

    # Create networks
    G = obj_factory(generator).to(device)
    D = obj_factory(discriminator).to(device)

    # Resume from a checkpoint or initialize the networks weights randomly
    checkpoint_dir = exp_dir if resume_dir is None else resume_dir
    G_path = os.path.join(checkpoint_dir, 'G_latest.pth')
    D_path = os.path.join(checkpoint_dir, 'D_latest.pth')
    best_loss = 1000000.
    curr_res = resolutions[0]
    if os.path.isfile(G_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        # G
        checkpoint = torch.load(G_path)
        if 'resolution' in checkpoint:
            curr_res = checkpoint['resolution']
            start_epoch = checkpoint['epoch'] if start_epoch is None else start_epoch
        else:
            curr_res = resolutions[1] if len(resolutions) > 1 else resolutions[0]
        best_loss = checkpoint['best_loss']
        G.apply(utils.init_weights)
        G.load_state_dict(checkpoint['state_dict'], strict=False)
        # D
        D.apply(utils.init_weights)
        if os.path.isfile(D_path):
            checkpoint = torch.load(D_path)
            D.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        print("=> randomly initializing networks...")
        G.apply(utils.init_weights)
        D.apply(utils.init_weights)

    # Initialize segmentation model
    if seg_model is None:
        raise RuntimeError('Segmentation model must be specified!')
    if not os.path.exists(seg_model):
        raise RuntimeError('Couldn\'t find segmentation model in path: ' + seg_model)
    checkpoint = torch.load(seg_model)
    S = obj_factory(checkpoint['arch']).to(device)
    S.load_state_dict(checkpoint['state_dict'])

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

        # Optimizer and scheduler
        optimizer_G = obj_factory(optimizer, G.parameters(), lr=res_lr_gen)
        optimizer_D = obj_factory(optimizer, D.parameters(), lr=res_lr_dis)
        scheduler_G = obj_factory(scheduler, optimizer_G)
        scheduler_D = obj_factory(scheduler, optimizer_D)

        # Initialize data loaders
        if res_iterations is None:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, len(train_dataset))
        else:
            train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, res_iterations)
        train_loader = tutils.data.DataLoader(train_dataset, batch_size=res_batch_size, sampler=train_sampler,
                                              num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)
        if val_dir:
            if res_iterations is None:
                val_sampler = tutils.data.sampler.WeightedRandomSampler(val_dataset.weights, len(val_dataset))
            else:
                val_iterations = (res_iterations * len(val_dataset.classes)) // len(train_dataset.classes)
                val_sampler = tutils.data.sampler.WeightedRandomSampler(val_dataset.weights, val_iterations)
            val_loader = tutils.data.DataLoader(val_dataset, batch_size=res_batch_size, sampler=val_sampler,
                                                num_workers=workers, pin_memory=True, drop_last=True, shuffle=False)

        # For each epoch
        for epoch in range(start_epoch, res_epochs):
            total_iter = len(train_loader) * train_loader.batch_size * epoch
            pbar = tqdm(train_loader, unit='batches')

            # Set networks training mode
            G.train(True)
            D.train(True)
            S.train(False)
            criterion_id.train(False)
            criterion_attr.train(False)

            # Reset logger
            logger.reset(prefix='TRAINING {}X{}: Epoch: {} / {}; LR: {:.0e}; '.format(
                res, res, epoch + 1, res_epochs, scheduler_G.get_lr()[0]))

            # For each batch in the training data
            for i, (img1, landmarks1, bbox1, target1, img2, landmarks2, bbox2, target2) in enumerate(pbar):
                # Prepare input
                img1 = img1[-ri - 1:]
                img2 = img2[-ri - 1:]
                landmarks2 = landmarks2[-ri - 1:]
                input = []
                for j in range(len(img1)):
                    img1[j] = img1[j].to(device)
                    img2[j] = img2[j].to(device)
                    landmarks2[j] = landmarks2[j].to(device)
                    input.append(torch.cat((img1[j], landmarks2[j]), dim=1))

                # Reenactment and segmentation
                img_pred, seg_pred = G(input)
                seg = S(img2[0])

                # Fake Detection and Loss
                img_pred_pyd = img_utils.create_pyramid(img_pred, len(img1))
                pred_fake_pool = D([x.detach() for x in img_pred_pyd])
                loss_D_fake = criterion_gan(pred_fake_pool, False)

                # Real Detection and Loss
                pred_real = D(img2)
                loss_D_real = criterion_gan(pred_real, True)

                loss_D_total = (loss_D_fake + loss_D_real) * 0.5

                # GAN loss (Fake Passability Loss)
                pred_fake = D(img_pred_pyd)
                loss_G_GAN = criterion_gan(pred_fake, True)

                # Reconstruction and segmentation loss
                loss_pixelwise = criterion_pixelwise(img_pred, img2[0])
                loss_id = criterion_id(img_pred, img2[0])
                loss_attr = criterion_attr(img_pred, img2[0])
                loss_rec = 0.1 * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr
                loss_seg = criterion_pixelwise(seg_pred, seg)

                loss_G_total = rec_weight * loss_rec + seg_weight * loss_seg + gan_weight * loss_G_GAN

                # Update generator weights
                optimizer_G.zero_grad()
                loss_G_total.backward()
                optimizer_G.step()

                # Update discriminator weights
                optimizer_D.zero_grad()
                loss_D_total.backward()
                optimizer_D.step()

                logger.update(pixelwise=loss_pixelwise, id=loss_id, rec=loss_rec, seg=loss_seg, g_gan=loss_G_GAN,
                              d_gan=loss_D_total)
                total_iter += train_loader.batch_size

                # Batch logs
                pbar.set_description(str(logger))
                if i % log_freq == 0:
                    logger.log_scalars_val('%dx%d/batch/loss' % (res, res), total_iter)

            # Epoch logs
            logger.log_scalars_avg('%dx%d/epoch/losses/train' % (res, res), epoch)

            # Set networks training mode
            G.train(False)
            D.train(False)
            S.train(False)
            criterion_id.train(False)
            criterion_attr.train(False)

            with torch.no_grad():
                logger.reset(prefix='VALIDATION {}X{}: Epoch: {} / {}; LR: {:.0e}; '.format(
                    res, res, epoch + 1, res_epochs, scheduler_G.get_lr()[0]))
                total_iter = len(val_loader) * val_loader.batch_size * epoch
                pbar = tqdm(val_loader, unit='batches')

                # For each batch in the validation data
                for i, (img1, landmarks1, bbox1, target1, img2, landmarks2, bbox2, target2) in enumerate(pbar):
                    # Prepare input
                    img1 = img1[-ri - 1:]
                    img2 = img2[-ri - 1:]
                    landmarks2 = landmarks2[-ri - 1:]
                    input = []
                    for j in range(len(img1)):
                        img1[j] = img1[j].to(device)
                        img2[j] = img2[j].to(device)
                        landmarks2[j] = landmarks2[j].to(device)
                        input.append(torch.cat((img1[j], landmarks2[j]), dim=1))

                    # Reenactment and segmentation
                    img_pred, seg_pred = G(input)
                    seg = S(img2[0])

                    # Fake Detection and Loss
                    img_pred_pyd = img_utils.create_pyramid(img_pred, len(img1))
                    pred_fake_pool = D([x.detach() for x in img_pred_pyd])
                    loss_D_fake = criterion_gan(pred_fake_pool, False)

                    # Real Detection and Loss
                    pred_real = D(img2)
                    loss_D_real = criterion_gan(pred_real, True)

                    loss_D_total = (loss_D_fake + loss_D_real) * 0.5

                    # GAN loss (Fake Passability Loss)
                    pred_fake = D(img_pred_pyd)
                    loss_G_GAN = criterion_gan(pred_fake, True)

                    # Reconstruction and segmentation loss
                    loss_pixelwise = criterion_pixelwise(img_pred, img2[0])
                    loss_id = criterion_id(img_pred, img2[0])
                    loss_attr = criterion_attr(img_pred, img2[0])
                    loss_rec = 0.1 * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr
                    loss_seg = criterion_pixelwise(seg_pred, seg)

                    # loss_G_total = rec_weight * loss_rec + seg_weight * loss_seg + gan_weight * loss_G_GAN

                    logger.update(pixelwise=loss_pixelwise, id=loss_id, rec=loss_rec, seg=loss_seg, g_gan=loss_G_GAN,
                                  d_gan=loss_D_total)
                    total_iter += train_loader.batch_size

                    # Batch logs
                    pbar.set_description(str(logger))

                # Validation epoch logs
                logger.log_scalars_avg('%dx%d/epoch/losses/val' % (res, res), epoch)

                # Log images
                blend_pred = seg_utils.blend_seg_pred(img2[0], seg_pred)
                grid = img_utils.make_grid(img1[0], img_pred, img2[0], blend_pred)
                logger.log_image('%dx%d/rec' % (res, res), grid, epoch)

            # Schedulers step (in PyTorch 1.1.0+ it must follow after the epoch training and validation steps)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_G.step(logger.log_dict['rec'].avg)
                scheduler_D.step(logger.log_dict['rec'].avg)
            else:
                scheduler_G.step()
                scheduler_D.step()

            # Save models checkpoints
            is_best = logger.log_dict['rec'].avg < best_loss
            best_loss = min(best_loss, logger.log_dict['rec'].avg)
            utils.save_checkpoint(exp_dir, 'G', {
                'resolution': res,
                'epoch': epoch + 1,
                'state_dict': G.module.state_dict() if gpus and len(gpus) > 1 else G.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'best_loss': best_loss,
            }, is_best)
            utils.save_checkpoint(exp_dir, 'D', {
                'resolution': res,
                'epoch': epoch + 1,
                'state_dict': D.module.state_dict() if gpus and len(gpus) > 1 else D.state_dict(),
                'optimizer': optimizer_D.state_dict(),
                'best_loss': best_loss,
            }, is_best)

        # Reset start epoch to 0 because it's should only effect the first training resolution
        start_epoch = 0


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('train_reenactment_attr')
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val', default=None, type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-e', '--epochs', default=90, type=int, nargs='+', metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-lrg', '--lr_gen', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial generator learning rate per resolution')
    parser.add_argument('-lrd', '--lr_dis', default=(1e-4,), type=float, nargs='+',
                        metavar='F', help='initial discriminator learning rate per resolution')
    parser.add_argument('-b', '--batch-size', default=64, type=int, nargs='+',
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-res', '--resolutions', default=(128, 256), type=int, nargs='+',
                        metavar='N', help='the training resolutions list (must be power of 2)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-td', '--train_dataset', default='fsgan.data.face_landmarks_dataset.FacePairLandmarksDataset',
                        type=str, help='train dataset object')
    parser.add_argument('-vd', '--val_dataset', default=None, type=str, help='val dataset object')
    parser.add_argument('-o', '--optimizer', default='optim.Adam(lr=1e-4,betas=(0.5,0.999))', type=str,
                        help='network\'s optimizer object')
    parser.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=30,gamma=0.1)', type=str,
                        help='scheduler object')
    parser.add_argument('-cp', '--criterion_pixelwise', default='nn.L1Loss', type=str,
                        help='pixelwise criterion object')
    parser.add_argument('-ci', '--criterion_id', default='vgg_loss.VGGLoss', type=str,
                        help='id criterion object')
    parser.add_argument('-ca', '--criterion_attr', default='vgg_loss.VGGLoss', type=str,
                        help='criterion object')
    parser.add_argument('-cg', '--criterion_gan', default='gan_loss.GANLoss(use_lsgan=False)', type=str,
                        help='GAN criterion object')
    parser.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-pt', '--pair_transforms', default=None, nargs='+', help='pair PIL transforms')
    parser.add_argument('-pt1', '--pil_transforms1', default=None, nargs='+', help='first PIL transforms')
    parser.add_argument('-pt2', '--pil_transforms2', default=None, nargs='+', help='second PIL transforms')
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-g', '--generator', default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3))',
                        help='generator object')
    parser.add_argument('-d', '--discriminator', default='discriminators_pix2pix.MultiscaleDiscriminator',
                        help='discriminator object')
    parser.add_argument('-sm', '--seg_model', default=None, metavar='PATH',
                        help='segmentation model')
    parser.add_argument('-sw', '--seg_weight', default=0.1, type=float, metavar='F',
                        help='segmentation weight')
    parser.add_argument('-rw', '--rec_weight', default=1.0, type=float, metavar='F',
                        help='reconstruction loss weight')
    parser.add_argument('-gw', '--gan_weight', default=0.001, type=float, metavar='F',
                        help='GAN loss weight')
    args = parser.parse_args()
    main(args.exp_dir, args.train, args.val, workers=args.workers, iterations=args.iterations, epochs=args.epochs,
         start_epoch=args.start_epoch, lr_gen=args.lr_gen, lr_dis=args.lr_dis, batch_size=args.batch_size,
         resolutions=args.resolutions, resume_dir=args.resume, seed=args.seed, gpus=args.gpus,
         tensorboard=args.tensorboard, optimizer=args.optimizer, scheduler=args.scheduler,
         criterion_pixelwise=args.criterion_pixelwise, criterion_id=args.criterion_id,
         criterion_attr=args.criterion_attr, criterion_gan=args.criterion_gan,
         log_freq=args.log_freq, train_dataset=args.train_dataset, val_dataset=args.val_dataset,
         pair_transforms=args.pair_transforms,
         pil_transforms1=args.pil_transforms1, pil_transforms2=args.pil_transforms2,
         tensor_transforms1=args.tensor_transforms1, tensor_transforms2=args.tensor_transforms2,
         generator=args.generator, discriminator=args.discriminator,
         seg_model=args.seg_model, seg_weight=args.seg_weight, rec_weight=args.rec_weight, gan_weight=args.gan_weight)
