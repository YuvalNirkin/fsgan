import os
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from fsgan.data.seg_landmarks_dataset import SegmentationLandmarksPairDataset
import fsgan.data.seg_landmark_transforms as seg_landmark_transforms
import fsgan.models.res_unet_split as res_unet_split
import fsgan.models.res_unet as res_unet
from fsgan.models.discriminators_pix2pix import MultiscaleDiscriminator
from fsgan.criterions.vgg_loss import VGGLoss
from fsgan.criterions.gan_loss import GANLoss
from fsgan.train_inpainting import main


if __name__ == '__main__':
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('../results/swapping', exp_name)
    train_dir = val_dir = '/data/datasets/ijb-c/ijbc_video_keyframes_v2'
    train_dataset = partial(SegmentationLandmarksPairDataset, img_list='seg_splits/img_list_train.txt',
                            landmarks_list='seg_splits/landmarks_train.npy', bboxes_list='seg_splits/bboxes_train.npy',
                            seg_list='seg_splits/seg_list_train.txt', same_prob=1)
    val_dataset = partial(SegmentationLandmarksPairDataset, img_list='seg_splits/img_list_val.txt',
                          landmarks_list='seg_splits/landmarks_val.npy', bboxes_list='seg_splits/bboxes_val.npy',
                          seg_list='seg_splits/seg_list_val.txt', same_prob=1)
    pair_transforms = [seg_landmark_transforms.RandomHorizontalFlipPair()]
    pil_transforms1 = [seg_landmark_transforms.FaceAlignCrop(), seg_landmark_transforms.Resize(256),
                       seg_landmark_transforms.Pyramids(2)]
    pil_transforms2 = [seg_landmark_transforms.FaceAlignCrop(), seg_landmark_transforms.Resize(256),
                       seg_landmark_transforms.Pyramids(2), seg_landmark_transforms.LandmarksToHeatmaps()]
    resolutions = [128, 256]
    lr_gen = [1e-4, 4e-5]
    lr_dis = [1e-5, 4e-6]
    epochs = [20, 40]
    batch_size = [32, 16]
    iterations = ['40k', '40k']
    workers = 16
    gen_renderer = res_unet_split.MultiScaleResUNet(in_nc=71, out_nc=(3, 3), flat_layers=(2, 0, 2, 3), ngf=128)
    gen_inpainter = res_unet.MultiScaleResUNet(in_nc=4, out_nc=3, flat_layers=(2, 0, 2, 3), ngf=128)
    discriminator = MultiscaleDiscriminator(use_sigmoid=True, num_D=2)
    criterion_id = VGGLoss('../../weights/vggface2_vgg19_256_2_0_id.pth')
    criterion_gan = GANLoss(use_lsgan=True)
    optimizer = partial(optim.Adam, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=10, gamma=0.5)
    gan_weight = 0.0
    background_value = -1.0
    resume_dir = '../results/reenactment/ijbc_v2_msr-unet_attr'
    start_epoch = 0

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    main(exp_dir, train_dir, val_dir, train_dataset=train_dataset, val_dataset=val_dataset,
         pair_transforms=pair_transforms, pil_transforms1=pil_transforms1, pil_transforms2=pil_transforms2,
         resolutions=resolutions, lr_gen=lr_gen, lr_dis=lr_dis, epochs=epochs, batch_size=batch_size,
         iterations=iterations, workers=workers, tensorboard=True, gpus=None, gen_renderer=gen_renderer,
         gen_inpainter=gen_inpainter, discriminator=discriminator, criterion_id=criterion_id,
         criterion_gan=criterion_gan, optimizer=optimizer, scheduler=scheduler, gan_weight=gan_weight,
         background_value=background_value, resume_dir=resume_dir, start_epoch=start_epoch)

    os.system('sudo shutdown')
