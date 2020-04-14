import os
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from fsgan.data.seg_landmarks_dataset import SegmentationLandmarksPairDataset
import fsgan.data.seg_landmark_transforms as seg_landmark_transforms
from fsgan.models.res_unet_split import MultiScaleResUNet
from fsgan.models.discriminators_pix2pix import MultiscaleDiscriminator
from fsgan.criterions.vgg_loss import VGGLoss
from fsgan.criterions.gan_loss import GANLoss
from fsgan.train_reenactment_stepwise import main


if __name__ == '__main__':
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('../results/reenactment', exp_name)
    train_dir = val_dir = '/data/datasets/ijb-c/ijbc_video_keyframes_v2'
    train_dataset = partial(SegmentationLandmarksPairDataset, img_list='seg_splits/img_list_train.txt',
                            landmarks_list='seg_splits/landmarks_3d_train.npy',
                            bboxes_list='seg_splits/bboxes_train.npy', seg_list='seg_splits/seg_list_train.txt',
                            same_prob=1)
    val_dataset = partial(SegmentationLandmarksPairDataset, img_list='seg_splits/img_list_val.txt',
                          landmarks_list='seg_splits/landmarks_3d_val.npy', bboxes_list='seg_splits/bboxes_val.npy',
                          seg_list='seg_splits/seg_list_val.txt', same_prob=1)
    pair_transforms = [seg_landmark_transforms.RandomHorizontalFlipPair()]
    pil_transforms1 = [seg_landmark_transforms.FaceAlignCrop(bbox_scale=1.2), seg_landmark_transforms.Resize(256),
                       seg_landmark_transforms.Pyramids(2)]
    pil_transforms2 = [seg_landmark_transforms.FaceAlignCrop(bbox_scale=1.2), seg_landmark_transforms.Resize(256),
                       seg_landmark_transforms.Pyramids(2)]
    resolutions = [128, 256]
    lr_gen = [1e-4, 4e-5]
    lr_dis = [1e-5, 4e-6]
    epochs = [40, 60]
    batch_size = [32, 16]
    iterations = ['40k', '40k']
    workers = 16
    generator = MultiScaleResUNet(in_nc=71, out_nc=(3, 3), flat_layers=(2, 0, 2, 3), ngf=128)
    discriminator = MultiscaleDiscriminator(use_sigmoid=True, num_D=2)
    criterion_id = VGGLoss('../../weights/vggface2_vgg19_256_2_0_id.pth')
    criterion_attr = VGGLoss('../../weights/celeba_vgg19_256_2_0_28_attr.pth')
    criterion_gan = GANLoss(use_lsgan=True)
    optimizer = partial(optim.Adam, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=10, gamma=0.5)
    seg_weight = 0.01

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    main(exp_dir, train_dir, val_dir, train_dataset=train_dataset, val_dataset=val_dataset,
         pair_transforms=pair_transforms, pil_transforms1=pil_transforms1, pil_transforms2=pil_transforms2,
         epochs=epochs, lr_gen=lr_gen, lr_dis=lr_dis, batch_size=batch_size, iterations=iterations,
         resolutions=resolutions, workers=workers, generator=generator, discriminator=discriminator,
         criterion_id=criterion_id, criterion_attr=criterion_attr, criterion_gan=criterion_gan,
         optimizer=optimizer, scheduler=scheduler, seg_weight=seg_weight, tensorboard=True, gpus=None)

    os.system('sudo shutdown')

