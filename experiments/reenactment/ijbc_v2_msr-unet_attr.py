import os
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from fsgan.data.face_landmarks_dataset import FacePairLandmarksDataset
import fsgan.data.landmark_transforms as landmark_transforms
from fsgan.models.res_unet_split import MultiScaleResUNet
from fsgan.models.discriminators_pix2pix import MultiscaleDiscriminator
from fsgan.criterions.vgg_loss import VGGLoss
from fsgan.criterions.gan_loss import GANLoss
from fsgan.train_reenactment_attr import main


if __name__ == '__main__':
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('../results/reenactment', exp_name)
    train_dir = val_dir = '/data/datasets/ijb-c/ijbc_video_keyframes_v2'
    train_dataset = partial(FacePairLandmarksDataset, img_list='img_list_train.txt',
                            landmarks_list='landmarks_train.npy', bboxes_list='bboxes_train.npy', same_prob=1)
    val_dataset = partial(FacePairLandmarksDataset, img_list='img_list_val.txt',
                          landmarks_list='landmarks_val.npy', bboxes_list='bboxes_val.npy', same_prob=1)
    pair_transforms = [landmark_transforms.RandomHorizontalFlipPair()]
    pil_transforms1 = [landmark_transforms.FaceAlignCrop(), landmark_transforms.Resize(256),
                       landmark_transforms.Pyramids(2)]
    pil_transforms2 = [landmark_transforms.FaceAlignCrop(), landmark_transforms.Resize(256),
                       landmark_transforms.Pyramids(2), landmark_transforms.LandmarksToHeatmaps()]
    epochs = [50]
    batch_size = [24, 12]
    iterations = ['40k']
    workers = 16
    generator = MultiScaleResUNet(in_nc=71, out_nc=(3, 3), flat_layers=(2, 0, 2, 3), ngf=128)
    discriminator = MultiscaleDiscriminator(use_sigmoid=True, num_D=2)
    criterion_id = VGGLoss('../../weights/vggface2_vgg19_256_2_0_id.pth')
    criterion_attr = VGGLoss('../../weights/celeba_vgg19_256_2_0_28_attr.pth')
    criterion_gan = GANLoss(use_lsgan=True)
    optimizer = partial(optim.Adam, lr=4e-5, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=10, gamma=0.5)
    seg_model = '../../weights/lfw_figaro_unet_256_segmentation.pth'
    seg_weight = 0.1
    resume_dir = None
    start_epoch = 0

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    main(exp_dir, train_dir, val_dir, train_dataset=train_dataset, val_dataset=val_dataset,
         pair_transforms=pair_transforms, pil_transforms1=pil_transforms1, pil_transforms2=pil_transforms2,
         epochs=epochs, batch_size=batch_size, iterations=iterations, workers=workers, tensorboard=True, gpus=None,
         generator=generator, discriminator=discriminator, criterion_id=criterion_id, criterion_attr=criterion_attr,
         criterion_gan=criterion_gan, optimizer=optimizer, scheduler=scheduler,
         seg_model=seg_model, seg_weight=seg_weight, resume_dir=resume_dir, start_epoch=start_epoch)

    os.system('sudo shutdown')
