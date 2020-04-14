import os
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from fsgan.datasets.opencv_video_seq_dataset import VideoSeqPairDataset
from fsgan.datasets.img_landmarks_transforms import RandomHorizontalFlip, Pyramids, ToTensor
from fsgan.criterions.vgg_loss import VGGLoss
from fsgan.criterions.gan_loss import GANLoss
from fsgan.models.res_unet import MultiScaleResUNet
from fsgan.models.discriminators_pix2pix import MultiscaleDiscriminator
from fsgan.train_blending import main


if __name__ == '__main__':
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('../results/swapping', exp_name)
    train_dataset = partial(VideoSeqPairDataset, '/data/datasets/ijb-c/ijbc_cropped/ijbc_cropped_r256_cs1.2',
                            'train_list.txt', frame_window=1, ignore_landmarks=True, same_prob=0.0)
    val_dataset = partial(VideoSeqPairDataset, '/data/datasets/ijb-c/ijbc_cropped/ijbc_cropped_r256_cs1.2',
                          'val_list.txt', frame_window=1, ignore_landmarks=True, same_prob=0.0)
    numpy_transforms = [RandomHorizontalFlip(), Pyramids(2)]
    tensor_transforms = [ToTensor()]
    resolutions = [128, 256]
    lr_gen = [1e-4, 4e-5]
    lr_dis = [1e-5, 4e-6]
    epochs = [24, 50]
    iterations = ['20k']
    batch_size = [32, 16]
    workers = 32
    pretrained = False
    criterion_id = VGGLoss('../../weights/vggface2_vgg19_256_1_2_id.pth')
    criterion_attr = VGGLoss('../../weights/celeba_vgg19_256_2_0_28_attr.pth')
    criterion_gan = GANLoss(use_lsgan=True)
    generator = MultiScaleResUNet(in_nc=7, out_nc=3, flat_layers=(2, 2, 2, 2), ngf=128)
    discriminator = MultiscaleDiscriminator(use_sigmoid=True, num_D=2)
    optimizer = partial(optim.Adam, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=10, gamma=0.5)
    reenactment_model = '../results/reenactment/ijbc_msrunet_reenactment_attr_no_seg/G_latest.pth'
    seg_model = '../../weights/lfw_figaro_unet_256_segmentation.pth'
    lms_model = '../../weights/hr18_wflw_landmarks.pth'
    rec_weight = 1.0
    gan_weight = 0.1
    background_value = -1.0

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset,
         numpy_transforms=numpy_transforms, tensor_transforms=tensor_transforms, resolutions=resolutions,
         lr_gen=lr_gen, lr_dis=lr_dis, epochs=epochs, iterations=iterations, batch_size=batch_size, workers=workers,
         optimizer=optimizer, scheduler=scheduler, pretrained=pretrained,
         criterion_id=criterion_id, criterion_attr=criterion_attr, criterion_gan=criterion_gan,
         generator=generator, discriminator=discriminator, reenactment_model=reenactment_model, seg_model=seg_model,
         lms_model=lms_model, rec_weight=rec_weight, gan_weight=gan_weight, background_value=background_value)

    os.system('sudo shutdown')
