import os
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from fsgan.datasets.image_seg_dataset import ImageSegDataset
from fsgan.datasets.img_landmarks_transforms import Crop, Resize, RandomHorizontalFlip, RandomRotation
from fsgan.datasets.img_landmarks_transforms import ColorJitter, RandomGaussianBlur
from fsgan.models.simple_unet_02 import UNet
from fsgan.train_segmentation import main


if __name__ == '__main__':
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('../results/segmentation', exp_name)
    train_dataset = partial(ImageSegDataset, '/data/datasets/celeba_mask_hq',
                            'img_list_train.txt', 'bboxes_train.npy', seg_classes=3)
    val_dataset = partial(ImageSegDataset, '/data/datasets/celeba_mask_hq',
                          'img_list_val.txt', 'bboxes_val.npy', seg_classes=3)
    numpy_transforms = [RandomRotation(30.0, ('cubic', 'nearest')), Crop(), Resize(256, ('cubic', 'nearest')),
                        RandomHorizontalFlip(), ColorJitter(0.5, 0.5, 0.5, 0.5, filter=(True, False)),
                        RandomGaussianBlur(filter=(True, False))]
    resolutions = [256]
    learning_rate = [1e-4]
    epochs = [60]
    iterations = ['40k']
    batch_size = [48]
    workers = 12
    pretrained = False
    optimizer = partial(optim.Adam, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    model = partial(UNet)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, numpy_transforms=numpy_transforms,
         resolutions=resolutions, learning_rate=learning_rate, epochs=epochs, iterations=iterations,
         batch_size=batch_size, workers=workers, optimizer=optimizer, scheduler=scheduler, pretrained=pretrained,
         criterion=criterion, model=model)

    os.system('sudo shutdown')
