""" General utilities. """

import os
import shutil
import torch
import random
import torch.nn.init as init
import warnings
import torch.backends.cudnn as cudnn


def init_weights(m, init_type='normal', gain=0.02):
    """ Randomly initialize a module's weights.

    Args:
        m (nn.Module): The module to initialize its weights
        init_type (str): Initialization type: 'normal', 'xavier', 'kaiming', or 'orthogonal'
        gain (float): Standard deviation of the normal distribution
    """
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """ Saves a model's checkpoint.

    Args:
        exp_dir (str): Experiment directory to save the checkpoint into.
        base_name (str): The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
        state (dict): The model state to save.
        is_best (bool): If True, <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + '_latest.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + '_best.pth'))


class ImagePool:
    """ Defines an image pool for improving GAN training.

    Given an image query, the images will be replaced with previous images with probability 0.5.

    Args:
        pool_size (int): The maximum number of images in the pool
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def next_pow2(n):
    n += (n == 0)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


mag_map = {'K': 3, 'M': 6, 'B': 9}


def str2int(s):
    """ Converts a string containing a number with 'K', 'M', or 'B' to an integer. """
    if isinstance(s, (list, tuple)):
        return [str2int(o) for o in s]
    if not isinstance(s, str):
        return s
    return int(float(s[:-1]) * 10 ** mag_map[s[-1].upper()]) if s[-1].upper() in mag_map else int(s)


def set_seed(seed):
    """ Sets random seed for deterministic behaviour. """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def set_device(gpus=None, use_cuda=None):
    """ Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


def topk_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k. """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred    = pred.t()
    pred = pred.view(batch_size, -1)
    target.view(-1, 1).expand_as(pred)

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:, :k].view(-1).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    import torch

    output = torch.rand(2, 10, 1, 1)
    target = torch.LongTensor(range(2))
    acc = topk_accuracy(output, target, topk=(1, 5))
    print(acc)


if __name__ == "__main__":
    main()
