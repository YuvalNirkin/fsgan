""" General utilities. """

import os
import shutil
from functools import partial
import torch
import random
import warnings
import requests
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from fsgan.utils.obj_factory import extract_args, obj_factory


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
    elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def set_device(gpus=None, use_cuda=True):
    """ Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    use_cuda = torch.cuda.is_available() if use_cuda else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


def set_seed(seed):
    """ Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


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


mag_map = {'K': 3, 'M': 6, 'B': 9}


def str2int(s):
    """ Converts a string containing a number with 'K', 'M', or 'B' to an integer. """
    if isinstance(s, (list, tuple)):
        return [str2int(o) for o in s]
    if not isinstance(s, str):
        return s
    return int(float(s[:-1]) * 10 ** mag_map[s[-1].upper()]) if s[-1].upper() in mag_map else int(s)


def get_arch(obj, *args, **kwargs):
    """ Extract the architecture (string representation) of an object given as a string or partial together
    with additional provided arguments.

    The returned architecture can be used to create the object using the obj_factory function.

    Args:
        obj (str or partial): The object string expresion or partial to be converted into an object
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        arch (str): The object's architecture (string representation).
    """
    obj_args, obj_kwargs = [], {}
    if isinstance(obj, str):
        if '(' in obj and ')' in obj:
            arg_pos = obj.find('(')
            func = obj[:arg_pos]
            args_exp = obj[arg_pos:]
            obj_args, obj_kwargs = eval('extract_args' + args_exp)
        else:
            func = obj
    elif isinstance(obj, partial):
        func = obj.func.__module__ + '.' + obj.func.__name__
        obj_args, obj_kwargs = obj.args, obj.keywords
    else:
        return None

    # Concatenate arguments
    obj_args = obj_args + args
    obj_kwargs.update(kwargs)

    # Convert object components to string representation
    args = ",".join(map(repr, obj_args))
    kwargs = ",".join("{}={!r}".format(k, v) for k, v in obj_kwargs.items())
    comma = ',' if args != '' and kwargs != '' else ''
    format_string = '{func}({args}{comma}{kwargs})'
    arch = format_string.format(func=func, args=args, comma=comma, kwargs=kwargs).replace(' ', '')

    return arch


def load_model(model_path, name='', device=None, arch=None, return_checkpoint=False, train=False):
    """ Load a model from checkpoint.

    This is a utility function that combines the model weights and architecture (string representation) to easily
    load any model without explicit knowledge of its class.

    Args:
        model_path (str): Path to the model's checkpoint (.pth)
        name (str): The name of the model (for printing and error management)
        device (torch.device): The device to load the model to
        arch (str): The model's architecture (string representation)
        return_checkpoint (bool): If True, the checkpoint will be returned as well
        train (bool): If True, the model will be set to train mode, else it will be set to test mode

    Returns:
        (nn.Module, dict (optional)): A tuple that contains:
            - model (nn.Module): The loaded model
            - checkpoint (dict, optional): The model's checkpoint (only if return_checkpoint is True)
    """
    assert model_path is not None, '%s model must be specified!' % name
    assert os.path.exists(model_path), 'Couldn\'t find %s model in path: %s' % (name, model_path)
    print('=> Loading %s model: "%s"...' % (name, os.path.basename(model_path)))
    checkpoint = torch.load(model_path)
    assert arch is not None or 'arch' in checkpoint, 'Couldn\'t determine %s model architecture!' % name
    arch = checkpoint['arch'] if arch is None else arch
    model = obj_factory(arch)
    if device is not None:
        model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.train(train)

    if return_checkpoint:
        return model, checkpoint
    else:
        return model


def random_pair(n, min_dist=1, index1=None):
    """ Return a random pair of integers in the range [0, n) with a minimum distance between them.

    Args:
        n (int): Determine the range size
        min_dist (int): The minimum distance between the random pair
        index1 (int, optional): If specified, this will determine the first integer

    Returns:
        (int, int): The random pair of integers.
    """
    r1 = random.randint(0, n - 1) if index1 is None else index1
    d_left = min(r1, min_dist)
    d_right = min(n - 1 - r1, min_dist)
    r2 = random.randint(0, n - 2 - d_left - d_right)
    r2 = r2 + d_left + 1 + d_right if r2 >= (r1 - d_left) else r2

    return r1, r2


def random_pair_range(a, b, min_dist=1, index1=None):
    """ Return a random pair of integers in the range [a, b] with a minimum distance between them.

    Args:
        a (int): The minimum number in the range
        b (int): The maximum number in the range
        min_dist (int): The minimum distance between the random pair
        index1 (int, optional): If specified, this will determine the first integer

    Returns:
        (int, int): The random pair of integers.
    """
    r1 = random.randint(a, b) if index1 is None else index1
    d_left = min(r1 - a, min_dist)
    d_right = min(b - r1, min_dist)
    r2 = random.randint(a, b - 1 - d_left - d_right)
    r2 = r2 + d_left + 1 + d_right if r2 >= (r1 - d_left) else r2

    return r1, r2


# Adapted from: https://github.com/Sudy/coling2018/blob/master/torchtext/utils.py
def download_from_url(url, output_path):
    """ Download file from url including Google Drive.

    Args:
        url (str): File URL
        output_path (str): Output path to write the file to
    """
    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        with open(output_path, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=1, desc=os.path.split(output_path)[1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        process_response(response)
        return

    # print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)


def main():
    from torch.optim.lr_scheduler import StepLR
    scheduler = partial(StepLR, step_size=10, gamma=0.5)
    print(get_arch(scheduler))
    scheduler = partial(StepLR, 10, 0.5)
    print(get_arch(scheduler))
    scheduler = partial(StepLR, 10, gamma=0.5)
    print(get_arch(scheduler))
    scheduler = partial(StepLR)
    print(get_arch(scheduler))
    print(get_arch(scheduler, 10, gamma=0.5))

    scheduler = 'torch.optim.lr_scheduler.StepLR(step_size=10,gamma=0.5)'
    print(get_arch(scheduler))
    scheduler = 'torch.optim.lr_scheduler.StepLR(10,0.5)'
    print(get_arch(scheduler))
    scheduler = 'torch.optim.lr_scheduler.StepLR(10,gamma=0.5)'
    print(get_arch(scheduler))
    scheduler = 'torch.optim.lr_scheduler.StepLR()'
    print(get_arch(scheduler))
    print(get_arch(scheduler, 10, gamma=0.5))


if __name__ == "__main__":
    main()
