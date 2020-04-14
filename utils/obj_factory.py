import os
import importlib
from functools import partial


KNOWN_MODULES = {
    # datasets
    'image_list_dataset': 'fsgan.datasets.image_list_dataset',
    'opencv_video_seq_dataset': 'fsgan.datasets.opencv_video_seq_dataset',
    'seq_dataset': 'fsgan.datasets.seq_dataset',
    'img_landmarks_transforms': 'fsgan.datasets.img_landmarks_transforms',
    'img_lms_pose_transforms': 'fsgan.datasets.img_lms_pose_transforms',
    'transforms': 'torchvision.transforms',

    # models
    'res_unet': 'fsgan.models.res_unet',
    'res_unet_split': 'fsgan.models.res_unet_split',
    'res_unet_msba': 'fsgan.models.res_unet_msba',

    # criterions
    'vgg_loss': 'fsgan.criterions.vgg_loss',
    'gan_loss': 'fsgan.criterions.gan_loss',

    # Torch
    'nn': 'torch.nn',
    'optim': 'torch.optim',
    'lr_scheduler': 'torch.optim.lr_scheduler',
}


def extract_args(*args, **kwargs):
    return args, kwargs


def obj_factory(obj_exp, *args, **kwargs):
    """ Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return obj_exp(*args, **kwargs)
    if not isinstance(obj_exp, str):
        return obj_exp

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name)
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def partial_obj_factory(obj_exp, *args, **kwargs):
    """ Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [partial_obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return partial(obj_exp.func, *(obj_exp.args + args), **{**obj_exp.keywords, **kwargs})
    if not isinstance(obj_exp, str):
        return partial(obj_exp)

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name)
    module_class = getattr(module, class_name)

    return partial(module_class, *args, **kwargs)


def main(obj_exp):
    # obj = obj_factory(obj_exp)
    # print(obj)

    import inspect
    partial_obj = partial_obj_factory(obj_exp)
    print(f'is obj_exp a class = {inspect.isclass(partial_obj.func)}')
    print(partial_obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('utils test')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)
