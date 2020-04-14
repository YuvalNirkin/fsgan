import os
import importlib
from functools import partial


known_modules = {
    # criterions
    'vgg_loss': 'fsgan.criterions.vgg_loss',
    'gan_loss': 'fsgan.criterions.gan_loss',

    # data
    'lfw_datasets': 'fsgan.data.lfw_datasets',
    'domain_dataset': 'fsgan.data.domain_dataset',
    'generic_face_dataset': 'fsgan.data.generic_face_dataset',
    'image_list_dataset': 'fsgan.data.image_list_dataset',
    'face_list_dataset': 'fsgan.data.face_list_dataset',
    'face_landmarks_dataset': 'fsgan.data.face_landmarks_dataset',
    'seg_landmarks_dataset': 'fsgan.data.seg_landmarks_dataset',
    'landmark_transforms': 'fsgan.data.landmark_transforms',
    'seg_landmark_transforms': 'fsgan.data.seg_landmark_transforms',

    # models
    'pg_clipped_enc_dec': 'fsgan.models.pg_clipped_enc_dec',
    'pg_sep_unet': 'fsgan.models.pg_sep_unet',
    'pg_enc_dec': 'fsgan.models.pg_enc_dec',
    'unet': 'fsgan.models.unet',
    'res_unet': 'fsgan.models.res_unet',
    'res_unet_split': 'fsgan.models.res_unet_split',
    'res_unet_mask': 'fsgan.models.res_unet_mask',
    'resnet': 'fsgan.models.resnet',
    'classifiers': 'fsgan.models.classifiers',
    'decoders': 'fsgan.models.decoders',
    'discriminators': 'fsgan.models.discriminators',
    'discriminators_pix2pix': 'fsgan.models.discriminators_pix2pix',
    'generators': 'fsgan.models.generators',
    'vgg': 'fsgan.models.vgg',

    # Torch
    'nn': 'torch.nn',
    'optim': 'torch.optim',
    'lr_scheduler': 'torch.optim.lr_scheduler',

    # Torchvision
    'datasets': 'torchvision.datasets',
    'transforms': 'torchvision.transforms'
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
    module = importlib.import_module(known_modules[module_name] if module_name in known_modules else module_name)
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def main(obj_exp):
    obj = obj_factory(obj_exp)
    print(obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('utils test')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)
