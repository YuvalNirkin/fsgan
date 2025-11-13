""" Utility script for downloading the FSGAN models.

This script should be placed in the root directory of the FSGAN repository.
"""

import os
import argparse
from fsgan.utils.utils import download_from_url
import logging
import traceback


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--models', metavar='STR', default='all', choices=['all', 'v1', 'v2'],
                    help="models to download: 'v1' - v1 models only, 'v2' - v2 models only, 'all' - all models")
parser.add_argument('-o', '--output', default='weights', metavar='DIR',
                    help='output directory')


model_links_v1 = {
    'https://drive.google.com/uc?export=download&id=1uyEETpnuvfwZeQLvlP8SzahrMBwst9Be':
        'ijbc_msrunet_256_2_0_reenactment_v1.pth',
    'https://drive.google.com/uc?export=download&id=1lrdW4daM63FEERwzjkl7hB3hHOpwEvN_':
        'ijbc_msrunet_256_1_2_reenactment_stepwise_v1.pth',
    'https://drive.google.com/uc?export=download&id=1KHH5VvrlDOGB_znPsolmxLImSezDduR9':
        'ijbc_msrunet_256_2_0_inpainting_v1.pth',
    'https://drive.google.com/uc?export=download&id=1a0wjI4kehLeJsEbH84tH_Q10xi4BuCwI':
        'ijbc_msrunet_256_2_0_blending_v1.pth',
    'https://drive.google.com/uc?export=download&id=1Wrah8rZ36bED2pVLhai932Z9mWZ9ckQh':
        'lfw_figaro_unet_256_2_0_segmentation_v1.pth',
    'https://github.com/YuvalNirkin/fsgan/releases/download/v1.0.1/vggface2_vgg19_256_2_0_id.pth': None,
    'https://github.com/YuvalNirkin/fsgan/releases/download/v1.0.1/celeba_vgg19_256_2_0_28_attr.pth': None,
    'https://github.com/YuvalNirkin/fsgan/releases/download/v1.0.1/hopenet_robust_alpha1.pth': None
}

model_links_v2 = {
    'https://drive.google.com/uc?export=download&id=1sK0cxnszgmFohatmXnl8fiAS1zCQxM9_':
        'nfv_msrunet_256_1_2_reenactment_v2.1.pth',
    'https://drive.google.com/uc?export=download&id=1o3XkwpEEMuo8wvIXwEB19OlSvGYsQq2E':
        'ijbc_msrunet_256_1_2_inpainting_v2.pth',
    'https://drive.google.com/uc?export=download&id=17M6RjE5bgSbeRKz0ZA04ofIPMg2Z3qnd':
        'ijbc_msrunet_256_1_2_blending_v2.pth',
    'https://drive.google.com/uc?export=download&id=1cPfZ89LtzxvGTsOZw66ghcubBDawTAG5':
        'celeba_unet_256_1_2_segmentation_v2.pth',
    'https://drive.google.com/uc?export=download&id=1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-':
        'WIDERFace_DSFD_RES152.pth',
    'https://github.com/YuvalNirkin/fsgan/releases/download/v2.0.0/hr18_wflw_landmarks.pth': None,
    'https://github.com/YuvalNirkin/fsgan/releases/download/v2.0.0/vggface2_vgg19_256_1_2_id.pth': None,
    'https://github.com/YuvalNirkin/fsgan/releases/download/v1.0.1/celeba_vgg19_256_2_0_28_attr.pth': None,
    'https://github.com/YuvalNirkin/fsgan/releases/download/v1.0.1/hopenet_robust_alpha1.pth': None
}


def main(models='all', output='weights'):
    # Set model links to download
    if models == 'v1':
        model_links = model_links_v1
    elif models == 'v2':
        model_links = model_links_v2
    elif models == 'all':
        model_links = {**model_links_v1, **model_links_v2}
    else:
        raise RuntimeError(f'Unknown models string: "{models}"')

    # Make sure the output directory exists
    os.makedirs(output, exist_ok=True)

    # For each mode link
    for i, (link, filename) in enumerate(model_links.items()):
        filename = os.path.split(link)[1] if filename is None else filename
        out_path = os.path.join(output, filename)
        if os.path.isfile(out_path):
            print('[%d/%d] Skipping "%s"' % (i + 1, len(model_links), filename))
            continue
        print('[%d/%d] Downloading "%s"...' % (i + 1, len(model_links), filename))
        try:
            download_from_url(link, out_path)
        except Exception as e:
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
