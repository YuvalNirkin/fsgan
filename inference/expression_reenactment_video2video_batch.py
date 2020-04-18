""" Batch version of expression only reenactment. """

import fsgan.inference.expression_reenactment_video2video as expression_reenactment_video2video
import os
from glob import glob
import traceback
import logging
from itertools import product
import numpy as np


def main(input, out_dir, arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         reenactment_model_path='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         seg_model_path='../weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
         inpainting_model_path='../weights/ijbc_msrunet_256_2_0_inpainting_v1.pth',
         blend_model_path='../weights/ijbc_msrunet_256_2_0_blending_v1.pth',
         pose_model_path='../weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         crop_size=256, reverse_output=False, verbose=0, output_crop=False, display=False):
    # Validate input
    if len(input) > 2:
        raise RuntimeError('input must contain either a single path or two paths!')

    # Process input
    if len(input) == 1:
        if os.path.isdir(input[0]):
            vid_paths = glob(os.path.join(input[0], '*.mp4'))
            pair_paths = list(product(vid_paths, vid_paths))
            pair_paths = [(p1, p2) for p1, p2 in pair_paths if p1 != p2]
        else:
            raise NotImplementedError('Single list file not implemented!')
    elif len(input) == 2:
        if os.path.isfile(input[0]) and os.path.isfile(input[1]):
            raise RuntimeError('Both inputs can\'t be files!')
        if os.path.isdir(input[0]) and os.path.isdir(input[1]):
            src_paths = glob(os.path.join(input[0], '*.mp4'))
            tgt_paths = glob(os.path.join(input[1], '*.mp4'))
            pair_paths = list(product(src_paths, tgt_paths))
            pair_paths = [(p1, p2) for p1, p2 in pair_paths if os.path.basename(p1) != os.path.basename(p2)]
        else:   # A directory and a pairs list file
            root_dir = input[0] if os.path.isdir(input[0]) else input[1]
            pair_list_file = input[0] if os.path.isfile(input[0]) else input[1]
            pairs_rel_paths = np.loadtxt(pair_list_file, str)
            pair_paths = [(os.path.join(root_dir, p1), os.path.join(root_dir, p2)) for p1, p2 in pairs_rel_paths]

    # For each input pair
    for i, (src_vid_path, tgt_vid_path) in enumerate(pair_paths):
        if reverse_output:
            out_vid_name = os.path.splitext(os.path.basename(tgt_vid_path))[0] + '_' + \
                           os.path.splitext(os.path.basename(src_vid_path))[0] + '.mp4'
        else:
            out_vid_name = os.path.splitext(os.path.basename(src_vid_path))[0] + '_' + \
                           os.path.splitext(os.path.basename(tgt_vid_path))[0] + '.mp4'
        out_vid_path = os.path.join(out_dir, out_vid_name)
        if os.path.exists(out_vid_path):
            print('[%d/%d] Skipping "%s"' % (i + 1, len(pair_paths), out_vid_name))
            continue
        else:
            print('[%d/%d] Processing "%s"...' % (i + 1, len(pair_paths), out_vid_name))
            try:
                expression_reenactment_video2video.main(
                    src_vid_path, tgt_vid_path, arch, reenactment_model_path, seg_model_path, inpainting_model_path,
                    blend_model_path, pose_model_path, pil_transforms1, pil_transforms2, tensor_transforms1,
                    tensor_transforms2, out_dir, crop_size, reverse_output, verbose, output_crop, display)
            except Exception as e:
                logging.error(traceback.format_exc())


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('expression_reenactment_video2video_batch')
    parser.add_argument('input', metavar='DIR', nargs='+', help='input path')
    parser.add_argument('-o', '--output', metavar='DIR', required=True, help='output directory')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-rm', '--reenactment_model', default='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
                        metavar='PATH', help='path to face reenactment model')
    parser.add_argument('-sm', '--seg_model', default='../weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
                        metavar='PATH', help='path to face segmentation model')
    parser.add_argument('-im', '--inpainting_model', default='../weights/ijbc_msrunet_256_2_0_inpainting_v1.pth',
                        metavar='PATH', help='path to face inpainting model')
    parser.add_argument('-bm', '--blending_model', default='../weights/ijbc_msrunet_256_2_0_blending_v1.pth',
                        metavar='PATH', help='path to face blending model')
    parser.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pth', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt1', '--pil_transforms1', nargs='+', help='first PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-pt2', '--pil_transforms2', nargs='+', help='second PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-cs', '--crop_size', default=256, type=int, metavar='N',
                        help='crop size of the images')
    parser.add_argument('-ro', '--reverse_output', action='store_true',
                        help='reverse the output name to be <target>_<source>')
    parser.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-oc', '--output_crop', action='store_true',
                        help='output crop around the face')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')

    args = parser.parse_args()
    main(args.input, args.output, args.arch, args.reenactment_model, args.seg_model, args.inpainting_model,
         args.blending_model, args.pose_model, args.pil_transforms1, args.pil_transforms2, args.tensor_transforms1,
         args.tensor_transforms2, args.crop_size, args.reverse_output, args.verbose, args.output_crop, args.display)
