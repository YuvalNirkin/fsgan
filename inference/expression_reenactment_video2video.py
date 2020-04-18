""" Expression only reenactment. """

import os
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.seg_utils import blend_seg_pred
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_3d_from_video
from fsgan.models.hopenet import Hopenet
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.utils.estimate_pose import rigid_transform_3D, matrix2angle, euler2mat


def process_image(fa, img, size=256):
    detected_faces = fa.face_detector.detect_from_image(img.copy())
    if len(detected_faces) != 1:
        return None, None

    preds = fa.get_landmarks(img, detected_faces)
    landmarks = preds[0]
    bbox = detected_faces[0][:4]

    # Convert bounding boxes format from [min, max] to [min, size]
    bbox[2:] = bbox[2:] - bbox[:2] + 1

    return landmarks, bbox


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img


def transfer_mask(img1, img2, mask):
    mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 3, 1, 1).float()
    out = img1 * mask + img2 * (1 - mask)

    return out


def create_pyramid(img, n=1):
    # If input is a list or tuple return it as it is (probably already a pyramid)
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(torch.nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt, circle_color=(0, 0, 255), line_color=(255, 255, 255), line_thickness=1):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, circle_color, 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), line_color, line_thickness)

    return image


def main(source_path, target_path,
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
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
         output_path=None, crop_size=256, reverse_output=False, verbose=0, output_crop=False, display=False):
    torch.set_grad_enabled(False)
    device, gpus = utils.set_device()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    landmarks2heatmaps = LandmarkHeatmap().to(device)

    # Load face reenactment model
    Gr = obj_factory(arch).to(device)
    checkpoint = torch.load(reenactment_model_path)
    Gr.load_state_dict(checkpoint['state_dict'])
    Gr.train(False)

    # Load face segmentation model
    if seg_model_path is not None:
        print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
        if seg_model_path.endswith('.pth'):
            checkpoint = torch.load(seg_model_path)
            Gs = obj_factory(checkpoint['arch']).to(device)
            Gs.load_state_dict(checkpoint['state_dict'])
        else:
            Gs = torch.jit.load(seg_model_path, map_location=device)
        if Gs is None:
            raise RuntimeError('Failed to load face segmentation model!')
            Gs.eval()
    else:
        Gs = None

    # Load face inpainting model
    if seg_model_path is not None:
        print('Loading face inpainting model: "' + os.path.basename(inpainting_model_path) + '"...')
        if inpainting_model_path.endswith('.pth'):
            checkpoint = torch.load(inpainting_model_path)
            Gi = obj_factory(checkpoint['arch']).to(device)
            Gi.load_state_dict(checkpoint['state_dict'])
        else:
            Gi = torch.jit.load(inpainting_model_path, map_location=device)
        if Gi is None:
            raise RuntimeError('Failed to load face segmentation model!')
        Gi.eval()
    else:
        Gi = None

    # Load face blending model
    print('Loading face blending model: "' + os.path.basename(blend_model_path) + '"...')
    checkpoint = torch.load(blend_model_path)
    Gb = obj_factory(checkpoint['arch']).to(device)
    Gb.load_state_dict(checkpoint['state_dict'])
    Gb.train(False)

    # Initialize pose
    Gp = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    Gp.load_state_dict(checkpoint['state_dict'])
    Gp.train(False)

    # Initialize transformations
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    # Extract landmarks, bounding boxes, and euler angles from source video
    source_frame_indices, source_landmarks, source_bboxes, source_eulers, source_landmarks_3d = \
        extract_landmarks_bboxes_euler_3d_from_video(source_path, Gp, fa, device=device)
    if source_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the source video: ' + source_path)

    # Extract landmarks, bounding boxes, and euler angles from target video
    target_frame_indices, target_landmarks, target_bboxes, target_eulers, target_landmarks_3d = \
        extract_landmarks_bboxes_euler_3d_from_video(target_path, Gp, fa, device=device)
    if target_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the target video: ' + target_path)

    # Open source video file
    source_vid = cv2.VideoCapture(source_path)
    if not source_vid.isOpened():
        raise RuntimeError('Failed to read source video: ' + source_path)

    # Open target video file
    target_vid = cv2.VideoCapture(target_path)
    if not target_vid.isOpened():
        raise RuntimeError('Failed to read video: ' + target_path)
    total_frames = int(target_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = target_vid.get(cv2.CAP_PROP_FPS)

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            if reverse_output:
                output_filename = os.path.splitext(os.path.basename(target_path))[0] + '_' + \
                                  os.path.splitext(os.path.basename(source_path))[0] + '.mp4'
            else:
                output_filename = os.path.splitext(os.path.basename(source_path))[0] + '_' + \
                                  os.path.splitext(os.path.basename(target_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_vid_size = (crop_size, crop_size) if verbose == 0 else (crop_size * 3, crop_size * 2)
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, out_vid_size)
    else:
        out_vid = None

    # For each frame in the target video
    valid_frame_ind = 0
    for i in tqdm(range(total_frames)):
        ret, source_curr_bgr = source_vid.read()
        if source_curr_bgr is None:
            continue
        if i not in source_frame_indices:
            continue
        ret, target_curr_bgr = target_vid.read()

        # Extract source data
        source_curr_rgb = source_curr_bgr[:, :, ::-1]
        source_curr_tensor, source_curr_landmarks, source_curr_bbox = img_transforms1(
            source_curr_rgb, source_landmarks[valid_frame_ind], source_bboxes[valid_frame_ind])
        _, source_curr_landmarks_3d, _ = img_transforms1(
            source_curr_rgb, source_landmarks_3d[valid_frame_ind], source_bboxes[valid_frame_ind])
        source_curr_euler = source_eulers[valid_frame_ind]

        # Extract target data
        if target_curr_bgr is not None:
            target_curr_rgb = target_curr_bgr[:, :, ::-1]
            target_curr_tensor, target_curr_landmarks, target_curr_bbox = img_transforms2(
                target_curr_rgb, target_landmarks[valid_frame_ind], target_bboxes[valid_frame_ind])
            _, target_curr_landmarks_3d, _ = img_transforms2(
                target_curr_rgb, target_landmarks_3d[valid_frame_ind], target_bboxes[valid_frame_ind])
            target_curr_euler = target_eulers[valid_frame_ind]

        valid_frame_ind += 1

        # Transform the target landmarks to the source landmarks
        R, t = rigid_transform_3D(target_curr_landmarks_3d[0].numpy(), source_curr_landmarks_3d[0].numpy())
        euler = np.array(matrix2angle(R))  # Yaw, Pitch, Roll
        # R = euler2mat([euler[2], -euler[0], euler[1]])

        pts = target_curr_landmarks_3d[0].numpy().transpose()
        out_pts = R @ pts
        translation = np.tile(t, (out_pts.shape[1], 1)).transpose()
        out_pts += translation
        out_pts = out_pts.transpose()

        # Transfer mouth points only
        source_landmarks_np = source_curr_landmarks[0].cpu().numpy().copy()
        mouth_pts = out_pts[48:, :2] - out_pts[48:, :2].mean(axis=0) + source_landmarks_np[48:, :].mean(axis=0)
        transformed_landmarks = source_landmarks_np
        transformed_landmarks[48:, :] = mouth_pts

        # # Transfer mouth points only
        # transformed_landmarks = source_curr_landmarks[0].cpu().numpy().copy()
        # transformed_landmarks[48:, :] = out_pts[48:, :2]

        # Create heatmap pyramids
        transformed_landmarks_tensor = torch.from_numpy(transformed_landmarks).unsqueeze(0).to(device)
        transformed_hm_tensor = landmarks2heatmaps(transformed_landmarks_tensor)
        transformed_hm_tensor_pyd = [transformed_hm_tensor]
        transformed_hm_tensor_pyd.append(
            F.interpolate(transformed_hm_tensor, scale_factor=0.5, mode='bilinear', align_corners=False))

        # Face reenactment
        reenactment_input_tensor = []
        for j in range(len(source_curr_tensor)):
            source_curr_tensor[j] = source_curr_tensor[j].unsqueeze(0).to(device)
            # transformed_hm_tensor_pyd[j] = transformed_hm_tensor_pyd[j].to(device)
            reenactment_input_tensor.append(
                torch.cat((source_curr_tensor[j], transformed_hm_tensor_pyd[j]), dim=1))
        reenactment_img_tensor, reenactment_seg_tensor = Gr(reenactment_input_tensor)

        # Transfer reenactment to original image
        source_orig_tensor = source_curr_tensor[0].to(device)
        face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1
        transfer_tensor = transfer_mask(reenactment_img_tensor, source_orig_tensor, face_mask_tensor)

        # Blend the transfer image with the source image
        blend_input_tensor = torch.cat(
            (transfer_tensor, source_orig_tensor, face_mask_tensor.unsqueeze(1).float()), dim=1)
        blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(source_curr_tensor))
        blend_tensor = Gb(blend_input_tensor_pyd)

        # Convert back to numpy images
        source_cropped_bgr = tensor2bgr(source_curr_tensor[0])
        target_cropped_bgr = tensor2bgr(target_curr_tensor[0])
        reenactment_bgr = tensor2bgr(reenactment_img_tensor)
        transfer_bgr = tensor2bgr(transfer_tensor)
        blend_bgr = tensor2bgr(blend_tensor)

        # Render
        if verbose == 0:
            render_img = blend_bgr
        else:
            source_render = plot_kpt(source_cropped_bgr, source_curr_landmarks[0].cpu().numpy())
            transform_render = plot_kpt(source_cropped_bgr, transformed_landmarks)
            target_render = plot_kpt(target_cropped_bgr, target_curr_landmarks[0].cpu().numpy())
            render_img1 = np.concatenate((source_render, transform_render, target_render), axis=1)
            render_img2 = np.concatenate((source_cropped_bgr, blend_bgr, target_cropped_bgr), axis=1)
            render_img = np.concatenate((render_img1, render_img2), axis=0)
        if out_vid is not None:
            out_vid.write(render_img)
        if out_vid is None or display:
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('expression_reenactment_video2video')
    parser.add_argument('source', metavar='VIDEO',
                        help='path to source video')
    parser.add_argument('-t', '--target', metavar='VIDEO',
                        help='paths to target video')
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
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output video path')
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
    main(args.source, args.target, args.arch, args.reenactment_model, args.seg_model, args.inpainting_model,
         args.blending_model, args.pose_model, args.pil_transforms1, args.pil_transforms2, args.tensor_transforms1,
         args.tensor_transforms2, args.output, args.crop_size, args.reverse_output, args.verbose, args.output_crop,
         args.display)
