""" Video segmentation. """

import os
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.seg_utils import blend_seg_pred
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_from_video
from fsgan.utils.bbox_utils import scale_bbox
from fsgan.models.hopenet import Hopenet


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


def crop2img(img, crop, bbox):
    scaled_bbox = scale_bbox(bbox)
    scaled_crop = cv2.resize(crop, (scaled_bbox[3], scaled_bbox[2]), interpolation=cv2.INTER_CUBIC)
    left = -scaled_bbox[0] if scaled_bbox[0] < 0 else 0
    top = -scaled_bbox[1] if scaled_bbox[1] < 0 else 0
    right = scaled_bbox[0] + scaled_bbox[2] - img.shape[1] if (scaled_bbox[0] + scaled_bbox[2] - img.shape[1]) > 0 else 0
    bottom = scaled_bbox[1] + scaled_bbox[3] - img.shape[0] if (scaled_bbox[1] + scaled_bbox[3] - img.shape[0]) > 0 else 0
    crop_bbox = np.array([left, top, scaled_bbox[2] - left - right, scaled_bbox[3] - top - bottom])
    scaled_bbox += np.array([left, top, -left - right, -top - bottom])

    out_img = img.copy()
    out_img[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]] = \
        scaled_crop[crop_bbox[1]:crop_bbox[1] + crop_bbox[3], crop_bbox[0]:crop_bbox[0] + crop_bbox[2]]

    return out_img


def main(source_path, output_path, seg_model_path='../weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
         pose_model_path='../weights/hopenet_robust_alpha1.pkl',
         pil_transforms=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)'),
         tensor_transforms=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         crop_size=256, verbose=0, output_crop=False, display=False, draw_bbox=0, start_time=0.0, start_interp=0.0,
         end_time=-1.0, end_interp=0.0):
    torch.set_grad_enabled(False)

    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    device, gpus = utils.set_device()

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

    # Initialize pose
    Gp = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    Gp.load_state_dict(checkpoint)
    Gp.train(False)

    # Initialize transformations
    pil_transforms = obj_factory(pil_transforms) if pil_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = landmark_transforms.ComposePyramids(pil_transforms + tensor_transforms)

    # Extract landmarks, bounding boxes, and euler angles from source video
    source_frame_indices, source_landmarks, source_bboxes, source_eulers = \
        extract_landmarks_bboxes_euler_from_video(source_path, Gp, fa, device=device)
    if source_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the source video: ' + source_path)

    # Open source video file
    source_vid = cv2.VideoCapture(source_path)
    if not source_vid.isOpened():
        raise RuntimeError('Failed to read source video: ' + source_path)
    total_frames = int(source_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = source_vid.get(cv2.CAP_PROP_FPS)
    source_vid_width = int(source_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_vid_height = int(source_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_time = total_frames / fps

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            output_filename = os.path.splitext(os.path.basename(source_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        if output_crop:
            out_vid_size = (crop_size, crop_size)
        else:
            out_vid_size = (source_vid_width, source_vid_height)
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, out_vid_size)
    else:
        out_vid = None

    # For each frame in the target video
    valid_frame_ind = 0
    for i in tqdm(range(total_frames)):
        ret, frame = source_vid.read()
        if frame is None:
            continue
        if i not in source_frame_indices:
            continue
        frame_rgb = frame[:, :, ::-1]
        frame_tensor, frame_landmarks, frame_bbox = img_transforms(frame_rgb, source_landmarks[valid_frame_ind],
                                                                   source_bboxes[valid_frame_ind])
        # frame_euler = source_eulers[valid_frame_ind]
        valid_frame_ind += 1

        # Segment target image
        source_img_tensor = frame_tensor.unsqueeze(0).to(device)
        source_seg_pred_tensor = Gs(source_img_tensor)
        # target_mask_tensor = source_seg_pred_tensor.argmax(1) == 1

        # Convert back to numpy images
        source_seg_bgr = tensor2bgr(blend_seg_pred(source_img_tensor, source_seg_pred_tensor))
        if output_crop:
            frame = tensor2bgr(source_img_tensor)

        # Render
        render_img = source_seg_bgr if output_crop else crop2img(frame, source_seg_bgr, frame_bbox.numpy())
        if draw_bbox > 0:
            scaled_bbox = scale_bbox(frame_bbox.numpy())
            cv2.rectangle(render_img, tuple(scaled_bbox[:2]), tuple(scaled_bbox[:2] + scaled_bbox[2:]), (0, 0, 255),
                          draw_bbox)

        # Mask background or face
        if False:
            mask_face = False
            background_value = -1.0
            mask = (source_seg_pred_tensor.argmax(1) == 1).unsqueeze(1)
            mask = mask if mask_face else ~mask
            source_img_tensor.masked_fill_(mask, background_value)
        # render_img = tensor2bgr(source_img_tensor)

        # Effects
        # curr_time = i / fps
        # start_interp_time = start_time + start_interp
        # min_start_time = min(start_time, start_interp_time)
        # max_start_time = max(start_time, start_interp_time)
        # end_interp_time = end_time + end_interp
        # min_end_time = min(end_time, end_interp_time)
        # max_end_time = max(end_time, end_interp_time)
        #
        # if curr_time < min_start_time or max_end_time < curr_time:
        #     render_img = frame
        # elif min_start_time <= curr_time <= max_start_time:
        #     alpha = (curr_time - min_start_time) / (max_start_time - min_start_time)
        #     render_img = frame * (1 - alpha) + render_img * alpha
        #     render_img = np.round(render_img).astype('uint8')
        # elif min_end_time <= curr_time <= max_end_time:
        #     alpha = (curr_time - min_end_time) / (max_end_time - min_end_time)
        #     render_img = render_img * (1 - alpha) + frame * alpha
        #     render_img = np.round(render_img).astype('uint8')

        # Write and draw
        if out_vid is not None:
            out_vid.write(render_img)
        if out_vid is None or display:
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('segmentation_video')
    parser.add_argument('source', metavar='VIDEO',
                        help='path to source video')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output video path')
    parser.add_argument('-sm', '--seg_model', default='../weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
                        metavar='PATH', help='path to face segmentation model')
    parser.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pkl', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt', '--pil_transforms', nargs='+',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)'),
                        help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-cs', '--crop_size', default=256, type=int, metavar='N',
                        help='crop size of the images')
    parser.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-oc', '--output_crop', action='store_true',
                        help='output crop around the face')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    parser.add_argument('-db', '--draw_bbox', default=0, type=int, metavar='N',
                        help='draw bounding box (=0: no, >0: thickness in pixels')
    parser.add_argument('-st', '--start_time', default=0.0, type=float, metavar='F',
                        help='start time [seconds]')
    parser.add_argument('-si', '--start_interp', default=0.0, type=float, metavar='F',
                        help='interpolation time relative to start time [seconds]')
    parser.add_argument('-et', '--end_time', default=-1.0, type=float, metavar='F',
                        help='start time [seconds]')
    parser.add_argument('-ei', '--end_interp', default=0.0, type=float, metavar='F',
                        help='interpolation time relative to end time [seconds]')
    args = parser.parse_args()
    main(args.source, args.output, args.seg_model, args.pose_model, args.pil_transforms, args.tensor_transforms,
         args.crop_size, args.verbose, args.output_crop, args.display, args.draw_bbox,
         args.start_time, args.start_interp, args.end_time, args.end_interp)
