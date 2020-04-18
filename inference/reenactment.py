""" Image to video face reenactment. """

import os
import face_alignment
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as F
from fsgan.data.landmark_transforms import crop_img, scale_bbox, Resize, generate_heatmaps
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_from_video
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

    # scaled_bbox = scale_bbox(bbox)
    # cropped_img, cropped_landmarks = crop_img(img, landmarks, scaled_bbox)
    # landmarks_resize = Resize(size)
    # cropped_img, cropped_landmarks, scaled_bbox = \
    #     landmarks_resize(Image.fromarray(cropped_img), cropped_landmarks, scaled_bbox)
    #
    # return np.array(cropped_img), cropped_landmarks


def process_cached_frame(frame, landmarks, bbox, size=128):
    scaled_bbox = scale_bbox(bbox)
    cropped_frame, cropped_landmarks = crop_img(frame, landmarks, scaled_bbox)
    landmarks_resize = Resize(size)
    cropped_frame, cropped_landmarks, scaled_bbox = \
        landmarks_resize(Image.fromarray(cropped_frame), cropped_landmarks, scaled_bbox)

    return np.array(cropped_frame), cropped_landmarks


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
    output_img = unnormalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img


def prepare_generator_input(img, landmarks, sigma=2):
    landmarks = generate_heatmaps(img.shape[1], img.shape[0], landmarks, sigma=sigma)
    landmarks = torch.from_numpy(landmarks)
    img = F.normalize(F.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return img, landmarks


def main(source_path, target_path,
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         model_path='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         pose_model_path='../weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, crop_size=256, display=False):
    torch.set_grad_enabled(False)

    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    device, gpus = utils.set_device()
    G = obj_factory(arch).to(device)
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['state_dict'])
    G.train(False)

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

    # Process source image
    source_bgr = cv2.imread(source_path)
    source_rgb = source_bgr[:, :, ::-1]
    source_landmarks, source_bbox = process_image(fa, source_rgb, crop_size)
    if source_bbox is None:
        raise RuntimeError("Couldn't detect a face in source image: " + source_path)
    source_tensor, source_landmarks, source_bbox = img_transforms1(source_rgb, source_landmarks, source_bbox)
    source_cropped_bgr = tensor2bgr(source_tensor[0] if isinstance(source_tensor, list) else source_tensor)
    for i in range(len(source_tensor)):
        source_tensor[i] = source_tensor[i].to(device)

    # Extract landmarks and bounding boxes from target video
    frame_indices, landmarks, bboxes, eulers = extract_landmarks_bboxes_euler_from_video(target_path, Gp, device=device)
    if frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the target video: ' + target_path)

    # Open target video file
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + target_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            output_filename = os.path.splitext(os.path.basename(source_path))[0] + '_' + \
                              os.path.splitext(os.path.basename(target_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_vid = cv2.VideoWriter(output_path, fourcc, fps,
                                  (source_cropped_bgr.shape[1]*3, source_cropped_bgr.shape[0]))
    else:
        out_vid = None

    # For each frame in the target video
    valid_frame_ind = 0
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            continue
        if i not in frame_indices:
            continue
        frame_rgb = frame[:, :, ::-1]
        frame_tensor, frame_landmarks, frame_bbox = img_transforms2(frame_rgb, landmarks[valid_frame_ind],
                                                                    bboxes[valid_frame_ind])
        valid_frame_ind += 1

        # frame_cropped_rgb, frame_landmarks = process_cached_frame(frame_rgb, landmarks[valid_frame_ind],
        #                                                           bboxes[valid_frame_ind], size)
        # frame_cropped_bgr = frame_cropped_rgb[:, :, ::-1].copy()
        # valid_frame_ind += 1

        #
        # frame_tensor, frame_landmarks_tensor = prepare_generator_input(frame_cropped_rgb, frame_landmarks)
        # frame_landmarks_tensor.to(device)
        input_tensor = []
        for j in range(len(source_tensor)):
            frame_landmarks[j] = frame_landmarks[j].to(device)
            input_tensor.append(torch.cat((source_tensor[j], frame_landmarks[j]), dim=0).unsqueeze(0).to(device))
        out_img_tensor, out_seg_tensor = G(input_tensor)

        # Transfer image1 mask to image2
        # face_mask_tensor = out_seg_tensor.argmax(1) == 1  # face
        # face_mask_tensor = out_seg_tensor.argmax(1) == 2    # hair
        # face_mask_tensor = out_seg_tensor.argmax(1) >= 1  # head

        # target_img_tensor = frame_tensor[0].view(1, frame_tensor[0].shape[0],
        #                                          frame_tensor[0].shape[1], frame_tensor[0].shape[2]).to(device)

        # Convert back to numpy images
        out_img_bgr = tensor2bgr(out_img_tensor)
        frame_cropped_bgr = tensor2bgr(frame_tensor[0])

        # Render
        # for point in np.round(frame_landmarks).astype(int):
        #     cv2.circle(frame_cropped_bgr, (point[0], point[1]), 2, (0, 0, 255), -1)
        render_img = np.concatenate((source_cropped_bgr, out_img_bgr, frame_cropped_bgr), axis=1)
        if out_vid is not None:
            out_vid.write(render_img)
        if out_vid is None or display:
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # for point in np.round(source_landmarks).astype(int):
    #     cv2.circle(source_cropped_bgr, (point[0], point[1]), 2, (0, 0, 255), -1)
    # cv2.imshow('frame', source_cropped_bgr)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('reenactment')
    parser.add_argument('source', metavar='IMAGE',
                        help='path to source image')
    parser.add_argument('-t', '--target', type=str, metavar='VIDEO',
                        help='paths to target video')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-m', '--model', default='../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth', metavar='PATH',
                        help='path to face reenactment model')
    parser.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pth', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt1', '--pil_transforms1', nargs='+', help='first PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-pt2', '--pil_transforms2', nargs='+', help='second PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'))
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
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()
    main(args.source, args.target, args.arch, args.model, args.pose_model, args.pil_transforms1, args.pil_transforms2,
         args.tensor_transforms1, args.tensor_transforms2, args.output, args.crop_size, args.display)
