""" Image to video stepwise face reenactment. """

import os
from math import cos, sin, atan2, asin
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm
import torch
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.img_utils import create_pyramid
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_3d_from_video
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.models.hopenet import Hopenet


def main(source_path, target_path,
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         model_path='../weights/ijbc_msrunet_256_1_2_reenactment_stepwise_v1.pth',
         pose_model_path='../weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop(bbox_scale=1.2)', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop(bbox_scale=1.2)', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, crop_size=256, display=False):
    torch.set_grad_enabled(False)

    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)
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

    # Initialize landmarks to heatmaps
    landmarks2heatmaps = [LandmarkHeatmap(kernel_size=13, size=(256, 256)).to(device),
                          LandmarkHeatmap(kernel_size=7, size=(128, 128)).to(device)]

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
        source_tensor[i] = source_tensor[i].unsqueeze(0).to(device)

    # Extract landmarks, bounding boxes, euler angles, and 3D landmarks from target video
    frame_indices, landmarks, bboxes, eulers, landmarks_3d = \
        extract_landmarks_bboxes_euler_3d_from_video(target_path, Gp, fa, device=device)
    if frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the target video: ' + target_path)

    # Open target video file
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read target video: ' + target_path)
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
        ret, target_bgr = cap.read()
        if target_bgr is None:
            continue
        if i not in frame_indices:
            continue
        target_rgb = target_bgr[:, :, ::-1]
        target_tensor, target_landmarks, target_bbox = img_transforms2(target_rgb, landmarks_3d[valid_frame_ind],
                                                                       bboxes[valid_frame_ind])
        target_euler = eulers[valid_frame_ind]
        valid_frame_ind += 1

        # TODO: Calculate the number of required reenactment iterations
        reenactment_iterations = 2

        # Generate landmarks sequence
        target_landmarks_sequence = []
        for ri in range(1, reenactment_iterations):
            interp_landmarks = []
            for j in range(len(source_tensor)):
                alpha = float(ri) / reenactment_iterations
                curr_interp_landmarks_np = interpolate_points(source_landmarks[j].cpu().numpy(),
                                                              target_landmarks[j].cpu().numpy(), alpha=alpha)
                interp_landmarks.append(torch.from_numpy(curr_interp_landmarks_np))
            target_landmarks_sequence.append(interp_landmarks)
        target_landmarks_sequence.append(target_landmarks)

        # Iterative reenactment
        out_img_tensor = source_tensor
        for curr_target_landmarks in target_landmarks_sequence:
            out_img_tensor = create_pyramid(out_img_tensor, 2)
            input_tensor = []
            for j in range(len(out_img_tensor)):
                curr_target_landmarks[j] = curr_target_landmarks[j].unsqueeze(0).to(device)
                curr_target_landmarks[j] = landmarks2heatmaps[j](curr_target_landmarks[j])
                input_tensor.append(torch.cat((out_img_tensor[j], curr_target_landmarks[j]), dim=1))
            out_img_tensor, out_seg_tensor = G(input_tensor)

        # Convert back to numpy images
        out_img_bgr = tensor2bgr(out_img_tensor)
        frame_cropped_bgr = tensor2bgr(target_tensor[0])

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


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def rigid_transform_3d(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print
        "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t


def euler2mat(angles):
    X = np.eye(3)
    Y = np.eye(3)
    Z = np.eye(3)

    x = angles[2]
    y = angles[1]
    z = angles[0]

    X[1, 1] = cos(x)
    X[1, 2] = -sin(x)
    X[2, 1] = sin(x)
    X[2, 2] = cos(x)

    Y[0, 0] = cos(y)
    Y[0, 2] = sin(y)
    Y[2, 0] = -sin(y)
    Y[2, 2] = cos(y)

    Z[0, 0] = cos(z)
    Z[0, 1] = -sin(z)
    Z[1, 0] = sin(z)
    Z[1, 1] = cos(z)

    R = Z @ Y @ X

    return R


def interpolate_points(points1, points2, alpha=0.5):
    R, t = rigid_transform_3d(points1, points2)
    euler = np.array(matrix2angle(R))  # Yaw, Pitch, Roll

    # Interpolate
    euler = euler * alpha
    t = t * alpha
    R = euler2mat([euler[2], -euler[0], euler[1]])

    out_pts = points1.transpose()
    out_pts = R @ out_pts
    translation = np.tile(t, (out_pts.shape[1], 1)).transpose()
    out_pts += translation
    out_pts = out_pts.transpose()

    return out_pts


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('reenactment_stepwise')
    parser.add_argument('source', metavar='IMAGE',
                        help='path to source image')
    parser.add_argument('-t', '--target', type=str, metavar='VIDEO',
                        help='paths to target video')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-m', '--model', default='../weights/ijbc_msrunet_256_1_2_reenactment_stepwise_v1.pth',
                        metavar='PATH', help='path to face reenactment model')
    parser.add_argument('-pm', '--pose_model', default='../weights/hopenet_robust_alpha1.pth', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt1', '--pil_transforms1', nargs='+', help='first PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop(bbox_scale=1.2)', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-pt2', '--pil_transforms2', nargs='+', help='second PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop(bbox_scale=1.2)', 'landmark_transforms.Resize(256)',
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
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()
    main(args.source, args.target, args.arch, args.model, args.pose_model, args.pil_transforms1, args.pil_transforms2,
         args.tensor_transforms1, args.tensor_transforms2, args.output, args.crop_size, args.display)
