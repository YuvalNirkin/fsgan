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
import fsgan.utils.estimate_pose as estimate_pose


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


def main(source_path, target_path, frontal_path='frontal.jpg',
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         model_path='../../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, crop_size=256, display=False):
    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)
    device, gpus = utils.set_device()
    G = obj_factory(arch).to(device)
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['state_dict'])
    G.train(False)

    # Initialize transformations
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    # Process source image
    print('Processing source image...')
    source_bgr = cv2.imread(source_path)
    source_rgb = source_bgr[:, :, ::-1]
    source_landmarks, source_bbox = process_image(fa, source_rgb, crop_size)
    # source_landmarks = source_landmarks[:, :2]
    if source_bbox is None:
        raise RuntimeError("Couldn't detect a face in source image: " + source_path)
    source_tensor, source_landmarks, source_bbox = img_transforms1(source_rgb, source_landmarks, source_bbox)
    source_cropped_bgr = tensor2bgr(source_tensor[0])

    # Process target image
    print('Processing target image...')
    target_bgr = cv2.imread(target_path)
    target_rgb = target_bgr[:, :, ::-1]
    target_landmarks, target_bbox = process_image(fa, target_rgb, crop_size)
    # source_landmarks = source_landmarks[:, :2]
    if target_bbox is None:
        raise RuntimeError("Couldn't detect a face in target image: " + target_path)
    target_tensor, target_landmarks, target_bbox = img_transforms1(target_rgb, target_landmarks, target_bbox)
    target_cropped_bgr = tensor2bgr(target_tensor[0])

    # Process frontal image
    print('Processing frontal image...')
    frontal_bgr = cv2.imread(frontal_path)
    frontal_rgb = frontal_bgr[:, :, ::-1]
    frontal_landmarks, frontal_bbox = process_image(fa, frontal_rgb, crop_size)
    # source_landmarks = source_landmarks[:, :2]
    if frontal_bbox is None:
        raise RuntimeError("Couldn't detect a face in frontal image: " + frontal_path)
    frontal_tensor, frontal_landmarks, frontal_bbox = img_transforms1(frontal_rgb, frontal_landmarks, frontal_bbox)

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            output_filename = os.path.splitext(os.path.basename(source_path))[0] + '_' + \
                              os.path.splitext(os.path.basename(target_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = 10.0
        out_vid = cv2.VideoWriter(output_path, fourcc, fps,
                                  (source_cropped_bgr.shape[1] * 3, source_cropped_bgr.shape[0]))
    else:
        out_vid = None

    # source_cropped_bgr = tensor2bgr(source_tensor[0] if isinstance(source_tensor, list) else source_tensor)
    # for i in range(len(source_tensor)):
    #     source_tensor[i] = source_tensor[i].to(device)

    # Estimate source euler angles
    P1 = estimate_pose.compute_similarity_transform(source_landmarks[0].numpy(), frontal_landmarks[0].numpy())
    s1, R1, t1 = estimate_pose.P2sRt(P1)
    euler1 = estimate_pose.matrix2angle(R1)    # Yaw, Pitch, Roll
    print('source euler:')
    print(np.array(euler1) * 180.0 / np.pi)

    # Estimate target euler angles
    P2 = estimate_pose.compute_similarity_transform(target_landmarks[0].numpy(), frontal_landmarks[0].numpy())
    s2, R2, t2 = estimate_pose.P2sRt(P2)
    euler2 = estimate_pose.matrix2angle(R2)    # Yaw, Pitch, Roll
    print('target euler:')
    print(np.array(euler2) * 180.0 / np.pi)

    # Estimate source to target transform
    # P3 = estimate_pose.compute_similarity_transform(target_landmarks[0].numpy(), source_landmarks[0].numpy())
    # s3, R3, t3 = estimate_pose.P2sRt(P3)
    # euler3 = estimate_pose.matrix2angle(R2)  # Yaw, Pitch, Roll

    R3, t3 = estimate_pose.rigid_transform_3D(source_landmarks[0].numpy(), target_landmarks[0].numpy())
    euler3 = np.array(estimate_pose.matrix2angle(R3))  # Yaw, Pitch, Roll

    # Interpolate transform
    # alpha = 0.5
    # euler3 = np.array(euler3) * alpha
    # R3i = estimate_pose.euler2mat([euler3[2], -euler3[0], euler3[1]])

    # Compose projection matrix
    # print(R1)
    # R2_composed = estimate_pose.euler2mat([euler2[2], -euler2[0], euler2[1]])

    # Transform source points to target points
    # A2 = (ret_R * A.T) + tile(ret_t, (1, n))
    # A2 = A2.T

    # pts = source_landmarks[0].numpy().transpose()
    # pts_h = np.concatenate((pts, np.ones((1, 68))), axis=0)
    # out_pts = P3 @ pts_h
    # out_pts = out_pts.transpose()

    for alpha in np.arange(0.0, 1.0, 0.01):
        # Interpolate
        # alpha = 0.5
        euler = np.array(euler3) * alpha
        t = t3 * alpha
        R = estimate_pose.euler2mat([euler[2], -euler[0], euler[1]])

        pts = source_landmarks[0].numpy().transpose()
        out_pts = R @ pts
        translation = np.tile(t, (out_pts.shape[1], 1)).transpose()
        # out_pts = out_pts[:2, :] + translation
        out_pts += translation
        out_pts = out_pts.transpose()

        # Render landmarks
        source_render = plot_kpt(source_cropped_bgr, source_landmarks[0].numpy())
        # interp_render = plot_kpt(source_cropped_bgr, out_pts)
        interp_render = plot_kpt(np.ones_like(source_cropped_bgr) * 255, out_pts, line_color=(0, 0, 0),
                                 line_thickness=1)
        # cv2.imshow('render', source_render)
        target_render = plot_kpt(target_cropped_bgr, target_landmarks[0].numpy())
        # cv2.imshow('target', target_render)
        render_img = np.concatenate((source_render, interp_render, target_render), axis=1)
        if out_vid is not None:
            out_vid.write(render_img)
        if out_vid is None or display:
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Face Reenactment')
    parser.add_argument('source', metavar='IMAGE',
                        help='path to source image')
    parser.add_argument('-t', '--target', metavar='IMAGE',
                        help='paths to target video')
    parser.add_argument('-f', '--frontal', default='frontal.jpg', metavar='IMAGE',
                        help='paths to frontal image')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-m', '--model', default='../../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
                        metavar='PATH', help='path to face reenactment model')
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
    main(args.source, args.target, args.frontal, args.arch, args.model, args.pil_transforms1, args.pil_transforms2,
         args.tensor_transforms1, args.tensor_transforms2, args.output, args.crop_size, args.display)
