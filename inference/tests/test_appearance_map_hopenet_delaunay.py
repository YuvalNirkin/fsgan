import os
import face_alignment
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as F
from fsgan.data.landmark_transforms import Resize, generate_heatmaps
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.img_utils import rgb2tensor
from fsgan.utils.bbox_utils import scale_bbox, crop_img
from sklearn.neighbors import KDTree
from fsgan.models.hopenet import Hopenet
from fsgan.utils.bbox_utils import get_main_bbox
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


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


def fuse_clusters(points, r=0.5):
    """ Select a single point from each cluster of points.

    The clustering is done using a KD-Tree data structure for querying points by radius.

    Args:
        points (np.array): A set of points of shape (N, 2) to fuse
        r (float): The radius for which to fuse the points

    Returns:
        np.array: The indices of remaining points.
    """
    kdt = cKDTree(points)
    indices = kdt.query_ball_point(points, r=r)

    # Build sorted neightbor list
    neighbors = [(i, l) for i, l in enumerate(indices)]
    neighbors.sort(key=lambda t: len(t[1]), reverse=True)

    # Mark remaining indices
    keep = np.ones(points.shape[0], dtype=bool)
    for i, cluster in neighbors:
        if not keep[i]:
            continue
        for j in cluster:
            if i == j:
                continue
            keep[j] = False

    return np.nonzero(keep)[0]


# fig = plt.figure(figsize=(12, 8))
fig = plt.figure(figsize=(24, 16))
def render_appearance_map(tri, points, query_point=None):
    # fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(12, 8))
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), linewidth=3)
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=12)
    if query_point is not None:
        tri_index = tri.find_simplex(query_point)
        tri_vertices = tri.simplices[tri_index]
        plt.plot(points[tri_vertices, 0], points[tri_vertices, 1], 'yo', markersize=12)
        plt.plot(query_point[0], query_point[1], 'rx', markersize=24, markeredgewidth=4)

    plt.xlim(points[:-4, 0].min() - 0.5, points[:-4, 0].max() + 0.5)
    plt.ylim(points[:-4, 1].min() - 0.5, points[:-4, 1].max() + 0.5)
    # plt.title('Appearance Map', fontsize=40)
    plt.xlabel('Yaw (angles)', fontsize=24)
    plt.ylabel('Pitch (angles)', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.draw()
    fig.canvas.draw()
    # plt.show()

    # grab the pixel buffer and dump it into a numpy array
    img = np.array(fig.canvas.renderer._renderer)
    plt.clf()

    return img[:, :, 2::-1]


def main(source_path, target_path, frontal_landmarks_path='frontal_landmarks_256_2_0.npy',
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         model_path='../../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         pose_model_path='../../weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, min_radius=2.0, crop_size=256, display=False, verbose=0):
    torch.set_grad_enabled(False)
    frontal_landmarks = np.load(frontal_landmarks_path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    device, gpus = utils.set_device()

    # Initialize reenactment model
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

    # Extract landmarks, bounding boxes, and euler angles from source video
    source_frame_indices, source_landmarks, source_bboxes, source_eulers = \
        extract_landmarks_bboxes_euler_from_video(fa, Gp, source_path, frontal_landmarks[0], device=device)
    if source_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the source video: ' + source_path)

    # Extract landmarks, bounding boxes, and euler angles from target video
    target_frame_indices, target_landmarks, target_bboxes, target_eulers = \
        extract_landmarks_bboxes_euler_from_video(fa, Gp, target_path, frontal_landmarks[0], device=device)
    if target_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the target video: ' + target_path)

    # Remove close points
    filtered_indices = fuse_clusters(source_eulers[:, :2], r=min_radius)

    # Build appearance map for source video
    points = source_eulers[filtered_indices, :2]
    limit_points = np.array([[-75., -75.], [-75., 75.], [75., -75.], [75., 75.]])
    points = np.concatenate((points, limit_points))
    tri = Delaunay(points)

    kdt = KDTree(source_eulers, metric='euclidean') # For comparison with nearest neighbor approach

    ### Debug ###
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()
    #############

    ### Debug ###
    # img = render_appearance_map(tri, points, np.array([0, 0]))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    #############

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
            output_filename = os.path.splitext(os.path.basename(source_path))[0] + '_' + \
                              os.path.splitext(os.path.basename(target_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fourcc = -1 #
        fourcc = cv2.VideoWriter_fourcc(*'hev1')  #
        # out_vid = cv2.VideoWriter(output_path, fourcc, fps, (crop_size * 3, crop_size))
        if verbose == 0:
            out_vid_size = (crop_size * 6, crop_size * 2)
        elif verbose == 1:  # Appearance map only
            out_vid_size = (crop_size * 6, crop_size * 4)
        elif verbose == 2:
            out_vid_size = (crop_size * 6, crop_size * 2)
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, out_vid_size)
    else:
        out_vid = None

    # For each frame in the target video
    valid_frame_ind = 0
    for i in tqdm(range(total_frames)):
        ret, target_frame = target_vid.read()
        if target_frame is None:
            continue
        if i not in target_frame_indices:
            continue
        frame_rgb = target_frame[:, :, ::-1]
        frame_tensor, frame_landmarks, frame_bbox = img_transforms2(frame_rgb, target_landmarks[valid_frame_ind],
                                                                    target_bboxes[valid_frame_ind])
        frame_euler = target_eulers[valid_frame_ind]
        valid_frame_ind += 1

        # Query nearest frame from the source video
        query_point = frame_euler[:2]
        tri_index = tri.find_simplex(query_point)
        b = tri.transform[tri_index, :2].dot(query_point - tri.transform[tri_index, 2])
        bw = np.array([b[0], b[1], 1 - b.sum()])
        tri_vertices = tri.simplices[tri_index]
        valid_tri_indices = tri_vertices < len(filtered_indices)
        tri_vertices = tri_vertices[valid_tri_indices]
        bw = bw[valid_tri_indices]
        bw /= bw.sum()
        source_indices = filtered_indices[tri_vertices]

        # For each source index
        source_pyd_batch_tensors = []
        for source_index in source_indices:
            source_vid.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            res, source_frame = source_vid.read()
            source_rgb = source_frame[:, :, ::-1]
            curr_source_tensor, curr_source_landmarks, curr_source_bbox = \
                img_transforms1(source_rgb, source_landmarks[source_index], source_bboxes[source_index])
            source_pyd_batch_tensors.append(curr_source_tensor)

        # For each pyramid resolution
        reenactment_input_pyd = []
        for j in range(len(frame_landmarks)):
            curr_source_batch_tensors = torch.cat([p[j].unsqueeze(0) for p in source_pyd_batch_tensors], dim=0).to(device)
            curr_heatmaps = frame_landmarks[j].unsqueeze(0).repeat(curr_source_batch_tensors.shape[0], 1, 1, 1).to(device)
            reenactment_input_pyd.append(torch.cat((curr_source_batch_tensors, curr_heatmaps), dim=1))

        reenactment_img_batch_tensors, reenactment_seg_batch_tensors = G(reenactment_input_pyd)

        # Interpolate reenactment tensors using the barycentric coordinates
        # out_img_tensor = bw[0] * out_img_batch_tensors[0]
        # for j in range(1, out_img_batch_tensors.shape[0]):
        #     out_img_tensor += (bw[j] * out_img_batch_tensors[j])

        # Interpolate reenactment tensors using the barycentric coordinates
        reenactment_img_tensor = bw[0] * reenactment_img_batch_tensors[0]
        reenactment_seg_tensor = bw[0] * reenactment_seg_batch_tensors[0]
        for j in range(1, reenactment_img_batch_tensors.shape[0]):
            reenactment_img_tensor += (bw[j] * reenactment_img_batch_tensors[j])
            reenactment_seg_tensor += (bw[j] * reenactment_seg_batch_tensors[j])
        reenactment_img_tensor = reenactment_img_tensor.unsqueeze(0)
        reenactment_seg_tensor = reenactment_seg_tensor.unsqueeze(0)

        # Remove the background of the aligned face
        aligned_face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1  # face
        aligned_background_mask_tensor = ~aligned_face_mask_tensor
        aligned_img_no_background_tensor = reenactment_img_tensor.clone()
        aligned_img_no_background_tensor.masked_fill_(aligned_background_mask_tensor.unsqueeze(1), 1.0)

        if verbose == 2:
            # Query nearest frame from the source video
            source_indices = kdt.query(np.expand_dims(frame_euler, 0), k=1, return_distance=False)
            # source_indices = kdt.query(np.expand_dims(frame_euler[:2], 0), k=1, return_distance=False)
            best_source_index = source_frame_indices[source_indices[0, 0]]
            source_vid.set(cv2.CAP_PROP_POS_FRAMES, best_source_index)
            res, source_frame = source_vid.read()
            source_rgb = source_frame[:, :, ::-1]
            curr_source_tensor, curr_source_landmarks, curr_source_bbox = \
                img_transforms1(source_rgb, source_landmarks[best_source_index], source_bboxes[best_source_index])
            # print(source_indices)

            input_tensor = []
            for j in range(len(curr_source_tensor)):
                frame_landmarks[j] = frame_landmarks[j].to(device)
                curr_source_tensor[j] = curr_source_tensor[j].to(device)
                input_tensor.append(torch.cat((curr_source_tensor[j], frame_landmarks[j]), dim=0).unsqueeze(0))
            reenactment_img_tensor, reenactment_seg_tensor = G(input_tensor)

            # Remove the background of the aligned face
            aligned_face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1  # face
            aligned_background_mask_tensor = ~aligned_face_mask_tensor
            aligned_img_no_background_nn_tensor = reenactment_img_tensor.clone()
            aligned_img_no_background_nn_tensor.masked_fill_(aligned_background_mask_tensor.unsqueeze(1), 1.0)
            out_img_nn_bgr = tensor2bgr(aligned_img_no_background_nn_tensor)

        # Convert back to numpy images
        # source_cropped_bgr = tensor2bgr(source_pyd_batch_tensors[0][0])
        source_selected_cropped_bgr = [tensor2bgr(x[0]) for x in source_pyd_batch_tensors]
        # out_img_bgr = tensor2bgr(out_img_tensor)
        out_img_bgr = tensor2bgr(aligned_img_no_background_tensor)
        target_cropped_bgr = tensor2bgr(frame_tensor[0]).copy()

        # Render selected source views
        source_selected_render_img = np.zeros((crop_size, crop_size * 3, 3), dtype=np.uint8)
        for j in range(len(source_selected_cropped_bgr)):
            source_selected_render_img[:, j * crop_size:(j + 1) * crop_size, :] = source_selected_cropped_bgr[j]

        # Render
        appearance_map_img = render_appearance_map(tri, points, query_point)
        if verbose == 0:
            appearance_map_img = cv2.resize(appearance_map_img, (crop_size * 3, crop_size * 2),
                                            interpolation=cv2.INTER_CUBIC)
            pose_axis_img = draw_axis(np.zeros_like(target_cropped_bgr), frame_euler[0], frame_euler[1], frame_euler[2])
            render_img = np.concatenate((pose_axis_img, out_img_bgr, target_cropped_bgr), axis=1)
            render_img = np.concatenate((source_selected_render_img, render_img), axis=0)
            render_img = np.concatenate((appearance_map_img, render_img), axis=1)
        elif verbose == 1:
            appearance_map_img = cv2.resize(appearance_map_img, out_vid_size, interpolation=cv2.INTER_CUBIC)
            render_img = appearance_map_img
        elif verbose == 2:
            appearance_map_img = cv2.resize(appearance_map_img, (crop_size * 3, crop_size * 2),
                                            interpolation=cv2.INTER_CUBIC)
            render_img = np.concatenate((out_img_nn_bgr, out_img_bgr, target_cropped_bgr), axis=1)
            render_img = np.concatenate((source_selected_render_img, render_img), axis=0)
            render_img = np.concatenate((appearance_map_img, render_img), axis=1)

        if out_vid is not None:
            out_vid.write(render_img)
        if out_vid is None or display:
            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def extract_landmarks_bboxes_euler_from_video(fa, Gp, video_path, frontal_landmarks, img_size=(224, 224), scale=1.2,
                                              device=None, cache_file=None):
    """ Extract face landmarks and bounding boxes from video and also read / write them from cache file.
    :param video_path: Path to video file.
    :param cache_file: Path to file to save the landmarks and bounding boxes in.
        By default it is saved in the same directory of the video file with the same name and extension .pkl.
    :return: tuple (numpy.array, numpy.array, numpy.array):
        frame_indices: The frame indices where a face was detected.
        landmarks: Face landmarks per detected face in each frame.
        bboxes: Bounding box per detected face in each frame.
    """
    cache_file = os.path.splitext(video_path)[0] + '.pkl' if cache_file is None else cache_file
    if not os.path.exists(cache_file):
        frame_indices = []
        landmarks = []
        bboxes = []
        eulers = []

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Failed to read video: ' + video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # For each frame in the video
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if frame is None:
                continue
            frame_rgb = frame[:, :, ::-1]
            detected_faces = fa.face_detector.detect_from_image(frame.copy())

            # Skip current frame there if no faces were detected
            if len(detected_faces) == 0:
                continue
            curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], frame.shape[:2])
            detected_faces = [curr_bbox]

            preds = fa.get_landmarks(frame_rgb, detected_faces)
            curr_landmarks = preds[0]
            # curr_bbox = detected_faces[0][:4]

            # Convert bounding boxes format from [min, max] to [min, size]
            curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

            # Calculate euler angles
            scaled_bbox = scale_bbox(curr_bbox, scale)
            cropped_frame_rgb, cropped_landmarks = crop_img(frame_rgb, curr_landmarks, scaled_bbox)

            scaled_frame_rgb = np.array(F.resize(Image.fromarray(cropped_frame_rgb), img_size, Image.BICUBIC))
            scaled_frame_tensor = rgb2tensor(scaled_frame_rgb.copy()).to(device)
            curr_euler = Gp(scaled_frame_tensor)  # Yaw, Pitch, Roll
            curr_euler = np.array([x.cpu().numpy() for x in curr_euler])


            ### Debug ###
            # scaled_frame_bgr = scaled_frame_rgb[:, :, ::-1].copy()
            # scaled_frame_bgr = draw_axis(scaled_frame_bgr, curr_euler[0], curr_euler[1], curr_euler[2])
            # cv2.imshow('debug', scaled_frame_bgr)
            # cv2.waitKey(1)
            #############

            # Append to list
            frame_indices.append(i)
            landmarks.append(curr_landmarks)
            bboxes.append(curr_bbox)
            eulers.append(curr_euler)

        # Convert to numpy array format
        frame_indices = np.array(frame_indices)
        landmarks = np.array(landmarks)
        bboxes = np.array(bboxes)
        eulers = np.array(eulers)

        # if frame_indices.size == 0:
        #     return frame_indices, landmarks, bboxes

        # Prepare smoothing kernel
        kernel_size = 7
        w = np.hamming(7)
        w /= w.sum()

        # Smooth landmarks
        orig_shape = landmarks.shape
        landmarks = landmarks.reshape(landmarks.shape[0], -1)
        landmarks_padded = np.pad(landmarks, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
        for i in range(landmarks.shape[1]):
            landmarks[:, i] = np.convolve(w, landmarks_padded[:, i], mode='valid')
        landmarks = landmarks.reshape(-1, orig_shape[1], orig_shape[2])

        # Smooth bounding boxes
        bboxes_padded = np.pad(bboxes, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
        for i in range(bboxes.shape[1]):
            bboxes[:, i] = np.convolve(w, bboxes_padded[:, i], mode='valid')

        # Smooth target euler angles
        eulers_padded = np.pad(eulers, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
        for i in range(eulers.shape[1]):
            eulers[:, i] = np.convolve(w, eulers_padded[:, i], mode='valid')

        # Save landmarks and bounding boxes to file
        with open(cache_file, "wb") as fp:  # Pickling
            pickle.dump(frame_indices, fp)
            pickle.dump(landmarks, fp)
            pickle.dump(bboxes, fp)
            pickle.dump(eulers, fp)
    else:
        # Load landmarks and bounding boxes from file
        with open(cache_file, "rb") as fp:  # Unpickling
            frame_indices = pickle.load(fp)
            landmarks = pickle.load(fp)
            bboxes = pickle.load(fp)
            eulers = pickle.load(fp)

    return frame_indices, landmarks, bboxes, eulers


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    from math import cos, sin

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Test video appearance map with hopenet')
    parser.add_argument('source', metavar='VIDEO',
                        help='path to source video')
    parser.add_argument('-t', '--target', metavar='VIDEO',
                        help='paths to target video')
    parser.add_argument('-fl', '--frontal_landmarks', default='frontal_landmarks_256_2_0.npy',
                        help='paths to frontal landmarks (.npy)')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-m', '--model', default='../../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
                        metavar='PATH', help='path to face reenactment model')
    parser.add_argument('-pm', '--pose_model', default='../../weights/hopenet_robust_alpha1.pth', metavar='PATH',
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
    parser.add_argument('-mr', '--min_radius', default=2.0, type=float, metavar='F',
                        help='minimum distance between points in the appearance map')
    parser.add_argument('-cs', '--crop_size', default=256, type=int, metavar='N',
                        help='crop size of the images')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    parser.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                        help='number of steps between each loss plot')
    args = parser.parse_args()
    main(args.source, args.target, args.frontal_landmarks, args.arch, args.model, args.pose_model, args.pil_transforms1,
         args.pil_transforms2, args.tensor_transforms1, args.tensor_transforms2, args.output, args.min_radius,
         args.crop_size, args.display, args.verbose)
