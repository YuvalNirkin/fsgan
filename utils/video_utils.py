""" Video utilities. """

import os
import face_alignment
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from glob import glob
import torchvision.transforms.functional as F
from fsgan.utils.bbox_utils import get_main_bbox
from fsgan.utils.img_utils import rgb2tensor
from fsgan.utils.bbox_utils import scale_bbox, crop_img


def extract_landmarks_bboxes_euler_from_video(video_path, face_pose, face_align=None, img_size=(224, 224),
                                              scale=1.2, device=None, cache_file=None):
    """ Extract face landmarks, bounding boxes, and pose from video and also read / write them to cache file.

    Args:
        video_path (str): Path to video file
        face_pose (nn.Module): Face pose model
        face_align (object): Face alignment model
        img_size (tuple of int): Image crop processing resolution for the face pose model
        scale (float): Multiplier factor to scale tight bounding box
        device (torch.device): Processing device
        cache_file (str): Output cache file path to save the landmarks and bounding boxes in. By default it is saved
            in the same directory of the video file with the same name and extension .pkl

    Returns:
        (np.array, np.array, np.array, np.array): Tuple containing:
            - frame_indices (np.array): The frame indices where a face was detected
            - landmarks (np.array): Face landmarks per detected face in each frame
            - bboxes (np.array): Bounding box per detected face in each frame
            - eulers (np.array): Pose euler angles per detected face in each frame
    """
    # Initialize models
    if face_align is None:
        face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    #
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
            detected_faces = face_align.face_detector.detect_from_image(frame.copy())

            # Skip current frame there if no faces were detected
            if len(detected_faces) == 0:
                continue
            curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], frame.shape[:2])
            detected_faces = [curr_bbox]

            preds = face_align.get_landmarks(frame_rgb, detected_faces)
            curr_landmarks = preds[0]
            # curr_bbox = detected_faces[0][:4]

            # Convert bounding boxes format from [min, max] to [min, size]
            curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

            # Calculate euler angles
            scaled_bbox = scale_bbox(curr_bbox, scale)
            cropped_frame_rgb, cropped_landmarks = crop_img(frame_rgb, curr_landmarks, scaled_bbox)
            scaled_frame_rgb = np.array(F.resize(Image.fromarray(cropped_frame_rgb), img_size, Image.BICUBIC))
            scaled_frame_tensor = rgb2tensor(scaled_frame_rgb.copy()).to(device)
            curr_euler = face_pose(scaled_frame_tensor)  # Yaw, Pitch, Roll
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


def check_landmarks_bboxes_euler_3d_cache(cache_file):
    try:
        with open(cache_file, "rb") as fp:  # Unpickling
            frame_indices = pickle.load(fp)
            landmarks = pickle.load(fp)
            bboxes = pickle.load(fp)
            eulers = pickle.load(fp)
            landmarks_3d = pickle.load(fp)
    except Exception as e:
        return False

    return True


def extract_landmarks_bboxes_euler_3d_from_video(video_path, face_pose, face_align=None, img_size=(224, 224),
                                                 scale=1.2, device=None, cache_file=None):
    """ Extract face landmarks, bounding boxes, pose, and 3D face landmarks from video and also read / write them
    to cache file.

    Args:
        video_path (str): Path to video file
        face_pose (nn.Module): Face pose model
        face_align (object): Face alignment model
        img_size (tuple of int): Image crop processing resolution for the face pose model
        scale (float): Multiplier factor to scale tight bounding box
        device (torch.device): Processing device
        cache_file (str): Output cache file path to save the landmarks and bounding boxes in. By default it is saved
            in the same directory of the video file with the same name and extension .pkl

    Returns:
        (np.array, np.array, np.array, np.array, np.array): Tuple containing:
            - frame_indices (np.array): The frame indices where a face was detected
            - landmarks (np.array): Face landmarks per detected face in each frame
            - bboxes (np.array): Bounding box per detected face in each frame
            - eulers (np.array): Pose euler angles per detected face in each frame
            - landmarks_3d (np.array): 3D face landmarks per detected face in each frame
    """
    # Initialize models
    if face_align is None:
        face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)

    #
    cache_file = os.path.splitext(video_path)[0] + '.pkl' if cache_file is None else cache_file
    if not os.path.exists(cache_file) or not check_landmarks_bboxes_euler_3d_cache(cache_file):
        frame_indices, landmarks, bboxes, eulers = \
            extract_landmarks_bboxes_euler_from_video(video_path, face_pose, None, img_size, scale, device, cache_file)
        landmarks_3d = []

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
            detected_faces = face_align.face_detector.detect_from_image(frame.copy())

            # Skip current frame there if no faces were detected
            if len(detected_faces) == 0:
                continue
            curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], frame.shape[:2])
            detected_faces = [curr_bbox]

            preds = face_align.get_landmarks(frame_rgb, detected_faces)
            curr_landmarks_3d = preds[0]

            # Append to list
            landmarks_3d.append(curr_landmarks_3d)

        # Convert to numpy array format
        landmarks_3d = np.array(landmarks_3d)

        # if frame_indices.size == 0:
        #     return frame_indices, landmarks, bboxes

        # Prepare smoothing kernel
        kernel_size = 7
        w = np.hamming(7)
        w /= w.sum()

        # Smooth 3D landmarks
        orig_shape = landmarks_3d.shape
        landmarks_3d = landmarks_3d.reshape(landmarks_3d.shape[0], -1)
        landmarks_3d_padded = np.pad(landmarks_3d, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
        for i in range(landmarks_3d.shape[1]):
            landmarks_3d[:, i] = np.convolve(w, landmarks_3d_padded[:, i], mode='valid')
        landmarks_3d = landmarks_3d.reshape(-1, orig_shape[1], orig_shape[2])

        # Save landmarks and bounding boxes to file
        with open(cache_file, "wb") as fp:  # Pickling
            pickle.dump(frame_indices, fp)
            pickle.dump(landmarks, fp)
            pickle.dump(bboxes, fp)
            pickle.dump(eulers, fp)
            pickle.dump(landmarks_3d, fp)
    elif check_landmarks_bboxes_euler_3d_cache(cache_file):
        # Load landmarks and bounding boxes from file
        with open(cache_file, "rb") as fp:  # Unpickling
            frame_indices = pickle.load(fp)
            landmarks = pickle.load(fp)
            bboxes = pickle.load(fp)
            eulers = pickle.load(fp)
            landmarks_3d = pickle.load(fp)

    return frame_indices, landmarks, bboxes, eulers, landmarks_3d


def extract_landmarks_bboxes_euler_from_images(img_dir, face_pose, face_align=None, img_size=(224, 224),
                                              scale=1.2, device=None, cache_file=None):
    """ Extract face landmarks, bounding boxes, and pose from images and also read / write them to cache file.

    Args:
        video_path (str): Path to video file
        face_pose (nn.Module): Face pose model
        face_align (object): Face alignment model
        img_size (tuple of int): Image crop processing resolution for the face pose model
        scale (float): Multiplier factor to scale tight bounding box
        device (torch.device): Processing device
        cache_file (str): Output cache file path to save the landmarks and bounding boxes in. By default it is saved
            in the same directory of the video file with the same name and extension .pkl

    Returns:
        (np.array, np.array, np.array, np.array): Tuple containing:
            - frame_indices (np.array): The frame indices where a face was detected
            - landmarks (np.array): Face landmarks per detected face in each frame
            - bboxes (np.array): Bounding box per detected face in each frame
            - eulers (np.array): Pose euler angles per detected face in each frame
    """
    # Initialize models
    if face_align is None:
        face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    #
    cache_file = img_dir + '.pkl' if cache_file is None else cache_file
    if not os.path.exists(cache_file):
        frame_indices = []
        landmarks = []
        bboxes = []
        eulers = []

        # Parse images
        img_paths = glob(os.path.join(img_dir, '*.jpg'))

        # For each image in the directory
        for i, img_path in tqdm(enumerate(img_paths), unit='images', total=len(img_paths)):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = img_bgr[:, :, ::-1]
            detected_faces = face_align.face_detector.detect_from_image(img_bgr.copy())

            # Skip current frame there if no faces were detected
            if len(detected_faces) == 0:
                continue
            curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], img_bgr.shape[:2])
            detected_faces = [curr_bbox]

            preds = face_align.get_landmarks(img_rgb, detected_faces)
            curr_landmarks = preds[0]
            # curr_bbox = detected_faces[0][:4]

            # Convert bounding boxes format from [min, max] to [min, size]
            curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

            # Calculate euler angles
            scaled_bbox = scale_bbox(curr_bbox, scale)
            cropped_frame_rgb, cropped_landmarks = crop_img(img_rgb, curr_landmarks, scaled_bbox)
            scaled_frame_rgb = np.array(F.resize(Image.fromarray(cropped_frame_rgb), img_size, Image.BICUBIC))
            scaled_frame_tensor = rgb2tensor(scaled_frame_rgb.copy()).to(device)
            curr_euler = face_pose(scaled_frame_tensor)  # Yaw, Pitch, Roll
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
