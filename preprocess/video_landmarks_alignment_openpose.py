import os
import shutil
from tqdm import tqdm
import pickle
import numpy as np
import subprocess
import ffmpeg
import cv2
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_3d_from_video


def main(video_path, out_dir=None):
    # Validation
    if not os.path.exists(video_path):
        raise RuntimeError('Input video does not exist: ' + video_path)
    # if not os.path.isdir(out_dir):
    #     raise RuntimeError('Output directory does not exist: ' + out_dir)

    # Find executable
    exe_name = 'OpenPoseDemo'
    exe_path = shutil.which(exe_name)
    exe_dir = os.path.split(os.path.split(exe_path)[0])[0]
    os.chdir(exe_dir)

    # Process video
    video_dir = os.path.split(video_path)[0]
    ret = subprocess.check_output([exe_name, '-video', video_path, '-display', '0', '-render_pose', '0', '-face', '-write_json', video_dir])


def extract_cache_alignment_openpose_iris(video_path, face_pose, face_align=None, img_size=(224, 224),
                                          scale=1.2, device=None, cache_file=None):
    """ Extract face landmarks and bounding boxes from video and also read / write them from cache file.
    :param video_path: Path to video file.
    :param cache_file: Path to file to save the landmarks and bounding boxes in.
        By default it is saved in the same directory of the video file with the same name and extension .pkl.
    :return: tuple (numpy.array, numpy.array, numpy.array):
        frame_indices: The frame indices where a face was detected.
        landmarks: Face landmarks per detected face in each frame.
        bboxes: Bounding box per detected face in each frame.
    """


    # frame_indices, landmarks, bboxes, eulers, landmarks_3d = extract_landmarks_bboxes_euler_3d_from_video(
    #     video_path, face_pose, face_align, img_size, scale, device, cache_file)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_openface')
    parser.add_argument('video_path', help='path to video file')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    args = parser.parse_args()
    main(args.video_path, args.output)
