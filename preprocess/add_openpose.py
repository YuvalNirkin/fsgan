import os
import shutil
import json
import numpy as np
from tqdm import tqdm
import subprocess
from itertools import groupby
import ffmpeg
import cv2
import face_alignment


def main(in_dir, img_list_file, out_dir):
    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory does not exist: ' + in_dir)
    if not os.path.exists(img_list_file):
        raise RuntimeError('Image list file does not exist: ' + img_list_file)
    if not os.path.isdir(out_dir):
        raise RuntimeError('Output directory does not exist: ' + out_dir)

    # Load data
    with open(img_list_file, 'r') as f:
        img_list = np.array(f.read().splitlines())

    # Find executable
    exe_name = '04_keypoints_from_images'
    exe_path = shutil.which(exe_name)
    exe_dir = os.path.split(os.path.split(exe_path)[0])[0]
    os.chdir(exe_dir)

    # Process images in groups
    landmarks_list = []
    out_img_list = []
    for key, group in tqdm(groupby(enumerate(img_list), lambda x: os.path.split(x[1])[0])):
        group_list = list(group)

        # Copy all group images to a temporary directory
        temp_dir = os.path.join(out_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
            for _, img_rel_path in group_list:
                img_path = os.path.join(in_dir, img_rel_path)
                out_path = os.path.join(temp_dir, os.path.basename(img_path))
                shutil.copy(img_path, out_path)

        # Process temp directory
        ret = subprocess.check_output([exe_name, '-image_dir', temp_dir, '-no_display', '-face', '-write_json', temp_dir])

        # Load jsons
        for i, (img_index, _) in enumerate(group_list):
            json_path = os.path.join(temp_dir, '%d_keypoints.json' % i)
            with open(json_path) as f:
                data = json.load(f)
            if len(data['people']) > 0:
                curr_landmarks = np.array(data['people'][0]['face_keypoints_2d']).reshape(-1, 3)[:, :2]
                landmarks_list.append(np.expand_dims(curr_landmarks, axis=0))
                out_img_list.append(img_list[img_index])

        # Clean up
        shutil.rmtree(temp_dir)

    # Combine landmarks
    landmarks = np.vstack(landmarks_list)

    # Calculate bounding boxes from landmarks
    bboxes = []
    for i in range(landmarks.shape[0]):
        bboxes.append(bbox_from_landmarks(landmarks[i], square=False, top_scale=1.50, bottom_scale=0.0))
    bboxes = np.vstack(bboxes)

    # Write output to file
    np.savetxt(os.path.join(out_dir, 'img_list.txt'), out_img_list, fmt='%s')
    np.save(os.path.join(out_dir, 'landmarks.npy'), landmarks)
    np.save(os.path.join(out_dir, 'bboxes.npy'), bboxes)


def bbox_from_landmarks(landmarks, square=True, top_scale=1.5, bottom_scale=0.1):
    # Calculate bounding box
    minp = np.min(landmarks, axis=0).astype('float')
    maxp = np.max(landmarks, axis=0).astype('float')
    size = maxp - minp + 1
    center = (maxp + minp)/2.0
    avg = np.round(np.mean(landmarks, axis=0))
    dev = center - avg
    dev_lt = np.round(np.array([0.1*size[0], size[1]*(np.maximum(size[0] / size[1], 1)*top_scale-1)])) + \
             np.abs(np.minimum(dev, 0))
    dev_rb = np.round(bottom_scale*size) + np.maximum(dev, 0)

    minp = minp - dev_lt
    maxp = maxp + dev_rb

    # Limit to frame boundaries
    # minp = np.maximum(minp - dev_lt, 0)
    # maxp = np.minimum(maxp + dev_rb, np.array([img_w - 1, img_h - 1]))

    # Make square
    if square:
        size = maxp - minp + 1
        sq_size = np.max(size)
        half_sq_size = np.round((sq_size - 1) / 2)
        center = np.round((maxp + minp) / 2.0)
        minp = center - half_sq_size
        maxp = center + half_sq_size

        # Limit to frame boundaries
        # minp = np.maximum(minp, 0)
        # maxp = np.minimum(maxp, np.array([img_w - 1, img_h - 1]))

    # Output bounding box
    bbox = np.round(np.array([minp[0], minp[1], maxp[0] - minp[0], maxp[1] - minp[1]])).astype('int32')
    return bbox


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('add_openpose')
    parser.add_argument('input', help='input directory')
    parser.add_argument('-i', '--img_list', required=True, help='image list file')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    args = parser.parse_args()
    main(args.input, args.img_list, args.output)
