import os
import numpy as np
from tqdm import tqdm
import cv2
import face_alignment


def main(in_dir, img_list_file, bboxes_file, out_path):
    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory does not exist: ' + in_dir)
    if not os.path.exists(img_list_file):
        raise RuntimeError('Image list file does not exist: ' + img_list_file)
    if not os.path.exists(bboxes_file):
        raise RuntimeError('Bounding boxes file does not exist: ' + bboxes_file)

    face_align_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)

    # Load data
    with open(img_list_file, 'r') as f:
        img_list = np.array(f.read().splitlines())
    bboxes = np.load(bboxes_file)

    # For each image file in the list
    landmarks_3d = []
    for i, img_file in tqdm(enumerate(img_list), total=len(img_list), unit='images'):
        img_path = os.path.join(in_dir, img_file)
        img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1]
        bbox = bboxes[i]
        detection = [np.concatenate((bbox[:2], bbox[:2] + bbox[2:]))]

        preds = face_align_3d.get_landmarks(img_rgb, detection)
        curr_landmarks_3d = preds[0]
        landmarks_3d.append(curr_landmarks_3d)

        ### Debug ###
        # bbox = np.round(bbox).astype(int)
        # cv2.rectangle(img_bgr, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:]), (0, 0, 255), 1)
        # for point in np.round(curr_landmarks_3d).astype(int):
        #     cv2.circle(img_bgr, (point[0], point[1]), 2, (0, 0, 255), -1)
        # cv2.imshow('img_bgr', img_bgr)
        # cv2.waitKey(0)
        #############

    # Write output to file
    np.save(out_path, landmarks_3d)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('add_3d_landmarks')
    parser.add_argument('input', help='input directory')
    parser.add_argument('-i', '--img_list', required=True, help='bounding boxes file')
    parser.add_argument('-b', '--bboxes', required=True, help='bounding boxes file')
    parser.add_argument('-o', '--output', required=True, help='output path')
    args = parser.parse_args()
    main(args.input, args.img_list, args.bboxes, args.output)
