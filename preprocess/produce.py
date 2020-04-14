import os
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2


def main(in_dir, out_dir, copy_files=True):
    img_files = []
    landmarks = np.array([], dtype=float).reshape(0, 68, 2)
    bboxes = np.array([], dtype=float).reshape(0, 4)
    eulers = np.array([], dtype=float).reshape(0, 3)
    landmarks_3d = np.array([], dtype=float).reshape(0, 68, 3)

    # For each sub-directory in the input directory
    subdirs = [os.path.join(in_dir, d) for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    for curr_dir in tqdm(sorted(subdirs)):
        curr_landmarks_file = curr_dir + '_landmarks.npy'
        curr_bboxes_file = curr_dir + '_bboxes.npy'
        curr_eulers_file = curr_dir + '_eulers.npy'
        curr_landmarks_3d_file = curr_dir + '_landmarks_3d.npy'
        if os.path.exists(curr_landmarks_file) and os.path.exists(curr_bboxes_file):
            curr_img_files = [os.path.basename(curr_dir) + '/' + os.path.basename(f) for f in glob(curr_dir + '/*.jpg')]
            curr_img_files = sorted(curr_img_files)
            curr_landmarks = np.load(curr_landmarks_file)
            curr_bboxes = np.load(curr_bboxes_file)
            curr_eulers = np.load(curr_eulers_file)
            curr_landmarks_3d = np.load(curr_landmarks_3d_file)
            img_files += curr_img_files
            landmarks = np.vstack((landmarks, curr_landmarks))
            bboxes = np.vstack((bboxes, curr_bboxes))
            eulers = np.vstack((eulers, curr_eulers))
            landmarks_3d = np.vstack((landmarks_3d, curr_landmarks_3d))
            if copy_files:
                shutil.copytree(curr_dir, os.path.join(out_dir, os.path.basename(curr_dir)))

            ### Debug ###
            # frame = cv2.imread(os.path.join(curr_dir, os.path.basename(curr_img_files[0])))
            # for point in curr_landmarks[0]:
            #     cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
            # bbox = curr_bboxes[0]
            # cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:]), (0, 255, 0), 1)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            #############

    # Write landmarks and bounding boxes to file
    np.savetxt(os.path.join(out_dir, 'img_list.txt'), img_files, fmt='%s')
    np.save(os.path.join(out_dir, 'landmarks.npy'), landmarks)
    np.save(os.path.join(out_dir, 'bboxes.npy'), bboxes)
    np.save(os.path.join(out_dir, 'eulers.npy'), eulers)
    np.save(os.path.join(out_dir, 'landmarks_3d.npy'), landmarks_3d)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('produce')
    parser.add_argument('input', metavar='DIR', help='input directory')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    parser.add_argument('-c', '--copy', action='store_true', help='copy files')
    args = parser.parse_args()
    main(args.input, args.output, args.copy)
