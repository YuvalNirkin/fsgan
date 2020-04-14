import os
import numpy as np
from tqdm import tqdm
import subprocess
import ffmpeg
import cv2
import face_alignment


def main(in_dir, img_list_file, out_dir, batch_size=16):
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


    params_list = ['FaceLandmarkImg', '-root', in_dir, '-out_dir', out_dir, '-2Dfp', '-gaze', '-pose']
    os.chdir('D:\\Dev\\Shared\\Installations\\OpenFace_2.1.0_win_x64')

    # For each image file in the list
    data = []
    openface_img_list = []
    openface_img_res = []
    for b in tqdm(range(0, len(img_list), batch_size), total=len(img_list) // batch_size, unit='batches'):
        curr_img_batch = img_list[b:min(b + batch_size, len(img_list))]

        # For each image file in the current batch
        curr_param_list = params_list.copy()
        for rel_path in curr_img_batch:
            curr_param_list += ['-f', rel_path]

        # Process images
        # subprocess.call(curr_param_list)
        ret = subprocess.check_output(curr_param_list)

        # Load processed data
        for rel_path in curr_img_batch:
            csv_path = os.path.join(out_dir, os.path.splitext(os.path.basename(rel_path))[0] + '.csv')
            txt_path = os.path.join(out_dir, os.path.splitext(os.path.basename(rel_path))[0] + '_of_details.txt')
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            if os.path.isfile(csv_path):
                curr_data = np.loadtxt(csv_path, delimiter=',', dtype='float32', skiprows=1)
                if len(curr_data.shape) > 1:
                    curr_data = curr_data[0]
                data.append(curr_data)
                openface_img_list.append(rel_path)
                os.remove(csv_path)

                # Get video resolution
                img_path = os.path.join(in_dir, rel_path)
                try:
                    probe = ffmpeg.probe(img_path)
                except ffmpeg.Error as e:
                    raise RuntimeError('Video probe error: ' + e.stderr)

                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream is None:
                    raise RuntimeError('Could not find any streams in the video: ' + os.path.basename(img_path))
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                openface_img_res.append((height, width))

    # Process accumulated data
    data = np.array(data)

    # Eyes: total 56 points
    left_eye_x = data[:, 10:10 + 8].mean(axis=1)
    right_eye_x = data[:, 38:38 + 8].mean(axis=1)
    left_eye_y = data[:, 66:66 + 8].mean(axis=1)
    right_eye_y = data[:, 94:94 + 8].mean(axis=1)
    left_eye = np.expand_dims(np.column_stack((left_eye_x, left_eye_y)), axis=1)
    right_eye = np.expand_dims(np.column_stack((right_eye_x, right_eye_y)), axis=1)

    # Face: total 68 points
    landmarks_data = data[:, 296:]
    col_indices = np.array([i for i in range(landmarks_data.shape[1])]) // 2
    col_indices[range(1, landmarks_data.shape[1], 2)] += (landmarks_data.shape[1] // 2)
    landmarks = landmarks_data[:, col_indices].reshape(-1, landmarks_data.shape[1] // 2, 2)
    landmarks = np.concatenate((landmarks, left_eye, right_eye), axis=1)

    # Calculate bounding boxes from landmarks
    bboxes = []
    for i in range(landmarks.shape[0]):
        height, width = openface_img_res[i]
        bboxes.append(bbox_from_landmarks(landmarks[i], width, height, square=False, top_scale=1.50, bottom_scale=0.0))
    bboxes = np.vstack(bboxes)

    # Write output to file
    np.savetxt(os.path.join(out_dir, 'img_list.txt'), openface_img_list, fmt='%s')
    np.save(os.path.join(out_dir, 'landmarks.npy'), landmarks)
    np.save(os.path.join(out_dir, 'bboxes.npy'), bboxes)


def bbox_from_landmarks(landmarks, img_w, img_h, square=True, top_scale=1.5, bottom_scale=0.1):
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
    parser = argparse.ArgumentParser('add_openface')
    parser.add_argument('input', help='input directory')
    parser.add_argument('-i', '--img_list', required=True, help='image list file')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    args = parser.parse_args()
    main(args.input, args.img_list, args.output, args.batch_size)
