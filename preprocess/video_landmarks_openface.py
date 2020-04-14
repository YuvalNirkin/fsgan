import os
import shutil
from tqdm import tqdm
import pickle
import numpy as np
import subprocess
import ffmpeg
import cv2


def main(video_path, out_dir):
    out_dir = os.path.split(video_path)[0] if out_dir is None else out_dir
    video_basename = os.path.splitext(video_path)[0]
    cache_file_pkl = video_basename + '_openface.pkl'
    cache_file_csv = video_basename + '.csv'
    video_details = video_basename + '_of_details.txt'

    if os.path.isfile(cache_file_pkl):
        print('Cache file already exists: ' + os.path.basename(cache_file_pkl))
        return

    # Process video
    if not os.path.isfile(cache_file_csv):
        exe = shutil.which('FeatureExtraction')
        exe_dir = os.path.split(exe)[0]
        os.chdir(exe_dir)
        subprocess.call(['FeatureExtraction', '-f', video_path, '-out_dir', out_dir, '-2Dfp', '-gaze', '-pose'])

    # Read output csv file
    data = np.loadtxt(cache_file_csv, delimiter=',', dtype='float32', skiprows=1)
    frame_indices = data[:, 0].astype(int) - 1

    # Eyes: total 56 points
    left_eye_x = data[:, 13:13 + 8].mean(axis=1)
    right_eye_x = data[:, 41:41 + 8].mean(axis=1)
    left_eye_y = data[:, 69:69 + 8].mean(axis=1)
    right_eye_y = data[:, 97:97 + 8].mean(axis=1)
    left_eye = np.expand_dims(np.column_stack((left_eye_x, left_eye_y)), axis=1)
    right_eye = np.expand_dims(np.column_stack((right_eye_x, right_eye_y)), axis=1)

    # Face: total 68 points
    landmarks_data = data[:, 299:]
    col_indices = np.array([i for i in range(landmarks_data.shape[1])]) // 2
    col_indices[range(1, landmarks_data.shape[1], 2)] += (landmarks_data.shape[1] // 2)
    landmarks = landmarks_data[:, col_indices].reshape(-1, landmarks_data.shape[1] // 2, 2)
    landmarks = np.concatenate((landmarks, left_eye, right_eye), axis=1)

    # Get video resolution
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise RuntimeError('Video probe error: ' + e.stderr)

    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise RuntimeError('Could not find any streams in the video: ' + os.path.basename(video_path))
    width = int(video_stream['width'])
    height = int(video_stream['height'])

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

    # Calculate bounding boxes from landmarks
    bboxes = []
    for i in range(landmarks.shape[0]):
        bboxes.append(bbox_from_landmarks(landmarks[i], width, height, square=False, top_scale=1.50, bottom_scale=0.0))
    bboxes = np.vstack(bboxes)

    ### Debug ###
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frame_ind = 0
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            continue
        if i not in frame_indices:
            continue

        curr_bbox = bboxes[valid_frame_ind]
        curr_landmarks = landmarks[valid_frame_ind]
        valid_frame_ind += 1

        # Render
        cv2.rectangle(frame, tuple(curr_bbox[:2]), tuple(curr_bbox[:2] + curr_bbox[2:]), (0, 0, 255), 1)
        for point in np.round(curr_landmarks[:68]).astype(int):
            cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
        for point in np.round(curr_landmarks[68:]).astype(int):
            cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    #############

    # Write output to file
    with open(cache_file_pkl, "wb") as fp:  # Pickling
        pickle.dump(frame_indices, fp)
        pickle.dump(landmarks, fp)
        pickle.dump(bboxes, fp)

    # Clean up
    os.remove(video_details)
    os.remove(cache_file_csv)


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


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_openface')
    parser.add_argument('video_path', help='path to video file')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    args = parser.parse_args()
    main(args.video_path, args.output)
