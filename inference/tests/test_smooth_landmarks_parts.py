import os
import pickle
import numpy as np


def smooth_landmarks(landmarks, kernel_size=7, weight_func='hamming'):
    # Prepare smoothing kernel
    w = np.hamming(kernel_size) if weight_func == 'hamming' else np.ones(kernel_size)
    w /= w.sum()

    # Smooth landmarks
    orig_shape = landmarks.shape
    landmarks = landmarks.reshape(landmarks.shape[0], -1)
    landmarks_padded = np.pad(landmarks, ((kernel_size // 2, kernel_size // 2), (0, 0)), 'reflect')
    for i in range(landmarks.shape[1]):
        landmarks[:, i] = np.convolve(w, landmarks_padded[:, i], mode='valid')
    landmarks = landmarks.reshape(-1, orig_shape[1], orig_shape[2])

    return landmarks


def main(input_path, output_path, indices=None, kernel_size=7, weight_func='hamming'):
    # Load landmarks and bounding boxes from file
    with open(input_path, "rb") as fp:  # Unpickling
        frame_indices = pickle.load(fp)
        landmarks = pickle.load(fp)
        bboxes = pickle.load(fp)
        eulers = pickle.load(fp)

    # Extract landmarks parts
    landmarks_parts = eval('landmarks[:, %s]' % indices) if indices is not None else landmarks

    # Smooth landmarks
    landmarks_parts = smooth_landmarks(landmarks_parts, kernel_size, weight_func)
    # if indices is not None:
    #     eval('landmarks[:, %s] = landmarks_parts[:, %s]' % (indices, indices))
    # else:
    #     landmarks = landmarks_parts

    # Save landmarks and bounding boxes to file
    with open(output_path, "wb") as fp:  # Pickling
        pickle.dump(frame_indices, fp)
        pickle.dump(landmarks, fp)
        pickle.dump(bboxes, fp)
        pickle.dump(eulers, fp)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('test_smooth_landmarks_parts')
    parser.add_argument('input', metavar='PATH',
                        help='path to input cache file')
    parser.add_argument('-o', '--output', metavar='PATH', required=True,
                        help='path to output cache file')
    parser.add_argument('-i', '--indices',
                        help='python style indices (e.g 0:10')
    parser.add_argument('-ks', '--kernel_size', default=7, type=int, metavar='N',
                        help='kernel size (default: 7)')
    parser.add_argument('-wf', '--weight_func', default='hamming',
                        help='weight function (default: hamming')
    args = parser.parse_args()
    main(args.input, args.output, args.indices, args.kernel_size, args.weight_func)
