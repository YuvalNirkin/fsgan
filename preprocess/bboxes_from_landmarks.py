import numpy as np
from tqdm import tqdm


def main(landmarks_file, out_bboxes_file, top_scale=1.5, bottom_scale=0.0):
    landmarks = np.load(landmarks_file)
    bboxes = []
    for curr_landmarks in tqdm(landmarks):
        curr_bbox = bbox_from_landmarks2(curr_landmarks, top_scale=top_scale, bottom_scale=bottom_scale)
        bboxes.append(curr_bbox)
    bboxes = np.vstack(bboxes)
    np.save(out_bboxes_file, bboxes)


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


def bbox_from_landmarks2(landmarks, square=True, top_scale=1.5, bottom_scale=0.1):
    center = landmarks[27, :]
    bottom = landmarks[8, :]
    radius = np.linalg.norm(center - bottom)

    minp = center - radius
    maxp = center + radius


    # Make square
    if square:
        size = maxp - minp + 1
        sq_size = np.max(size)
        half_sq_size = np.round((sq_size - 1) / 2)
        center = np.round((maxp + minp) / 2.0)
        minp = center - half_sq_size
        maxp = center + half_sq_size

    # Output bounding box
    bbox = np.round(np.array([minp[0], minp[1], maxp[0] - minp[0], maxp[1] - minp[1]])).astype('int32')
    return bbox


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('bboxes_from_landmarks')
    parser.add_argument('input', help='input landmarks file')
    parser.add_argument('-o', '--output', required=True, help='output bounding box file')
    parser.add_argument('-ts', '--top_scale', default=1.5, type=float, help='boundibg box top scale')
    parser.add_argument('-bs', '--bottom_scale', default=0.0, type=float, help='boundibg box bottom scale')
    args = parser.parse_args()
    main(args.input, args.output)
