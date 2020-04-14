import os
import random
import pickle
import torch.utils.data as data
import numpy as np
import cv2
import torch
from fsgan.utils.utils import random_pair, random_pair_range
from fsgan.utils.video_utils import Sequence
import fsgan.datasets.img_landmarks_transforms as img_landmarks_transforms


def is_video(fname):
    ext = os.path.splitext(fname)[1]
    return ext.lower() == '.mp4'


def make_dataset(dir):
    files = []
    dir = os.path.expanduser(dir)
    for fname in sorted(os.listdir(dir)):
        if is_video(fname):
            path = os.path.join(dir, fname)
            files.append(path)

    return files


def make_dataset_dirs(dir):
    files = []
    dir = os.path.expanduser(dir)
    for fname in sorted(os.listdir(dir)):
        path = os.path.join(dir, fname)
        if os.path.isdir(path):
            files += make_dataset(path)
        elif is_video(fname):
            files.append(path)

    return files


def parse_file_paths(root, file_list=None):
    if file_list is None:
        return make_dataset_dirs(root)

    file_list_path = file_list if os.path.exists(file_list) else os.path.join(root, file_list)
    if not os.path.exists(file_list_path):
        raise (RuntimeError('Could not find image list file: ' + file_list))
    with open(file_list_path, 'r') as f:
        file_rel_paths = f.read().splitlines()
    file_paths = [os.path.join(root, f) for f in file_rel_paths]

    return file_paths


def calc_weights_for_balanced_classes(targets):
    _, class_weights = np.unique(targets, return_counts=True)
    class_weights = np.sum(class_weights) / class_weights
    weights = np.array([class_weights[i] for i in targets])
    weights = weights / np.sum(weights)

    return weights


class VideoSeqDataset(data.Dataset):
    """A dataset for loading video sequences.

    Args:
        root (string): Root directory path or file list path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
     Attributes:
        video_paths (list): List of video paths
    """

    def __init__(self, root, file_list=None, target_list=None, transform=None, frame_window=1,
                 seq_postfix='_dsfd_seq.pkl', ignore_landmarks=False):
        if os.path.isdir(root):
            self.video_paths = parse_file_paths(root, file_list)
        else:
            self.video_paths = [root]
        if len(self.video_paths) == 0:
            raise RuntimeError("Found 0 videos in subfolders of: " + root + "\n")
        self.seq_paths = [os.path.splitext(p)[0] + seq_postfix for p in self.video_paths]
        self.root = root
        self.transform = transform
        self.frame_window = frame_window
        self.ignore_landmarks = ignore_landmarks

        if target_list is None:
            self.targets = None
            self.weights = np.ones(len(self.seq_paths))
            self.classes = list(range(len(self.seq_paths)))
        else:
            targets_list_path = target_list if os.path.exists(target_list) else os.path.join(root, target_list)
            if not os.path.exists(targets_list_path):
                raise (RuntimeError('Could not find target list file: ' + target_list))
            self.targets = np.loadtxt(targets_list_path, dtype='int64')
            self.classes = np.unique(self.targets)
            self.weights = calc_weights_for_balanced_classes(self.targets)

        # Make sure all sequence files exist
        for seq_path in self.seq_paths:
            if not os.path.isfile(seq_path):
                raise RuntimeError('Sequence file "%s" does not exist!' % seq_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target) where target is True for same identity else False.
        """
        target = self.targets[index] if self.targets is not None else None

        # Open video file
        cap = cv2.VideoCapture(self.video_paths[index])
        # print(self.video_paths[index])  # debug

        # Read corresponding sequences from file
        with open(self.seq_paths[index], "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # Randomly select a sequence from the sequence list
        seq = seq_list[random.randint(0, len(seq_list) - 1)]
        start_index = random.randint(seq.start_index, seq.start_index + len(seq) - self.frame_window)

        frame_list, landmarks_list, bbox_list = [], [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        for i, frame_index in enumerate(range(start_index, start_index + self.frame_window)):
            ret, frame_bgr = cap.read()
            if frame_bgr is None:
                raise RuntimeError('Failed to read frame from video: "%s"' % os.path.basename(self.video_paths[index]))
            frame_rgb = frame_bgr[:, :, ::-1]
            bbox = seq.detections[frame_index - seq.start_index]
            landmarks = seq.landmarks[frame_index - seq.start_index] if not self.ignore_landmarks else None

            # Add to lists
            frame_list.append(frame_rgb)
            landmarks_list.append(landmarks)
            bbox_list.append(bbox)

        # Apply transformation
        if self.transform is not None:
            if self.ignore_landmarks:
                frame_list = self.transform(frame_list, bbox_list)
            else:
                frame_list, landmarks_list = self.transform(frame_list, bbox_list, landmarks_list)

        # Check for pyramids
        if isinstance(frame_list[0], (list, tuple)):
            frame_window, landmarks_window = [], []
            for p in range(len(frame_list[0])):
                frame_window.append(torch.cat([frame[p].unsqueeze(1) for frame in frame_list], dim=1).squeeze(1))
                if not self.ignore_landmarks:
                    landmarks_window.append(torch.cat([lms[p].unsqueeze(0) for lms in landmarks_list], dim=0).squeeze(1))
        else:
            frame_window = torch.cat([frame.unsqueeze(1) for frame in frame_list], dim=1).squeeze(1)
            if not self.ignore_landmarks:
                landmarks_window = torch.cat([lms.unsqueeze(0) for lms in landmarks_list], dim=0).squeeze(1)

        if self.ignore_landmarks:
            if target is None:
                return frame_window
            else:
                return frame_window, target

        else:
            if target is None:
                return frame_window, landmarks_window
            else:
                return frame_window, landmarks_window, target

    def __len__(self):
        return len(self.video_paths)


class VideoSeqPairDataset(VideoSeqDataset):
    """A dataset for loading video sequences pairs.

    Args:
        root (string): Root directory path or file list path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
     Attributes:
        video_paths (list): List of video paths
    """

    def __init__(self, root, file_list=None, target_list=None, transform=None, frame_window=5,
                 seq_postfix='_dsfd_seq.pkl', same_prob=0.5, ignore_landmarks=False):
        super(VideoSeqPairDataset, self).__init__(root, file_list, target_list, transform, frame_window, seq_postfix,
                                                  ignore_landmarks)
        self.same_prob = same_prob

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target) where target is True for same identity else False.
        """
        if random.random() < self.same_prob:
            # Same identitiy
            same = True

            # Open video file
            cap = cv2.VideoCapture(self.video_paths[index])

            # Read corresponding sequences from file
            with open(self.seq_paths[index], "rb") as fp:  # Unpickling
                seq_list = pickle.load(fp)

            # Randomly select a sequence from the sequence list
            seq = seq_list[random.randint(0, len(seq_list) - 1)]

            # Randomly select a pair of non intersecting frame windows
            pair_start_indices = random_pair_range(seq.start_index, seq.start_index + len(seq) - 1, self.frame_window)

            # Read frame windows
            frame_list_pair, landmarks_list_pair, bbox_list_pair = [], [], []
            for p, start_index in enumerate(pair_start_indices):
                frame_list, landmarks_list, bbox_list = [], [], []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
                for i, frame_index in enumerate(range(start_index, start_index + self.frame_window)):
                    ret, frame_bgr = cap.read()
                    if frame_bgr is None:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Debug
                        raise RuntimeError(
                            'Failed to read frame from video: "%s" [%d/%d]' %
                            (os.path.basename(self.video_paths[index]), frame_index,  total_frames - 1))
                    frame_rgb = frame_bgr[:, :, ::-1]
                    bbox = seq.detections[frame_index - seq.start_index]
                    landmarks = seq.landmarks[frame_index - seq.start_index] if not self.ignore_landmarks else None

                    # Add to lists
                    frame_list.append(frame_rgb)
                    landmarks_list.append(landmarks)
                    bbox_list.append(bbox)

                # Add to pair lists
                frame_list_pair.append(frame_list)
                landmarks_list_pair.append(landmarks_list)
                bbox_list_pair.append(bbox_list)

            # Apply transformation
            if self.transform is not None:
                if self.ignore_landmarks:
                    frame_list_pair = self.transform(frame_list_pair, bbox_list_pair)
                else:
                    frame_list_pair, landmarks_list_pair = \
                        self.transform(frame_list_pair, bbox_list_pair, landmarks_list_pair)

            # Check for pyramids
            frame_window_pair, landmarks_window_pair = [], []
            for p in range(len(frame_list_pair)):
                if isinstance(frame_list_pair[p][0], (list, tuple)):
                    frame_window, landmarks_window = [], []
                    for i in range(len(frame_list_pair[p][0])):
                        frame_window.append(torch.cat([frame[i].unsqueeze(1) for frame in frame_list_pair[p]], dim=1).squeeze(1))
                        if not self.ignore_landmarks:
                            landmarks_window.append(torch.cat([lms[i].unsqueeze(0) for lms in landmarks_list_pair[p]], dim=0).squeeze(1))
                else:
                    frame_window = torch.cat([frame.unsqueeze(1) for frame in frame_list_pair[p]], dim=1).squeeze(1)
                    if not self.ignore_landmarks:
                        landmarks_window = torch.cat([lms.unsqueeze(0) for lms in landmarks_list_pair[p]], dim=0).squeeze(1)

                # Add to frame window pair list
                frame_window_pair.append(frame_window)
                if not self.ignore_landmarks:
                    landmarks_window_pair.append(landmarks_window)
        else:
            # Different identity
            same = False
            pair_indices = random_pair(len(self.video_paths), index1=index)
            frame_window_pair, landmarks_window_pair = [], []
            for pair_index in pair_indices:
                if self.ignore_landmarks:
                    frame_window = super(VideoSeqPairDataset, self).__getitem__(pair_index)
                else:
                    frame_window, landmarks_window = super(VideoSeqPairDataset, self).__getitem__(pair_index)
                    landmarks_window_pair.append(landmarks_window)
                frame_window_pair.append(frame_window)

        if self.ignore_landmarks:
            return frame_window_pair, np.array(same, dtype='float32')
        else:
            return frame_window_pair, landmarks_window_pair, np.array(same, dtype='float32')


def main(dataset='opencv_video_seq_dataset.VideoSeqDataset', np_transforms=None,
         tensor_transforms=('img_landmarks_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         workers=4, batch_size=4):
    import time
    from fsgan.utils.obj_factory import obj_factory
    from fsgan.utils.img_utils import tensor2bgr

    np_transforms = obj_factory(np_transforms) if np_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = img_landmarks_transforms.Compose(np_transforms + tensor_transforms)
    dataset = obj_factory(dataset, transform=img_transforms)
    # dataset = VideoSeqDataset(root_path, img_list_path, transform=img_transforms, frame_window=frame_window)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, drop_last=True,
                                 shuffle=True)

    start = time.time()
    for frame_window, landmarks_window in dataloader:
        # print(frame_window.shape)

        if isinstance(frame_window, (list, tuple)):
            # For each batch
            for b in range(frame_window[0].shape[0]):
                # For each frame window in the list
                for p in range(len(frame_window)):
                    # For each frame in the window
                    for f in range(frame_window[p].shape[2]):
                        print(frame_window[p][b, :, f, :, :].shape)
                        # Render
                        render_img = tensor2bgr(frame_window[p][b, :, f, :, :]).copy()
                        landmarks = landmarks_window[p][b, f, :, :].numpy()
                        # for point in np.round(landmarks).astype(int):
                        for point in landmarks:
                            cv2.circle(render_img, (point[0], point[1]), 2, (0, 0, 255), -1)
                        cv2.imshow('render_img', render_img)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            break
        else:
            # For each batch
            for b in range(frame_window.shape[0]):
                # For each frame in the window
                for f in range(frame_window.shape[2]):
                    print(frame_window[b, :, f, :, :].shape)
                    # Render
                    render_img = tensor2bgr(frame_window[b, :, f, :, :]).copy()
                    landmarks = landmarks_window[b, f, :, :].numpy()
                    # for point in np.round(landmarks).astype(int):
                    for point in landmarks:
                        cv2.circle(render_img, (point[0], point[1]), 2, (0, 0, 255), -1)
                    cv2.imshow('render_img', render_img)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
    end = time.time()
    print('elapsed time: %f[s]' % (end - start))


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument('dataset', metavar='OBJ',
                        help='dataset object')
    parser.add_argument('-nt', '--np_transforms', default=None, nargs='+', help='Numpy transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('img_landmarks_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    main(**vars(parser.parse_args()))
