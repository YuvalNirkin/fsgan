import os
import random
import pickle
import torch.utils.data as data
import numpy as np
import cv2
import torch
from fsgan.utils.utils import random_pair, random_pair_range
from fsgan.utils.video_utils import Sequence, get_video_info
import fsgan.datasets.img_lms_pose_transforms as img_lms_pose_transforms
from fsgan.utils.seg_utils import decode_binary_mask


def parse_file_paths(root, file_list, seq_postfix='_dsfd_seq.pkl', postfixes=('.mp4',)):
    file_list_path = file_list if os.path.exists(file_list) else os.path.join(root, file_list)
    if not os.path.exists(file_list_path):
        raise (RuntimeError('Could not find image list file: ' + file_list))
    with open(file_list_path, 'r') as f:
        file_rel_paths = f.read().splitlines()
    file_paths = [os.path.join(root, f) for f in file_rel_paths]

    # For each file path
    out_file_paths = []
    for file_path in file_paths:
        file_path_no_ext = os.path.splitext(file_path)[0]
        file_name_no_ext = os.path.basename(file_path_no_ext)
        seq_file_path = os.path.join(file_path_no_ext, file_name_no_ext + seq_postfix)

        if not os.path.isfile(seq_file_path):
            continue

        # Load sequences from file
        with open(seq_file_path, "rb") as fp:  # Unpickling
            seq_list = pickle.load(fp)

        # For each sequence
        seq_file_paths = []
        for seq in seq_list:
            curr_file_base_path = os.path.join(file_path_no_ext, file_name_no_ext + '_seq%02d' % seq.id)

            # For each postfix
            curr_seq_file_paths = []
            for postfix in postfixes:
                curr_seq_file_path = curr_file_base_path + postfix
                assert os.path.isfile(curr_seq_file_path), 'missing sequence file: "%s"' % curr_seq_file_path
                curr_seq_file_paths.append(curr_seq_file_path)

            seq_file_paths.append(curr_seq_file_paths)

        out_file_paths.append(seq_file_paths)

    return out_file_paths


def calc_weights_for_balanced_classes(targets):
    _, class_weights = np.unique(targets, return_counts=True)
    class_weights = np.sum(class_weights) / class_weights
    weights = np.array([class_weights[i] for i in targets])
    weights = weights / np.sum(weights)

    return weights


def get_total_frames_from_file(file_path):
    if file_path.endswith('.mp4'):
        _, _, total_frames, _ = get_video_info(file_path)
        return total_frames
    elif file_path.endswith('_lms.npz'):
        return np.load(file_path)['landmarks_smoothed'].shape[0]
    elif file_path.endswith('_pose.npz'):
        return np.load(file_path)['poses_smoothed'].shape[0]
    elif file_path.endswith('_seg.pkl'):
        return len(np.load(file_path, allow_pickle=True))

    return 0


class SeqDataset(data.Dataset):
    """A dataset for loading video sequences and their meta-data.

    Args:
        root (str): Root directory path
        file_list (str, optional): A path to a list of files. Can be relative to the root directory
        target_list (str, optional): A path to a list of targets corresponding to to each of the files in
            ``file_list``. Can be relative to the root directory
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        frame_window (int): The size of the temporal frame window to load for each query. If greater than one,
            an additional temporal dimension of the same size will be added for each returned tensor
        seq_postfix (str): Sequence cache file postfix
        postfixes (list of str): The postfixes of the sequence files to load. The order of the postifixes will
            determine the order of the corresponding tensors that will be returned. By default only the cropped video
            sequence files will be loaded

    Attributes:
        file_paths (list of str): List of the parsed file paths according to the specified postfixes
        targets (np.array of int) The targets corresponding to each of the files in ``file_paths``
        classes (list of int) The classes corresponding to each of the files in ``file_paths``. The classes will be
            automatically determined from the targets if they are specified, otherwise they will be the file indices
        weights (np.array): Per file weights determined by their targets. Used for balanced training
    """
    def __init__(self, root, file_list=None, target_list=None, transform=None, frame_window=1,
                 seq_postfix='_dsfd_seq.pkl', postfixes=('.mp4',)):
        assert os.path.isdir(root), 'root must be path to a directory: "%s"' % root
        assert file_list is not None, 'initializing the dataset without a file list is not implemented'
        self.file_paths = parse_file_paths(root, file_list, seq_postfix, postfixes)
        assert len(self.file_paths) > 0, 'No files found'
        self.root = root
        self.transform = transform
        self.frame_window = frame_window

        if target_list is None:
            self.targets = None
            self.weights = np.ones(len(self.file_paths))
            self.classes = list(range(len(self.file_paths)))
        else:
            targets_list_path = target_list if os.path.exists(target_list) else os.path.join(root, target_list)
            if not os.path.exists(targets_list_path):
                raise (RuntimeError('Could not find target list file: ' + target_list))
            self.targets = np.loadtxt(targets_list_path, dtype='int64')
            self.classes = np.unique(self.targets)
            self.weights = calc_weights_for_balanced_classes(self.targets)

    def query(self, vid_index, seq_index, frame_index):
        """
        Args:
            vid_index (int): Index of the original video
            seq_index (int): Index of the video sequence
            frame_index (int): Index of the frame corresponding to the video sequence

        Returns:
            (np.array, ..., int (optional)): Tuple containing:
                - tuple of np.array: Sampled data corresponding to the specified postfixes
                - int, optional: The target corresponding to the original video if ``target_list`` was specified
        """
        target = self.targets[vid_index] if self.targets is not None else None
        all_seq_paths = self.file_paths[vid_index]
        seq_paths = all_seq_paths[seq_index]
        frame_index = [frame_index] if not isinstance(frame_index, (list, tuple)) else frame_index

        # For each sequence path
        data = []
        for seq_path in seq_paths:
            seq_queries = []
            if seq_path.endswith('.mp4'):
                # Open video
                vid = cv2.VideoCapture(seq_path)

                # For each frame index
                for fi in frame_index:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, fi)

                    # Read the frames from the video
                    frame_list = []
                    for i in range(self.frame_window):
                        ret, frame_bgr = vid.read()
                        assert frame_bgr is not None, 'Failed to read frame from video: "%s"' % seq_path
                        frame_rgb = frame_bgr[:, :, ::-1]
                        frame_list.append(frame_rgb)
                    seq_queries.append(frame_list if self.frame_window > 1 else frame_list[0])
            elif seq_path.endswith('_lms.npz'):
                landmarks = np.load(seq_path)['landmarks']
                for fi in frame_index:
                    landmarks_window = landmarks[fi:fi + self.frame_window]
                    seq_queries.append(landmarks_window if self.frame_window > 1 else landmarks_window[0])
            elif seq_path.endswith('_pose.npz'):
                poses = np.load(seq_path)['poses']
                for fi in frame_index:
                    poses_window = poses[fi:fi + self.frame_window]
                    seq_queries.append(poses_window if self.frame_window > 1 else poses_window[0])
            elif seq_path.endswith('_seg.pkl'):
                segmentations = np.load(seq_path, allow_pickle=True)
                for fi in frame_index:
                    segmentations_window = segmentations[fi:fi + self.frame_window]
                    segmentations_window = [decode_binary_mask(s) for s in segmentations_window]
                    seq_queries.append(segmentations_window if self.frame_window > 1 else segmentations_window[0])
            else:
                raise RuntimeError('Unknown file type: "%s"' % seq_path)
            data.append(seq_queries if len(frame_index) > 1 else seq_queries[0])

        # Apply transformation
        if self.transform is not None:
            data = self.transform(data)

        if target is None:
            return tuple(data)
        else:
            return tuple(data) + (target,)

    def __getitem__(self, index):
        """ Sample the dataset given a file index.

        Args:
            index (int): File index

        Returns:
            (np.array, ..., int (optional)): Tuple containing:
                - tuple of np.array: Sampled data corresponding to the specified postfixes
                - int, optional: The target corresponding to the original video if ``target_list`` was specified
        """
        assert index < len(self.file_paths), 'index out of range: [%d / %d]' % (index, len(self.file_paths))
        all_seq_paths = self.file_paths[index]
        seq_index = random.randint(0, len(all_seq_paths) - 1)
        seq_paths = all_seq_paths[seq_index]
        total_frames = get_total_frames_from_file(seq_paths[0])
        frame_index = random.randint(0, total_frames - self.frame_window)

        return self.query(index, seq_index, frame_index)

    def __len__(self):
        return len(self.file_paths)


class SeqPairDataset(SeqDataset):
    """ A dataset for loading video sequence pairs and their meta-data.

    Args:
        root (str): Root directory path
        file_list (str, optional): A path to a list of files. Can be relative to the root directory
        target_list (str, optional): A path to a list of targets corresponding to to each of the files in
            ``file_list``. Can be relative to the root directory
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        frame_window (int): The size of the temporal frame window to load for each query. If greater than one,
            an additional temporal dimension of the same size will be added for each returned tensor
        seq_postfix (str): Sequence cache file postfix
        postfixes (list of str): The postfixes of the sequence files to load. The order of the postifixes will
            determine the order of the corresponding tensors that will be returned. By default only the cropped video
            sequence files will be loaded
        same_prob (float): The probability to return both samples from the same video sequence
        return_target (bool): If True, return the target corresponding to the original videos as well

    Attributes:
        file_paths (list of str): List of the parsed file paths according to the specified postfixes
        targets (np.array of int) The targets corresponding to each of the files in ``file_paths``
        classes (list of int) The classes corresponding to each of the files in ``file_paths``. The classes will be
            automatically determined from the targets if they are specified, otherwise they will be the file indices
        weights (np.array): Per file weights determined by their targets. Used for balanced training
    """
    def __init__(self, root, file_list=None, target_list=None, transform=None, frame_window=1,
                 seq_postfix='_dsfd_seq.pkl', postfixes=('.mp4',), same_prob=0.5, return_target=True):
        super(SeqPairDataset, self).__init__(root, file_list, target_list, transform, frame_window, seq_postfix,
                                             postfixes)
        self.same_prob = same_prob
        self.return_target = return_target

    def __getitem__(self, index):
        """ Sample the dataset given a file index.

        Args:
            index (int): File index

        Returns:
            (np.array, ..., int (optional)): Tuple containing:
                - tuple of np.array: Sampled data corresponding to the specified postfixes
                - int, optional: The target corresponding to the original video if ``target_list`` was specified
        """
        assert index < len(self.file_paths), 'index out of range: [%d / %d]' % (index, len(self.file_paths))
        if random.random() < self.same_prob:
            # Same identity
            same = True
            all_seq_paths = self.file_paths[index]
            seq_index = random.randint(0, len(all_seq_paths) - 1)
            seq_paths = all_seq_paths[seq_index]
            total_frames = get_total_frames_from_file(seq_paths[0])

            # Randomly select a pair of non intersecting frame windows
            frame_pair_indices = random_pair_range(0, total_frames - 1, self.frame_window)

            data = self.query(index, seq_index, frame_pair_indices)
        else:
            # Different identity
            same = False
            pair_indices = random_pair(len(self.file_paths), index1=index)
            queries = []
            for pair_index in pair_indices:
                queries.append(super(SeqPairDataset, self).__getitem__(pair_index))
            data = tuple(zip(*queries))

        if self.return_target:
            return data + (np.array(same, dtype='float32'),)
        else:
            return data


class SeqTripletDataset(SeqDataset):
    """ A dataset for loading video sequence triplets and their meta-data.

    For each triplet, the first two samples are always from the video sequence while the third from a different one.

    Args:
        root (str): Root directory path
        file_list (str, optional): A path to a list of files. Can be relative to the root directory
        target_list (str, optional): A path to a list of targets corresponding to to each of the files in
            ``file_list``. Can be relative to the root directory
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        frame_window (int): The size of the temporal frame window to load for each query. If greater than one,
            an additional temporal dimension of the same size will be added for each returned tensor
        seq_postfix (str): Sequence cache file postfix
        postfixes (list of str): The postfixes of the sequence files to load. The order of the postifixes will
            determine the order of the corresponding tensors that will be returned. By default only the cropped video
            sequence files will be loaded

    Attributes:
        file_paths (list of str): List of the parsed file paths according to the specified postfixes
        targets (np.array of int) The targets corresponding to each of the files in ``file_paths``
        classes (list of int) The classes corresponding to each of the files in ``file_paths``. The classes will be
            automatically determined from the targets if they are specified, otherwise they will be the file indices
        weights (np.array): Per file weights determined by their targets. Used for balanced training
    """
    def __init__(self, root, file_list=None, target_list=None, transform=None, frame_window=1,
                 seq_postfix='_dsfd_seq.pkl', postfixes=('.mp4',)):
        super(SeqTripletDataset, self).__init__(root, file_list, target_list, transform, frame_window, seq_postfix,
                                                postfixes)

    def __getitem__(self, index):
        """ Sample the dataset given a file index.

        Args:
            index (int): File index

        Returns:
            (np.array, ..., int (optional)): Tuple containing:
                - tuple of np.array: Sampled data corresponding to the specified postfixes
                - int, optional: The target corresponding to the original video if ``target_list`` was specified
        """
        assert index < len(self.file_paths), 'index out of range: [%d / %d]' % (index, len(self.file_paths))
        file_pair_indices = random_pair(len(self.file_paths), index1=index)

        # Load pair of the same identity
        same_paths = self.file_paths[file_pair_indices[0]]
        seq_index = random.randint(0, len(same_paths) - 1)
        seq_paths = same_paths[seq_index]
        total_frames = get_total_frames_from_file(seq_paths[0])

        # Randomly select a pair of non intersecting frame windows
        frame_pair_indices = random_pair_range(0, total_frames - 1, self.frame_window)
        data = self.query(index, seq_index, frame_pair_indices)

        # Load different identity
        data.append(super(SeqTripletDataset, self).__getitem__(file_pair_indices[1]))

        return tuple(data)


class SeqInferenceDataset(data.Dataset):
    """ A dataset for loading a single cropped video sequence and its meta-data without random access.

    Args:
        vid_path (str): Path to a cropped video
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        postfixes (list of str): The postfixes of the sequence files to load (not including the cropped video sequence).
            The order of the postifixes will determine the order of the corresponding tensors that will be returned.
            By default only the cropped video sequence file will be loaded

    Attributes:
        cap (cv2.VideoCapture): OpenCV's video capture object
        data (list): List of preloaded meta-data
    """
    def __init__(self, vid_path, transform=None, postfixes=None):
        self.vid_path = vid_path
        self.transform = transform
        self.cap = None

        # Get video info
        self.width, self.height, self.total_frames, self.fps = get_video_info(vid_path)

        # Additional data
        self.data = []
        vid_path_no_ext = os.path.splitext(vid_path)[0]
        if postfixes is not None:
            for postfix in postfixes:
                if postfix == '_lms.npz':
                    self.data.append(np.load(vid_path_no_ext + postfix)['landmarks_smoothed'])
                elif postfix == '_pose.npz':
                    self.data.append(np.load(vid_path_no_ext + postfix)['poses_smoothed'])
                elif postfix == '_seg.pkl':
                    self.data.append(np.load(vid_path_no_ext + postfix, allow_pickle=True))
                else:
                    raise RuntimeError('Unknown postfix: "%s"' % postfix)

    def __getitem__(self, index):
        """ Sample the dataset at the given index.

        Args:
            index (int): Frame index

        Returns:
            Tuple of np.array: A tuple containing the current video frame and additional meta-data in the order
                specified by postfixes.
        """
        if self.cap is None:
            # Open video file
            self.cap = cv2.VideoCapture(self.vid_path)

        ret, frame_bgr = self.cap.read()
        assert frame_bgr is not None, 'Failed to read frame from video in index: %d' % index
        frame_rgb = frame_bgr[:, :, ::-1]

        # Add additional data
        data = [frame_rgb]
        if len(self.data) > 0:
            for d in self.data:
                if isinstance(d[index], bytes):
                    data.append(decode_binary_mask(d[index]))
                else:
                    data.append(d[index])
            # data += [d[index] for d in self.data]

        # Apply transformation
        if self.transform is not None:
            data = self.transform(data)

        return tuple(data) if len(data) > 1 else data[0]

    def __len__(self):
        return self.total_frames


class SingleSeqRandomDataset(data.Dataset):
    """ A dataset for loading a single cropped video sequence and its meta-data.

    Args:
        vid_path (str): Path to a cropped video
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        postfixes (list of str): The postfixes of the sequence files to load (not including the cropped video sequence).
            The order of the postifixes will determine the order of the corresponding tensors that will be returned.
            By default only the cropped video sequence file will be loaded

    Attributes:
        cap (cv2.VideoCapture): OpenCV's video capture object
        data (list): List of preloaded meta-data
    """

    def __init__(self, vid_path, transform=None, postfixes=None):
        self.vid_path = vid_path
        self.transform = transform
        self.cap = {}

        # Get video info
        self.width, self.height, self.total_frames, self.fps = get_video_info(vid_path)

        # Additional data
        self.data = []
        vid_path_no_ext = os.path.splitext(vid_path)[0]
        if postfixes is not None:
            for postfix in postfixes:
                if postfix == '_lms.npz':
                    self.data.append(np.load(vid_path_no_ext + postfix)['landmarks'])
                elif postfix == '_pose.npz':
                    self.data.append(np.load(vid_path_no_ext + postfix)['poses'])
                else:
                    raise RuntimeError('Unknown postfix: "%s"' % postfix)

    def __getitem__(self, index):
        """ Sample the dataset at the given index.

        Args:
            index (int): Frame index

        Returns:
            Tuple of np.array: A tuple containing the current video frame and additional meta-data in the order
                specified by postfixes.
        """
        pid = os.getpid()
        if pid not in self.cap:
            # Open video file
            self.cap[pid] = cv2.VideoCapture(self.vid_path)
        self.cap[pid].set(cv2.CAP_PROP_POS_FRAMES, index)

        ret, frame_bgr = self.cap[pid].read()
        assert frame_bgr is not None, 'Failed to read frame from video in index: %d' % index
        frame_rgb = frame_bgr[:, :, ::-1]

        # Add additional data
        data = [frame_rgb]
        if len(self.data) > 0:
            data += [d[index] for d in self.data]

        # Apply transformation
        if self.transform is not None:
            data = self.transform(data)

        return tuple(data)

    def __len__(self):
        return self.total_frames


class SingleSeqRandomPairDataset(SingleSeqRandomDataset):
    """ A dataset for loading random pairs of frames from a single cropped video sequence and its meta-data.

    Args:
        vid_path (str): Path to a cropped video
        transform (callable, optional): A function/transform that takes in a numpy image and returns
            a transformed version. E.g, ``img_lms_pose_transforms.Crop``
        postfixes (list of str): The postfixes of the sequence files to load (not including the cropped video sequence).
            The order of the postifixes will determine the order of the corresponding tensors that will be returned.
            By default only the cropped video sequence file will be loaded

    Attributes:
        cap (cv2.VideoCapture): OpenCV's video capture object
        data (list): List of preloaded meta-data
    """
    def __init__(self, vid_path, transform=None, postfixes=None):
        super(SingleSeqRandomPairDataset, self).__init__(vid_path, transform, postfixes)

    def __getitem__(self, index):
        """ Sample the dataset at the given index.

        Args:
            index (int): Frame index

        Returns:
            Tuple of (np.array, np.array): A tuple containing the current video frame pairs and additional meta-data
                in the order specified by postfixes.
        """
        indices = random_pair(self.total_frames, index1=index)
        data = zip(*[super(SingleSeqRandomPairDataset, self).__getitem__(i) for i in indices])

        return tuple(data)


def main(dataset='fsgan.datasets.seq_dataset.SeqDataset', np_transforms=None,
         tensor_transforms=('img_lms_pose_transforms.ToTensor()', 'img_lms_pose_transforms.Normalize()'),
         workers=4, batch_size=4):
    import time
    import fsgan
    from fsgan.utils.obj_factory import obj_factory
    from fsgan.utils.img_utils import tensor2bgr

    np_transforms = obj_factory(np_transforms) if np_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    img_transforms = img_lms_pose_transforms.Compose(np_transforms + tensor_transforms)
    dataset = obj_factory(dataset, transform=img_transforms)
    # dataset = VideoSeqDataset(root_path, img_list_path, transform=img_transforms, frame_window=frame_window)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, drop_last=True,
                                 shuffle=True)

    start = time.time()
    if isinstance(dataset, fsgan.datasets.seq_dataset.SeqPairDataset):
        for frame, landmarks, pose, target in dataloader:
            pass
    elif isinstance(dataset, fsgan.datasets.seq_dataset.SeqDataset):
        for frame, landmarks, pose in dataloader:
            # For each batch
            for b in range(frame.shape[0]):
                # Render
                render_img = tensor2bgr(frame[b]).copy()
                curr_landmarks = landmarks[b].numpy() * render_img.shape[0]
                curr_pose = pose[b].numpy() * 99.

                for point in curr_landmarks:
                    cv2.circle(render_img, (point[0], point[1]), 2, (0, 0, 255), -1)
                msg = 'Pose: %.1f, %.1f, %.1f' % (curr_pose[0], curr_pose[1], curr_pose[2])
                cv2.putText(render_img, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('render_img', render_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break


        # print(frame_window.shape)

        # if isinstance(frame_window, (list, tuple)):
        #     # For each batch
        #     for b in range(frame_window[0].shape[0]):
        #         # For each frame window in the list
        #         for p in range(len(frame_window)):
        #             # For each frame in the window
        #             for f in range(frame_window[p].shape[2]):
        #                 print(frame_window[p][b, :, f, :, :].shape)
        #                 # Render
        #                 render_img = tensor2bgr(frame_window[p][b, :, f, :, :]).copy()
        #                 landmarks = landmarks_window[p][b, f, :, :].numpy()
        #                 # for point in np.round(landmarks).astype(int):
        #                 for point in landmarks:
        #                     cv2.circle(render_img, (point[0], point[1]), 2, (0, 0, 255), -1)
        #                 cv2.imshow('render_img', render_img)
        #                 if cv2.waitKey(0) & 0xFF == ord('q'):
        #                     break
        # else:
        #     # For each batch
        #     for b in range(frame_window.shape[0]):
        #         # For each frame in the window
        #         for f in range(frame_window.shape[2]):
        #             print(frame_window[b, :, f, :, :].shape)
        #             # Render
        #             render_img = tensor2bgr(frame_window[b, :, f, :, :]).copy()
        #             landmarks = landmarks_window[b, f, :, :].numpy()
        #             # for point in np.round(landmarks).astype(int):
        #             for point in landmarks:
        #                 cv2.circle(render_img, (point[0], point[1]), 2, (0, 0, 255), -1)
        #             cv2.imshow('render_img', render_img)
        #             if cv2.waitKey(0) & 0xFF == ord('q'):
        #                 break
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
                        default=('img_lms_pose_transforms.ToTensor()', 'img_lms_pose_transforms.Normalize()'))
    parser.add_argument('-w', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    main(**vars(parser.parse_args()))
