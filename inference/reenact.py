""" Face reenactment inference pipeline.

This script implements face reenactment for both images and videos using an appearance map for the source subject.

Information about both source and target files will be extracted and cached in directories by the file's name without
the extension, residing in the same directory as the file. The information contains: face detections, face sequences,
and cropped videos per sequence. In addition for each cropped video, the corresponding pose, landmarks, and
segmentation masks will be computed and cached.
"""

import os
import argparse
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler, DataLoader
from fsgan.preprocess.preprocess_video import VideoProcessBase, base_parser
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.utils import load_model
from fsgan.utils.img_utils import tensor2bgr
from fsgan.utils.landmarks_utils import LandmarksHeatMapDecoder
from fsgan.datasets.img_lms_pose_transforms import RandomHorizontalFlip, Rotate, Pyramids, ToTensor, Normalize
from fsgan.datasets import img_lms_pose_transforms
from fsgan.datasets.seq_dataset import SingleSeqRandomPairDataset
from fsgan.datasets.appearance_map import AppearanceMapDataset
from fsgan.utils.video_renderer import VideoRenderer
from fsgan.utils.batch import main as batch


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 parents=[base_parser])
parser.add_argument('source', metavar='SOURCE', nargs='+',
                    help='image or video per source: files, directories, file lists or queries')
parser.add_argument('-t', '--target', metavar='TARGET', nargs='+',
                    help='video per target: files, directories, file lists or queries')
parser.add_argument('-o', '--output', metavar='DIR',
                    help='output directory')
parser.add_argument('-ss', '--select_source', default='longest', metavar='STR',
                    help='source selection method ["longest" | sequence number]')
parser.add_argument('-st', '--select_target', default='longest', metavar='STR',
                    help='target selection method ["longest" | sequence number]')
parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('-rm', '--reenactment_model', metavar='PATH',
                    default='../weights/nfv_msrunet_256_1_2_reenactment_v2.1.pth', help='reenactment model')
parser.add_argument('-ci', '--criterion_id', default="vgg_loss.VGGLoss('../weights/vggface2_vgg19_256_1_2_id.pth')",
                    metavar='OBJ', help='id criterion object')
parser.add_argument('-mr', '--min_radius', default=2.0, type=float, metavar='F',
                    help='minimum distance between points in the appearance map')
parser.add_argument('-rp', '--renderer_process', action='store_true',
                    help='If True, the renderer will be run in a separate process')

finetune = parser.add_argument_group('finetune')
finetune.add_argument('-f', '--finetune', action='store_true',
                      help='Toggle whether to finetune the reenactment generator (default: False)')
finetune.add_argument('-fi', '--finetune_iterations', default=800, type=int, metavar='N',
                      help='number of finetune iterations')
finetune.add_argument('-fl', '--finetune_lr', default=1e-4, type=float, metavar='F',
                      help='finetune learning rate')
finetune.add_argument('-fb', '--finetune_batch_size', default=4, type=int, metavar='N',
                      help='finetune batch size')
finetune.add_argument('-fw', '--finetune_workers', default=4, type=int, metavar='N',
                      help='finetune workers')
finetune.add_argument('-fs', '--finetune_save', action='store_true',
                      help='enable saving finetune checkpoint')
d = parser.get_default


class FaceReenactment(VideoProcessBase):
    def __init__(self, resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
        cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
        # Detection arguments:
        detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
        # Sequence arguments:
        iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
        center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
        seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
        # Pose arguments:
        pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
        cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
        # Landmarks arguments:
        lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
        cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
        # Segmentation arguments:
        seg_model=d('seg_model'), smooth_segmentation=d('smooth_segmentation'),
        segmentation_postfix=d('segmentation_postfix'), cache_segmentation=d('cache_segmentation'),
        seg_batch_size=d('seg_batch_size'), seg_remove_mouth=d('seg_remove_mouth'),
        # Finetune arguments:
        finetune=d('finetune'), finetune_iterations=d('finetune_iterations'), finetune_lr=d('finetune_lr'),
        finetune_batch_size=d('finetune_batch_size'), finetune_workers=d('finetune_workers'),
        finetune_save=d('finetune_save'),
        # Reenactment arguments:
        batch_size=d('batch_size'), reenactment_model=d('reenactment_model'), criterion_id=d('criterion_id'),
        min_radius=d('min_radius'), renderer_process=d('renderer_process')):
        super(FaceReenactment, self).__init__(
            resolution, crop_scale, gpus, cpu_only, display, verbose, encoder_codec,
            detection_model=detection_model, det_batch_size=det_batch_size, det_postfix=det_postfix,
            iou_thresh=iou_thresh, min_length=min_length, min_size=min_size, center_kernel=center_kernel,
            size_kernel=size_kernel, smooth_det=smooth_det, seq_postfix=seq_postfix, write_empty=write_empty,
            pose_model=pose_model, pose_batch_size=pose_batch_size, pose_postfix=pose_postfix,
            cache_pose=True, cache_frontal=cache_frontal, smooth_poses=smooth_poses,
            lms_model=lms_model, lms_batch_size=lms_batch_size, landmarks_postfix=landmarks_postfix,
            cache_landmarks=True, smooth_landmarks=smooth_landmarks, seg_model=seg_model,
            seg_batch_size=seg_batch_size, segmentation_postfix=segmentation_postfix,
            cache_segmentation=True, smooth_segmentation=smooth_segmentation,
            seg_remove_mouth=seg_remove_mouth)
        self.batch_size = batch_size
        self.min_radius = min_radius
        self.finetune_enabled = finetune
        self.finetune_iterations = finetune_iterations
        self.finetune_lr = finetune_lr
        self.finetune_batch_size = finetune_batch_size
        self.finetune_workers = finetune_workers
        self.finetune_save = finetune_save

        # Load reenactment model
        self.Gr, checkpoint = load_model(reenactment_model, 'face reenactment', self.device, return_checkpoint=True)
        self.Gr.arch = checkpoint['arch']
        self.reenactment_state_dict = checkpoint['state_dict']

        # Initialize landmarks decoders
        self.landmarks_decoders = []
        for res in (128, 256):
            self.landmarks_decoders.insert(0, LandmarksHeatMapDecoder(res).to(self.device))

        # Initialize losses
        self.criterion_pixelwise = nn.L1Loss().to(self.device)
        self.criterion_id = obj_factory(criterion_id).to(self.device)

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.Gr = nn.DataParallel(self.Gr, self.gpus)
            self.criterion_id.vgg = nn.DataParallel(self.criterion_id.vgg, self.gpus)

        # Initialize video renderer
        self.video_renderer = FaceReenactmentRenderer(self.display, self.verbose, True, self.resolution,
                                                      self.crop_scale, encoder_codec, renderer_process)
        self.video_renderer.start()

    def __del__(self):
        if hasattr(self, 'video_renderer'):
            self.video_renderer.kill()

    def finetune(self, source_path, save_checkpoint=True):
        checkpoint_path = os.path.splitext(source_path)[0] + '_Gr.pth'
        if os.path.isfile(checkpoint_path):
            print('=> Loading the reenactment generator finetuned on: "%s"...' % os.path.basename(source_path))
            checkpoint = torch.load(checkpoint_path)
            if self.gpus and len(self.gpus) > 1:
                self.Gr.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.Gr.load_state_dict(checkpoint['state_dict'])
            return

        print('=> Finetuning the reenactment generator on: "%s"...' % os.path.basename(source_path))
        torch.set_grad_enabled(True)
        self.Gr.train(True)
        img_transforms = img_lms_pose_transforms.Compose([Pyramids(2), ToTensor(), Normalize()])
        train_dataset = SingleSeqRandomPairDataset(source_path, transform=img_transforms, postfixes=('_lms.npz',))
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.finetune_iterations)
        train_loader = DataLoader(train_dataset, batch_size=self.finetune_batch_size, sampler=train_sampler,
                                  num_workers=self.finetune_workers, pin_memory=True, drop_last=True, shuffle=False)
        optimizer = optim.Adam(self.Gr.parameters(), lr=self.finetune_lr, betas=(0.5, 0.999))

        # For each batch in the training data
        for i, (img, landmarks) in enumerate(tqdm(train_loader, unit='batches', file=sys.stdout)):
            # Prepare input
            with torch.no_grad():
                # For each view images and landmarks
                landmarks[1] = landmarks[1].to(self.device)
                for j in range(len(img)):
                    # For each pyramid image: push to device
                    for p in range(len(img[j])):
                        img[j][p] = img[j][p].to(self.device)

                # Concatenate pyramid images with context to derive the final input
                input = []
                for p in range(len(img[0])):
                    context = self.landmarks_decoders[p](landmarks[1])
                    input.append(torch.cat((img[0][p], context), dim=1))

            # Reenactment
            img_pred = self.Gr(input)

            # Reconstruction loss
            loss_pixelwise = self.criterion_pixelwise(img_pred, img[1][0])
            loss_id = self.criterion_id(img_pred, img[1][0])
            loss_rec = 0.1 * loss_pixelwise + loss_id

            # Update generator weights
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # Save finetuned weights to file
        if save_checkpoint:
            arch = self.Gr.module.arch if self.gpus and len(self.gpus) > 1 else self.Gr.arch
            state_dict = self.Gr.module.state_dict() if self.gpus and len(self.gpus) > 1 else self.Gr.state_dict()
            torch.save({'state_dict': state_dict, 'arch': arch}, checkpoint_path)

        torch.set_grad_enabled(False)
        self.Gr.train(False)

    def __call__(self, source_path, target_path, output_path=None, select_source='longest', select_target='longest',
                 finetune=None):
        is_vid = os.path.splitext(source_path)[1] == '.mp4'
        finetune = self.finetune_enabled and is_vid if finetune is None else finetune and is_vid

        # Validation
        assert os.path.isfile(source_path), 'Source path "%s" does not exist' % source_path
        assert os.path.isfile(target_path), 'Target path "%s" does not exist' % target_path

        # Cache input
        source_cache_dir, source_seq_file_path, _ = self.cache(source_path)
        target_cache_dir, target_seq_file_path, _ = self.cache(target_path)

        # Load sequences from file
        with open(source_seq_file_path, "rb") as fp:  # Unpickling
            source_seq_list = pickle.load(fp)
        with open(target_seq_file_path, "rb") as fp:  # Unpickling
            target_seq_list = pickle.load(fp)

        # Select source and target sequence
        source_seq = select_seq(source_seq_list, select_source)
        target_seq = select_seq(target_seq_list, select_target)

        # Set source and target sequence videos paths
        src_path_no_ext, src_ext = os.path.splitext(source_path)
        src_vid_seq_name = os.path.basename(src_path_no_ext) + '_seq%02d%s' % (source_seq.id, src_ext)
        src_vid_seq_path = os.path.join(source_cache_dir, src_vid_seq_name)
        tgt_path_no_ext, tgt_ext = os.path.splitext(target_path)
        tgt_vid_seq_name = os.path.basename(tgt_path_no_ext) + '_seq%02d%s' % (target_seq.id, tgt_ext)
        tgt_vid_seq_path = os.path.join(target_cache_dir, tgt_vid_seq_name)

        # Set output path
        if output_path is not None:
            if os.path.isdir(output_path):
                output_filename = f'{os.path.basename(src_path_no_ext)}_{os.path.basename(tgt_path_no_ext)}.mp4'
                output_path = os.path.join(output_path, output_filename)

        # Initialize appearance map
        src_transform = img_lms_pose_transforms.Compose([Rotate(), Pyramids(2), ToTensor(), Normalize()])
        tgt_transform = img_lms_pose_transforms.Compose([ToTensor(), Normalize()])
        appearance_map = AppearanceMapDataset(src_vid_seq_path, tgt_vid_seq_path, src_transform, tgt_transform,
                                              self.landmarks_postfix, self.pose_postfix, self.segmentation_postfix,
                                              self.min_radius)
        appearance_map_loader = DataLoader(appearance_map, batch_size=self.batch_size, num_workers=1, pin_memory=True,
                                           drop_last=False, shuffle=False)

        # Initialize video renderer
        self.video_renderer.init(target_path, target_seq, output_path, _appearance_map=appearance_map)

        # Finetune reenactment model on source sequences
        if finetune:
            self.finetune(src_vid_seq_path, self.finetune_save)

        print(f'=> Face reenactment: "{src_vid_seq_name}" -> "{tgt_vid_seq_name}"...')

        # For each batch of frames in the target video
        for i, (src_frame, src_landmarks, src_poses, bw, tgt_frame, tgt_landmarks, tgt_pose, tgt_mask) \
                in enumerate(tqdm(appearance_map_loader, unit='batches', file=sys.stdout)):
            # Prepare input
            for p in range(len(src_frame)):
                src_frame[p] = src_frame[p].to(self.device)
            tgt_landmarks = tgt_landmarks.to(self.device)
            bw = bw.to(self.device)
            bw_indices = torch.nonzero(torch.any(bw > 0, dim=0), as_tuple=True)[0]
            bw = bw[:, bw_indices]

            # For each source frame perform reenactment
            reenactment_triplet = []
            for j in bw_indices:
                input = []
                for p in range(len(src_frame)):
                    context = self.landmarks_decoders[p](tgt_landmarks)
                    input.append(torch.cat((src_frame[p][:, j], context), dim=1))

                # Reenactment
                reenactment_triplet.append(self.Gr(input).unsqueeze(1))
            reenactment_tensor = torch.cat(reenactment_triplet, dim=1)

            # Barycentric interpolation of reenacted frames
            reenactment_tensor = (reenactment_tensor * bw.view(*bw.shape, 1, 1, 1)).sum(dim=1)

            # Write output
            if self.verbose == 0:
                self.video_renderer.write(reenactment_tensor)
            elif self.verbose == 1:
                self.video_renderer.write(src_frame[0][:, 0], src_frame[0][:, 1], src_frame[0][:, 2],
                                          reenactment_tensor, tgt_frame)
            else:
                self.video_renderer.write(src_frame[0][:, 0], src_frame[0][:, 1], src_frame[0][:, 2],
                                          reenactment_tensor, tgt_frame, tgt_pose)

        # Load original reenactment weights
        if finetune:
            if self.gpus and len(self.gpus) > 1:
                self.Gr.module.load_state_dict(self.reenactment_state_dict)
            else:
                self.Gr.load_state_dict(self.reenactment_state_dict)

        # Wait for the video render to finish rendering
        self.video_renderer.finalize()
        self.video_renderer.wait_until_finished()


class FaceReenactmentRenderer(VideoRenderer):
    def __init__(self, display=False, verbose=0, output_crop=False, resolution=256, crop_scale=1.2,
                 encoder_codec='avc1', separate_process=False):
        self._appearance_map = None
        self._fig = None
        self._figsize = (24, 16)

        # Calculate verbose size
        verbose_size, self._appearance_map_size = None, None
        if verbose == 1:
            verbose_size = (resolution * 5, resolution)
        elif verbose >= 2:
            fig_ratio = self._figsize[0] / self._figsize[1]
            height = 5 * resolution
            self._appearance_map_size = (int(np.round(height * fig_ratio)), height)
            verbose_size = (self._appearance_map_size[0] + resolution, self._appearance_map_size[1])

        super(FaceReenactmentRenderer, self).__init__(display, verbose, verbose_size, output_crop, resolution,
                                                      crop_scale, encoder_codec, separate_process)

    def on_render(self, *args):
        if self._verbose <= 0:
            return tensor2bgr(args[0])
        elif self._verbose == 1:
            return tensor2bgr(torch.cat(args, dim=2))
        else:
            if self._fig is None:
                self._fig = plt.figure(figsize=self._figsize)
            results_bgr = tensor2bgr(torch.cat(args[:5], dim=1))
            tgt_pose = args[5].numpy()
            appearance_map_bgr = render_appearance_map(self._fig, self._appearance_map.tri, self._appearance_map.points,
                                                       tgt_pose[:2])
            appearance_map_bgr = cv2.resize(appearance_map_bgr, self._appearance_map_size,
                                            interpolation=cv2.INTER_CUBIC)
            render_bgr = np.concatenate((appearance_map_bgr, results_bgr), axis=1)
            tgt_pose *= 99.     # Unnormalize the target pose for printing
            msg = 'Pose: %.1f, %.1f, %.1f' % (tgt_pose[0], tgt_pose[1], tgt_pose[2])
            cv2.putText(render_bgr, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            return render_bgr


def render_appearance_map(fig, tri, points, query_point=None, render_scale=99.):
    points_scaled = points * render_scale
    plt.triplot(points_scaled[:, 0], points_scaled[:, 1], tri.simplices.copy(), linewidth=3)
    plt.plot(points_scaled[:, 0], points_scaled[:, 1], 'o', markersize=12)
    if query_point is not None:
        query_point_scaled = query_point[:2] * render_scale
        tri_index = tri.find_simplex(query_point[:2])
        tri_vertices = tri.simplices[tri_index]
        plt.plot(points_scaled[tri_vertices, 0], points_scaled[tri_vertices, 1], 'yo', markersize=12)
        plt.plot(query_point_scaled[0], query_point_scaled[1], 'rx', markersize=24, markeredgewidth=4)

    plt.xlim(points_scaled[:-4, 0].min() - 0.5, points_scaled[:-4, 0].max() + 0.5)
    plt.ylim(points_scaled[:-4, 1].min() - 0.5, points_scaled[:-4, 1].max() + 0.5)
    plt.xlabel('Yaw (angles)', fontsize=24)
    plt.ylabel('Pitch (angles)', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.tight_layout()
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    img = np.array(fig.canvas.renderer._renderer)
    plt.clf()

    return img[:, :, 2::-1]


def select_seq(seq_list, select='longest'):
    if select == 'longest':
        seq = seq_list[np.argmax([len(s) for s in seq_list])]
    elif select.isnumeric():
        seq = seq_list[[s.id for s in seq_list].index(int(select))]
    else:
        raise RuntimeError(f'Unknown selection method: "{select}"')

    return seq


def main(source, target, output=None, select_source=d('select_source'), select_target=d('select_target'),
         # General arguments
         resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
         cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
         # Detection arguments:
         detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
         # Sequence arguments:
         iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
         center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
         seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
         # Pose arguments:
         pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
         cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
         # Landmarks arguments:
         lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
         cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
         # Segmentation arguments:
         seg_model=d('seg_model'), seg_batch_size=d('seg_batch_size'), segmentation_postfix=d('segmentation_postfix'),
         cache_segmentation=d('cache_segmentation'), smooth_segmentation=d('smooth_segmentation'),
         seg_remove_mouth=d('seg_remove_mouth'),
         # Finetune arguments:
         finetune=d('finetune'), finetune_iterations=d('finetune_iterations'), finetune_lr=d('finetune_lr'),
         finetune_batch_size=d('finetune_batch_size'), finetune_workers=d('finetune_workers'),
         finetune_save=d('finetune_save'),
         # Reenactment arguments:
         batch_size=d('batch_size'), reenactment_model=d('reenactment_model'), criterion_id=d('criterion_id'),
         min_radius=d('min_radius'), renderer_process=d('renderer_process')):
    face_reenactment = FaceReenactment(
        resolution, crop_scale, gpus, cpu_only, display, verbose, encoder_codec,
        detection_model=detection_model, det_batch_size=det_batch_size, det_postfix=det_postfix,
        iou_thresh=iou_thresh, min_length=min_length, min_size=min_size, center_kernel=center_kernel,
        size_kernel=size_kernel, smooth_det=smooth_det, seq_postfix=seq_postfix, write_empty=write_empty,
        pose_model=pose_model, pose_batch_size=pose_batch_size, pose_postfix=pose_postfix,
        cache_pose=cache_pose, cache_frontal=cache_frontal, smooth_poses=smooth_poses,
        lms_model=lms_model, lms_batch_size=lms_batch_size, landmarks_postfix=landmarks_postfix,
        cache_landmarks=cache_landmarks, smooth_landmarks=smooth_landmarks,
        seg_model=seg_model, seg_batch_size=seg_batch_size, segmentation_postfix=segmentation_postfix,
        cache_segmentation=cache_segmentation, smooth_segmentation=smooth_segmentation,
        seg_remove_mouth=seg_remove_mouth,
        finetune=finetune, finetune_iterations=finetune_iterations, finetune_lr=finetune_lr,
        finetune_batch_size=finetune_batch_size, finetune_workers=finetune_workers, finetune_save=finetune_save,
        batch_size=batch_size, reenactment_model=reenactment_model, criterion_id=criterion_id, min_radius=min_radius,
        renderer_process=renderer_process)
    if len(source) == 1 and len(target) == 1 and os.path.isfile(source[0]) and os.path.isfile(target[0]):
        face_reenactment(source[0], target[0], output, select_source, select_target)
    else:
        batch(source, target, output, face_reenactment, postfix='.mp4', skip_existing=True)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
