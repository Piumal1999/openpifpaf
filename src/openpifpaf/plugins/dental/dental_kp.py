"""
Interface for custom data.

This module handles datasets and is the class that you need to inherit from for your custom dataset.
This class gives you all the handles so that you can train with a new –dataset=mydataset.
The particular configuration of keypoints and skeleton is specified in the headmeta instances
"""

import argparse
from typing import List, Optional, Tuple

import torch
import torch.utils.data
import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

import openpifpaf
import openpifpaf.transforms

from .constants import get_constants
from .metrics import MeanPixelError
from ..coco.dataset import CocoDataset


class DentalKp(openpifpaf.datasets.DataModule):
    """
    DataModule for the Dental Dataset.
    """
    debug = False
    pin_memory = False

    train_annotations = 'data-dental/annotations/dental_keypoints_66_train.json'
    val_annotations = 'data-dental/annotations/dental_keypoints_66_val.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-dental/images/train/'
    val_image_dir = 'data-dental/images/val/'
    eval_image_dir = val_image_dir

    n_images = None
    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    # car-specific configuration
    hflip = None
    weights: Optional[List[float]] = None
    dental_categories = None
    dental_keypoints = None
    dental_skeleton: Optional[List[Tuple[int, int]]] = None
    dental_sigmas = None
    dental_pose = None
    dental_score_weights = None

    def __init__(self):
        super().__init__()

        assert self.dental_keypoints is not None
        assert self.dental_sigmas is not None
        assert self.dental_skeleton is not None

        if self.weights is not None:
            caf_weights = []
            for bone1, bone2 in self.dental_skeleton:  # pylint: disable=not-an-iterable
                caf_weights.append(max(self.weights[bone1 - 1],
                                       self.weights[bone2 - 1]))
            w_np = np.array(caf_weights)
            caf_weights = list(w_np / np.sum(w_np) * len(caf_weights))
        else:
            caf_weights = None

        cif = openpifpaf.headmeta.Cif('cif', 'dental',
                                      keypoints=self.dental_keypoints,
                                      sigmas=self.dental_sigmas,
                                      pose=self.dental_pose,
                                      draw_skeleton=self.dental_skeleton,
                                      score_weights=self.dental_score_weights,
                                      training_weights=self.weights)
        caf = openpifpaf.headmeta.Caf('caf', 'dental',
                                      keypoints=self.dental_keypoints,
                                      sigmas=self.dental_sigmas,
                                      pose=self.dental_pose,
                                      skeleton=self.dental_skeleton,
                                      training_weights=caf_weights)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Dental')

        group.add_argument('--dental-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--dental-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--dental-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--dental-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--dental-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--dental-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--dental-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--dental-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--dental-no-augmentation',
                           dest='dental_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--dental-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--dental-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--dental-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--dental-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')

        # evaluation
        assert cls.eval_annotation_filter
        group.add_argument('--dental-no-eval-annotation-filter',
                           dest='dental_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--dental-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--dental-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--dental-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)
        group.add_argument('--dental-use-24-kps', default=False, action='store_true',
                           help=('The Dental dataset can '
                                 'be trained with 24 or 66 kps. If you want to train a model '
                                 'with 24 kps activate this flag. Change the annotations '
                                 'path to the json files with 24 kps.'))

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # Apollo specific
        cls.train_annotations = args.dental_train_annotations
        cls.val_annotations = args.dental_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.dental_train_image_dir
        cls.val_image_dir = args.dental_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.dental_square_edge
        cls.extended_scale = args.dental_extended_scale
        cls.orientation_invariant = args.dental_orientation_invariant
        cls.blur = args.dental_blur
        cls.augmentation = args.dental_augmentation  # loaded by the dest name
        cls.rescale_images = args.dental_rescale_images
        cls.upsample_stride = args.dental_upsample
        cls.min_kp_anns = args.dental_min_kp_anns
        cls.b_min = args.dental_bmin
        if args.dental_use_24_kps:
            (cls.dental_keypoints, cls.dental_skeleton, cls.hflip, cls.dental_sigmas, cls.dental_pose,
             cls.dental_categories, cls.dental_score_weights) = get_constants(24)
        # evaluation
        cls.eval_annotation_filter = args.dental_eval_annotation_filter
        cls.eval_long_edge = args.dental_eval_long_edge
        cls.eval_orientation_invariant = args.dental_eval_orientation_invariant
        cls.eval_extended_scale = args.dental_eval_extended_scale

    def _preprocess(self):
        encoders = (openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                             1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            # openpifpaf.transforms.AnnotationJitter(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(self.dental_keypoints, self.hflip), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.MinSize(min_side=32.0),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    self.dental_categories,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(self.dental_categories),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        # TODO: make sure that 24kp flag is activated when evaluating a 24kp model
        if COCO is None:
            return []
        return [openpifpaf.metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=self.dental_sigmas,
        ), MeanPixelError()]
