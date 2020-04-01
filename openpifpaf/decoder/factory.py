import logging

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS
from . import generator
from .field_config import FieldConfig
from .caf_scored import CafScored
from .cif_hr import CifHr
from .cif_seeds import CifSeeds
from .processor import Processor
from .. import visualizer

LOG = logging.getLogger(__name__)


def cli(parser, *,
        force_complete_pose=True,
        seed_threshold=0.2,
        instance_threshold=0.0,
        keypoint_threshold=None,
        workers=None):
    group = parser.add_argument_group('decoder configuration')
    group.add_argument('--seed-threshold', default=seed_threshold, type=float,
                       help='minimum threshold for seeds')
    group.add_argument('--instance-threshold', type=float,
                       default=instance_threshold,
                       help='filter instances by score')
    group.add_argument('--keypoint-threshold', type=float,
                       default=keypoint_threshold,
                       help='filter keypoints by score')
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')
    group.add_argument('--dense-connections', default=False, action='store_true',
                       help='use dense connections')
    group.add_argument('--dense-coupling', default=0.01, type=float,
                       help='dense coupling')
    group.add_argument('--caf-seeds', default=False, action='store_true',
                       help='[experimental]')

    if force_complete_pose:
        group.add_argument('--no-force-complete-pose', dest='force_complete_pose',
                           default=True, action='store_false')
    else:
        group.add_argument('--force-complete-pose', dest='force_complete_pose',
                           default=False, action='store_true')

    group.add_argument('--profile-decoder', default=None, action='store_true',
                       help='profile decoder')

    group = parser.add_argument_group('CifCaf decoders')
    group.add_argument('--cif-th', default=CifHr.v_threshold, type=float,
                       help='cif threshold')
    group.add_argument('--caf-th', default=CafScored.default_score_th, type=float,
                       help='caf threshold')
    group.add_argument('--connection-method',
                       default=generator.CifCaf.connection_method,
                       choices=('max', 'blend'),
                       help='connection method to use, max is faster')
    group.add_argument('--greedy', default=False, action='store_true',
                       help='greedy decoding')


def configure(args):
    # configure CifHr
    CifHr.v_threshold = args.cif_th

    # configure CifSeeds
    CifSeeds.threshold = args.seed_threshold

    # configure CafScored
    CafScored.default_score_th = args.caf_th

    # configure debug visualizer
    CifSeeds.debug_visualizer = visualizer.Seeds()
    CifHr.debug_visualizer = visualizer.CifHr(keypoints=COCO_KEYPOINTS)
    Processor.debug_visualizer = visualizer.Occupancy(keypoints=COCO_KEYPOINTS)
    generator.CifCaf.debug_visualizer = visualizer.Occupancy(keypoints=COCO_KEYPOINTS)

    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    # check consistency
    if args.force_complete_pose:
        assert args.keypoint_threshold == 0.0
    assert args.seed_threshold >= args.keypoint_threshold

    # configure decoder generator
    generator.CifCaf.force_complete = args.force_complete_pose
    generator.CifCaf.keypoint_threshold = args.keypoint_threshold
    generator.CifCaf.greedy = args.greedy
    generator.CifCaf.connection_method = args.connection_method

    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       not args.debug:
        args.decoder_workers = args.batch_size


def factory_from_args(args, model, device=None):
    configure(args)

    decode = factory_decode(model,
                            dense_coupling=args.dense_coupling,
                            dense_connections=args.dense_connections,
                            caf_seeds=args.caf_seeds,
                            multi_scale=args.multi_scale,
                            multi_scale_hflip=args.multi_scale_hflip)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     profile=args.profile_decoder,
                     worker_pool=args.decoder_workers,
                     device=device)


def factory_decode(model, *,
                   dense_coupling=0.0,
                   dense_connections=False,
                   caf_seeds=False,
                   multi_scale=False,
                   multi_scale_hflip=True,
                   **kwargs):
    """Instantiate a decoder."""
    assert not caf_seeds, 'not implemented'

    head_names = tuple(model.head_names)
    LOG.debug('head names = %s', head_names)

    if head_names in (('cifdet',),):
        return CifDet(model.head_strides[0], head_index=0, **kwargs)

    if head_names in (('cif', 'caf', 'caf25'),):
        field_config = FieldConfig()

        if multi_scale:
            if not dense_connections:
                field_config.cif_indices = [v * 3 for v in range(5)]
                field_config.caf_indices = [v * 3 + 1 for v in range(5)]
            else:
                field_config.cif_indices = [v * 2 for v in range(5)]
                field_config.caf_indices = [v * 2 + 1 for v in range(5)]
            field_config.cif_strides = [model.head_strides[i] for i in field_config.cif_indices]
            field_config.caf_strides = [model.head_strides[i] for i in field_config.caf_indices]
            field_config.cif_min_scales = [0.0, 12.0, 16.0, 24.0, 40.0]
            field_config.caf_min_distances = [v * 3.0 for v in field_config.cif_min_scales]
            field_config.caf_max_distances = [160.0, 240.0, 320.0, 480.0, None]
        if multi_scale and multi_scale_hflip:
            if not dense_connections:
                field_config.cif_indices = [v * 3 for v in range(10)]
                field_config.caf_indices = [v * 3 + 1 for v in range(10)]
            else:
                field_config.cif_indices = [v * 2 for v in range(10)]
                field_config.caf_indices = [v * 2 + 1 for v in range(10)]
            field_config.cif_strides = [model.head_strides[i] for i in field_config.cif_indices]
            field_config.caf_strides = [model.head_strides[i] for i in field_config.caf_indices]
            field_config.cif_min_scales *= 2
            field_config.caf_min_distances *= 2
            field_config.caf_max_distances *= 2

        if dense_connections:
            field_config.confidence_scales = (
                [1.0 for _ in COCO_PERSON_SKELETON] +
                [dense_coupling for _ in DENSER_COCO_PERSON_CONNECTIONS]
            )
            skeleton = COCO_PERSON_SKELETON + DENSER_COCO_PERSON_CONNECTIONS
        else:
            skeleton = COCO_PERSON_SKELETON

        field_config.cif_visualizers = [
            visualizer.Cif(model.head_names[i],
                           stride=field_config.cif_strides[i],
                           keypoints=COCO_KEYPOINTS, skeleton=skeleton)
            for i, stride in zip(field_config.cif_indices, field_config.cif_strides)
        ]
        field_config.caf_visualizers = [
            visualizer.Caf(model.head_names[i],
                           stride=stride,
                           keypoints=COCO_KEYPOINTS, skeleton=skeleton)
            for i, stride in zip(field_config.caf_indices, field_config.caf_strides)
        ]

        return generator.CifCaf(
            field_config,
            keypoints=COCO_KEYPOINTS,
            skeleton=skeleton,
            out_skeleton=COCO_PERSON_SKELETON,
            **kwargs
        )

    raise Exception('decoder unknown for head names: {}'.format(head_names))
