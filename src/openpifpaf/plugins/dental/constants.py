import os

import numpy as np
import openpifpaf

KEYPOINTS = [
    'cej_left',       # 1
    'cej_right',        # 2
    'aeac_left',    # 3
    'aeac_right',     # 4
    'apex_left',      # 5
    'apex_right'        # 6
]

SKELETON = [
    (1, 3), (3, 5), (2, 4), (4, 6)
]

SIGMAS = [0.05] * len(KEYPOINTS)

split, error = divmod(len(KEYPOINTS), 4)
SCORE_WEIGHTS = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(SCORE_WEIGHTS) == len(KEYPOINTS)

HFLIP = {
    'cej_left': 'cej_right',
    'cej_right': 'cej_left',
    'aeac_left': 'aeac_right',
    'aeac_right': 'aeac_left',
    'apex_left': 'apex_right',
    'apex_right': 'apex_left'
}

CATEGORIES = ['tooth']

FRONT = -6.0

# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
POSE = np.array([
    [-2.9, 4.0, FRONT * 0.5],  # 'front_up_right',              # 1
    [2.9, 4.0, FRONT * 0.5],   # 'front_up_left',               # 2
    [-2.0, 2.0, FRONT],  # 'front_light_right',           # 3
    [2.0, 2.0, FRONT],  # 'front_light_left',             # 4
    [-2.5, 0.0, FRONT],  # 'front_low_right',             # 5
    [2.5, 0.0, FRONT]  # 'front_low_left',              # 6
])

def get_constants():
    POSE[:, 2] = 2.0
    return [KEYPOINTS, SKELETON, HFLIP, SIGMAS, POSE, CATEGORIES, SCORE_WEIGHTS]

def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_tooth.png', keypoint_painter=keypoint_painter)


def print_associations():
    print("\nAssociations of the tooth skeleton with 24 keypoints")
    for j1, j2 in SKELETON:
        print(KEYPOINTS[j1 - 1], '-', KEYPOINTS[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(POSE, sigmas = SIGMAS, skel = SKELETON,
#                    kps = KEYPOINTS, scr_weights = SCORE_WEIGHTS)
# =============================================================================


if __name__ == '__main__':
    main()
