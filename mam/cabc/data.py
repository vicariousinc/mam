import imageio
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.spatial.distance import cdist
from skimage.measure import label
from skimage.morphology import convex_hull_image
from skimage.transform import resize


def dilate_image(image, iterations=6):
    s = np.zeros((3, 3))
    s[1, :3] = 1
    s[:3, 1] = 1
    image = binary_dilation(image, s, iterations=iterations).astype(np.uint8)
    return image


def load_image(fname):
    img = imageio.imread(fname)
    data = dict(img=img, letter0=img[..., 0], letter1=img[..., 1], fname=fname)
    return data


def get_marker_mask(img, marker_threshold=32, target_shape=None):
    masks = np.logical_and(img > marker_threshold, img < 255).astype(np.int32)
    labels = label(masks, connectivity=2, background=0)
    if len(np.unique(labels)) != 3:
        for n_iter in range(1, 5):
            masks = dilate_image(
                np.logical_and(img > 0, img < 255).astype(np.int32), iterations=2
            )
            labels = label(masks, connectivity=2, background=0)
            if len(np.unique(labels)) == 3:
                break

    if not len(np.unique(labels)) == 3:
        for backup_threshold in [120, 100]:
            print(
                f'Failed to identify markers with threshold {marker_threshold}. Changing threshold to {backup_threshold}'
            )
            masks = np.logical_and(img > backup_threshold, img < 255).astype(np.int32)
            labels = label(masks, connectivity=2, background=0)
            if len(np.unique(labels)) == 3:
                break

    if not len(np.unique(labels)) == 3:
        print(
            f'Failed to identify two unique markers. Ended up with {len(np.unique(labels))} markers'
        )

    marker_mask_list = [
        convex_hull_image(labels == ii).astype(np.float32) for ii in [1, 2]
    ]
    if target_shape is not None:
        marker_mask_list = [
            resize(marker_mask, target_shape, anti_aliasing=True) > 0
            for marker_mask in marker_mask_list
        ]

    return marker_mask_list


def assign_marker_to_letter(marker_mask_list, letter_mask_list):
    assignment = np.zeros(len(marker_mask_list), dtype=int)
    for ii in range(len(marker_mask_list)):
        overlap_size = np.array(
            [
                np.sum(np.logical_and(marker_mask_list[ii], letter_mask))
                for letter_mask in letter_mask_list
            ]
        )
        if np.all(overlap_size == 0):
            assignment[ii] = np.argmin(
                [
                    np.min(
                        cdist(
                            np.argwhere(marker_mask_list[ii]), np.argwhere(letter_mask)
                        )
                    )
                    for letter_mask in letter_mask_list
                ]
            )
            #  assignment[ii] = len(letter_mask_list)
        else:
            assignment[ii] = np.argmax(overlap_size)

    return assignment
