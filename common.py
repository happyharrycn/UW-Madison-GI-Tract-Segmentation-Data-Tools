# only support python 3.x
import os
import glob
import argparse

# NumPy, Pandas, surface_distance
import numpy as np
import pandas as pd

from surface_distance import compute_surface_distances

def get_blacklist_case_ids():
    """
    These scans has some known issues in their annotations. They are thus
    exclued in our evalauation.
    Returns:
        list of case ids
    """
    return [
        "case61_day14",
        "case9_day27",
        "case7_day0",
        "case81_day30",
        "case43_day18",
        "case138_day0",
        "case34_day20",
        "case43_day26",
        "case124_day19",
        "case144_day15",
        "case83_day11",
        "case13_day0"
    ]

def get_organ_types():
    """
    Types of organs considerd in the dataset
    Returns:
        list of organ types
    """
    return ["small_bowel", "large_bowel", "stomach"]

def get_default_spacing():
    """
    The default spacing in x, y, z axis (followed by most of the scans)
    Returns:
        dict of the spacing
    """
    return {'x': 1.5, 'y': 1.5, 'z': 3}

def rle_encode(img):
    '''
    Helper function for run-length encoding
    ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    Inputs:
        img: (binary) numpy array, 1 - mask, 0 - background
    Returns:
        string formated RLE
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, dtype=np.uint8):
    '''
    Helper function for run-length decoding
    ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    Inputs:
        mask_rle: string formated RLE
        shape: (height,width) of array to return
        dtype: data type of array to return (default: uint8)
    Returns:
        numpy array, 1 as mask, 0 as background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=dtype)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def compute_3d_hausdorff_distance(mask_gt, mask_pred, spacing_mm):
    """
    Compute the 3D Hausdoff distance between a gt mask and a predicted mask.
    Both masks lies in the 3D space and must be non-empty.
    Inputs:
        mask_gt: 3D bool Numpy array (D, H, W). The ground truth binary mask.
        mask_pred: 3D bool Numpy array (D, H, W). The predicted binary mask.
        spacing_mm: 3-element list-like structure. Voxel spacing in (D, H, W)
    Returns:
        Hausdoff distance between the masks

    Note: The Kaggle evaluation code normalizes all distances into the range of
    [0, 1], and ignores those scans without ground-truth masks.
    """
    assert (mask_gt > 0).any(), "GT mask must be non-empty"
    assert (mask_pred > 0).any(), "Predicted mask must be non-empty"

    output_dict = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
    distances_gt_to_pred = output_dict["distances_gt_to_pred"]
    distances_pred_to_gt = output_dict["distances_pred_to_gt"]
    hd = max(np.max(distances_gt_to_pred), np.max(distances_pred_to_gt))
    return hd

def compute_3d_dice_score(mask_gt, mask_pred):
    """
    Compute the 3D Dice score give a gt mask and a predicted mask.
    Both masks lies in the 3D space.
    Inputs:
        mask_gt: 3D bool Numpy array (D, H, W). The ground truth binary mask.
        mask_pred: 3D bool Numpy array (D, H, W). The predicted binary mask.
    Returns:
        Dice score of the predicted mask

    Note: The Kaggle evaluation code adapts a 2D approxmiation of the 3D Dice scores,
    and ignores those scans without ground-truth masks.
    """
    dice = (
        2 * np.sum(np.logical_and(mask_gt, mask_pred))
        / (np.sum(mask_gt) + np.sum(mask_pred))
    )
    return dice

def process_img(img, gamma=.6):
    '''
    Adjust the contrast and normalize the pixel values.
    Inputs:
        img: numpy array of an image (np.float32)
        gamma: used for gamma correction
    Returns:
        numpy array of an image
    '''
    img = img.astype(np.float32)
    img = img ** gamma
    img = img / np.max(img)
    return img
