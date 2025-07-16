"""Utility functions"""

import numpy as np
from utils import find_loc_max

def combine(img_list:list) -> np.ndarray:
    """Combines images into a single image"""
    imgs = np.array(img_list)
    return np.median(imgs, axis=0, out=np.ndarray(shape=img_list[0].shape, dtype=int))

def make_bad_pixel_mask(master_dark:np.ndarray, threshold=30) -> np.ndarray:
    mask_rgb = master_dark<threshold
    mask = np.all(mask_rgb, axis=-1)
    return np.array(mask, dtype=int)

def find_stars(img:np.ndarray, bad_pixel_mask:np.ndarray=None, threshold:float=50., neighborhood_size:int=15) -> list:
    """
    Finds stars in the field of view and returns a list of their 2D position (x,y).
    """
    if len(img.shape) > 2: img = np.mean(img, axis=-1) # convert to grayscale if not already
    if bad_pixel_mask is not None: img = np.where(bad_pixel_mask, img, np.nan) # remove the bad pixels so they're not considered as stars
    starsx, starsy = find_loc_max(img, threshold=threshold, neighborhood_size=neighborhood_size)
    stars_pos_list = [np.array([starsx[i], starsy[i]]) for i in range(len(starsx))]
    return stars_pos_list

def get_stars_relative_positions(stars_pos_list:list) -> list:
    """
    Compute the relative position between each star in the field of view to prepare for alignment. (norm, angle)
    """
    relative_positions_list = []
    for i in range(len(stars_pos_list)):
        relative_positions_list.append([])
        for k in range(len(stars_pos_list)-1):
            vec = stars_pos_list[(k+i+1)%len(stars_pos_list)]-stars_pos_list[i]
            relative_positions_list[-1].append((np.linalg.norm(vec), np.arctan2(vec[1],vec[0])))
    return relative_positions_list