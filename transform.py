"""Utility functions"""

import numpy as np

def combine(img_list:list) -> np.ndarray:
    """Combines images into a single image"""
    imgs = np.array(img_list)
    return np.median(imgs, axis=3)