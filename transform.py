"""Utility functions"""

import numpy as np
from utils import find_loc_max
from scipy.ndimage import rotate
import copy
import matplotlib.pyplot as plt
from utils import print_color

def combine(img_list:list) -> np.ndarray:
    """Combines images into a single image"""
    imgs = np.array(img_list)
    return np.median(imgs, axis=0, out=np.ndarray(shape=img_list[0].shape, dtype=int))

def stack(img_list:list, translations:list, angles:list) -> np.ndarray:
    img_list_rot_transl = []
    hh, ww = img_list[0].shape
    canvas = np.empty((ww*3, hh*3))
    origin = (hh, ww)
    for i in range(len(img_list)):
        img = img_list[i]
        img_transl = copy.copy(canvas)
        y_transl, x_transl = origin[0]+translations[i][0], origin[1]+translations[i][1]
        img_transl[y_transl:y_transl+hh, x_transl:x_transl+ww]
        img_rot_transl = rotate(input=img_transl, angle=angles[i])
        img_list_rot_transl.append(img_rot_transl)
    return combine(img_list_rot_transl)


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

def associate_stars(all_relative_stars_pos_list:list, tolerance:float=0.05) -> list:
    """From the relative positions between each star in each image, create a list of all the different stars and find the ones imaged more than once."""

    associated_dict = {}
    # Go through every image
    for nth_light in range(len(all_relative_stars_pos_list)):
        # Go through every star in this image
        for nth_star in range(len(all_relative_stars_pos_list[nth_light])):
            # Go through every image to try and find if the same star is in there
            print()

    pass

class ReferenceImage:
    """Light image that is sharp and contains star positions."""
    def __init__(self, light_img:np.ndarray):
        self.light_img:np.ndarray = light_img
        self.gray_img:np.ndarray = light_img

    def choose_stars(self):
        """This function provides an interactive interface to add stars."""
        print_color("\nPressing ENTER without typing anything shows the image along with the stars you've added so far." \
                    "\nTo add a star, enter coordinates such 'X,Y' and press ENTER." \
                    "\nTo remove a star, enter 'del X,Y'." \
                    "\nTo save and exit, enter 'q' and press ENTER."
        )
        running = True
        star_pos_list = []
        while running:
            inp = input(">>> ")
            if (inp.lower())[:3] == "del":
                coords = np.array([float(s) for s in inp[4:].split(",")])
                star_pos_list.pop(star_pos_list.index(coords))
                print_color(f"Star succesfully removed from {coords}", color="red")
            elif inp == "":
                self.show(star_list=star_pos_list)
            elif inp.lower() in ["q","quit","exit"]:
                running = False
            else:
                coords = np.array([float(s) for s in inp.split(",")])
                star_pos_list.append(coords)
                print_color(f"Star succesfully added at {coords}", color="green")
        print_color(f"Final star_pos_list = {star_pos_list}", color="magenta")
        return star_pos_list

    def find_stars(self, bad_pixel_mask:np.ndarray=None, star_pos_list:list=None):
        self.found_stars = find_stars(self.light_img, bad_pixel_mask)
        self.found_rel_pos = get_stars_relative_positions(self.found_stars)
        found_lenghts = [np.array(self.found_rel_pos[i])[:,0] for i in range(len(self.found_rel_pos))]
        self.star_pos_list:list = star_pos_list if star_pos_list is not None else self.found_stars
        self.rel_pos = get_stars_relative_positions(self.star_pos_list)
        self.rel_lenghts = [np.array(self.rel_pos[i])[:,0] for i in range(len(self.rel_pos))]

        # Check if all given star positions are found automatically:
        all_found = True
        if not all_found:
            print("Not all stars you've input are found automatically. This might cause a bad alignment if you do not change them.")
        
    def assign_stars(self, star_pos_list:list, tolerance:float=5) -> list:
        """
        Returns a list of indices that correspond to the nth-star in self.star_pos_list
        
        e.g. If self.star_pos_list is [(24,58), (93,10)], and star_pos_list is [(25.2,56), (34.2, 29), (92.8,9)],
             then the resulting list would be [0,2].
        """
        outlist = []
        _rel_pos = get_stars_relative_positions(star_pos_list)
        _lenghts = [np.array(_rel_pos[i])[:,0] for i in range(len(_rel_pos))]
        for i, lengths in enumerate(self.rel_lenghts): # Iterate over each ref. star
            # Search for a similar length in the list:
            for k, other_lengths in enumerate(_lenghts):
                certainty = 0
                for length in lengths:
                    if np.any(np.abs(other_lengths-length)/length < tolerance/100): # Tolerance is in percentage
                        certainty += 1
                if certainty > 1:
                    # At least two distances are the same, we can consider the star is a match (the same)
                    if len(outlist) > i: raise ValueError("More than a star was found to match, remove this image from alignment.")
                    outlist.append(k)
            if len(outlist) == i:
                outlist.append(None)
        return outlist
            


    

    def show(self, star_list:list=None):
        if star_list is None: star_list = self.star_pos_list
        plt.imshow(self.light_img)
        for i in range(len(star_list)):
            plt.plot(star_list[i][0], star_list[i][1], "o", color="red")
        plt.show()
        pass