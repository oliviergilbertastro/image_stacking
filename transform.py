"""Utility functions"""

import numpy as np
from utils import find_loc_max
from scipy.ndimage import rotate
import copy
import matplotlib.pyplot as plt
from utils import print_color
from tqdm import tqdm

def combine(img_list:list, axis=0) -> np.ndarray:
    """Combines images into a single image"""
    imgs = np.array(img_list)
    return np.median(imgs, axis=axis, out=np.ndarray(shape=img_list[0].shape, dtype=int))

def transform_fn(x:float|np.ndarray, y:float|np.ndarray, theta:float, shape:tuple) -> tuple :
    height, width = shape
    theta = theta - np.floor(theta / (2*np.pi))*2*np.pi # Make it so theta is between 0 and 2*np.pi
    if theta <= np.pi/2:
        x_ = x*np.cos(theta)+y*np.sin(theta)
        y_ = y*np.cos(theta)-x*np.sin(theta)+width*np.sin(theta)
    elif (theta > np.pi/2) and (theta <= np.pi):
        x_ = x*np.cos(theta)+y*np.sin(theta)+width*np.sin(theta-np.pi/2)
        y_ = y*np.cos(theta)-x*np.sin(theta)+width*np.sin(theta)+height*np.sin(theta-np.pi/2)
    elif (theta > np.pi) and (theta <= 3*np.pi/2):
        x_ = x*np.cos(theta)+y*np.sin(theta)+width*np.cos(theta-np.pi)+height*np.sin(theta-np.pi)
        y_ = y*np.cos(theta)-x*np.sin(theta)+height*np.cos(theta-np.pi)
    else:
        x_ = x*np.cos(theta)+y*np.sin(theta)+height*np.cos(theta-3*np.pi/2)
        y_ = y*np.cos(theta)-x*np.sin(theta)
    return (x_, y_)

def stack(img_list:list, translations:list, angles:list, align_pos:list) -> np.ndarray:
    """
    align_pos : list of the positions of the alignment stars used to calculate the translation and the angle
    """
    img_list_rot_transl = []
    hh, ww, d = img_list[0].shape
    print(hh, ww, d)
    canvas = np.empty((hh*3, ww*3, d))
    print(canvas.shape)
    origin = (hh, ww)
    for i in tqdm(range(len(img_list))):
        img = img_list[i]
        img_transl = copy.copy(canvas)
        # Shift to correct for the translation
        y_transl, x_transl = origin[0]+translations[i][0], origin[1]+translations[i][1]
        #           img_transl[int(y_transl):int(y_transl+hh), int(x_transl):int(x_transl+ww)] = img
        # Rotate
        img_rot = rotate(input=img, angle=angles[i]/np.pi*360)
        new_pos = transform_fn(*(align_pos[i]), theta=angles[i], shape=(hh,ww))
        # Re-shift to correct for the rotation
        rotation_translation = np.array(new_pos) - np.array(align_pos[i])
        y_transl, x_transl = origin[0]+translation[1], origin[1]+translation[0]
        img_rot_transl[int(y_transl):int(y_transl+hh), int(x_transl):int(x_transl+ww)] = img_rot_transl
        img_list_rot_transl.append(img_rot_transl)
    return combine(img_list_rot_transl, axis=0)

def stack_add(img_list:list, translations:list, angles:list) -> np.ndarray:
    hh, ww, d = img_list[0].shape
    canvas = np.empty((hh*5, ww*5, d))
    print(hh, ww, d)
    print(canvas.shape)
    origin = (2*hh, 2*ww)
    for i in tqdm(range(len(img_list))):
        img = img_list[i]
        y_transl, x_transl = origin[0]+translations[i][1], origin[1]+translations[i][0]
        canvas[int(y_transl):int(y_transl+hh), int(x_transl):int(x_transl+ww), :] = (canvas[int(y_transl):int(y_transl+hh), int(x_transl):int(x_transl+ww), :] + img)/2
        #plt.imshow(np.array(canvas/255, dtype=float))
        #plt.show()
    return np.array(canvas/255, dtype=float)


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

    def get_translations_and_rotations(self, star_positions:list, star_assignations:list) -> tuple[list, list, list]:
        """
        star_positions : the list of all the positions of the stars
        star_assignation : the list of the indices that correspond to the ref. image stars.
        Returns the translation and the rotation lists.
        """
        assert len(star_positions) == len(star_assignations) # Check that they match just to be sure
        ref_star_idx_list = []
        transl_list = []
        rot_list = []
        for i in range(len(star_positions)):
            ref_star_idx = None
            for k in range(len(star_assignations[i])):
                if (star_assignations[i][k] is not None) and ref_star_idx is None: # Take the first star that was recognized
                    ref_star_idx = k
                    this_star_idx = star_assignations[i][k]
            try:
                transl = np.array(self.star_pos_list[ref_star_idx]) - np.array(star_positions[i][this_star_idx])
            except Exception as e:
                print(e)
                print(star_assignations[i])
            rot = 0
            ref_star_idx_list.append(ref_star_idx)
            transl_list.append(transl)
            rot_list.append(rot)
        return ref_star_idx_list, transl_list, rot_list

    def show(self, star_list:list=None):
        if star_list is None: star_list = self.star_pos_list
        plt.imshow(self.light_img)
        for i in range(len(star_list)):
            plt.plot(star_list[i][0], star_list[i][1], "o", color="red")
        plt.show()
        pass