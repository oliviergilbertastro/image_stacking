"""
Main module to stack images in the lights and darks folders.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
from transform import *
from tqdm import tqdm
from utils import print_color, plot_img

lights_folder = r"lights/"
darks_folder = r"darks/"

lights_paths = [f for f in os.listdir(lights_folder) if os.path.isfile(os.path.join(lights_folder, f))]
darks_paths = [f for f in os.listdir(darks_folder) if os.path.isfile(os.path.join(darks_folder, f))]
lights = [np.array(imread(f"{lights_folder}{lights_paths[i]}")) for i in range(len(lights_paths))]
darks = [np.array(imread(f"{darks_folder}{darks_paths[i]}")) for i in range(len(darks_paths))]

master_dark = combine(darks)
bp_mask = make_bad_pixel_mask(master_dark)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9,5))
axes[0].imshow(master_dark)
axes[1].imshow(bp_mask, vmin=0, vmax=1)
axes[0].set_title("Master dark", fontsize=17)
axes[1].set_title("Bad pixel mask", fontsize=17)
axes[0].set_xlabel("$x$ [px]", fontsize=17)
axes[0].set_ylabel("$y$ [px]", fontsize=17)
axes[1].set_xlabel("$x$ [px]", fontsize=17)
axes[1].set_ylabel("$y$ [px]", fontsize=17)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.show()

ref_img = ReferenceImage(lights[3])
star_pos_list = None
if input("Type 'Y' to select stars on reference image manually.\n") == "Y":
    star_pos_list = ref_img.choose_stars()
ref_img.find_stars(bad_pixel_mask=bp_mask, star_pos_list=star_pos_list) # Leave star_pos_list to None to find the stars automatically
print(f"Star pos in reference image: {ref_img.star_pos_list}")
ref_img.show()
outlist, angle_outlist = ref_img.assign_stars(star_pos_list=[np.array([612., 438.]), np.array([111., 854.]), np.array([512., 269.])], tolerance=5)


all_star_pos_list = []
all_relative_pos_list = []
good_light_indices = []
good_assignations = []
good_angles= []
good_star_positions = []
for s in tqdm(range(len(lights))):
    if lights[s].shape != ref_img.light_img.shape:
        print_color(f"image is of a bad shape: {lights[s].shape} instead of {ref_img.light_img.shape}", color="red")
    stars = find_stars(lights[s], bad_pixel_mask=bp_mask)
    if len(stars) > 1:
        try:
            outlist, angle_outlist = ref_img.assign_stars(star_pos_list=stars, tolerance=2)
            if np.any([x is not None for x in outlist]):
                good_light_indices.append(s)
                good_assignations.append(outlist)
                good_angles.append(angle_outlist)
                good_star_positions.append(stars)
        except Exception as e:
            #print(e)
            pass
print_color(f"Final count: {len(good_light_indices)} images ready to be aligned. First is {good_light_indices[0]}")
print(len(good_star_positions), len(good_assignations))

# Calculate the translations/rotations
ref_idx, translations, rotations = ref_img.get_translations_and_rotations(good_star_positions, good_assignations, good_angles)
print(ref_idx)
align_pos_list = [(good_star_positions[i])[ref] for i, ref in enumerate(ref_idx)]
good_lights = [lights[i] for i in good_light_indices]
stacked = stack(good_lights, translations, rotations, align_pos_list)
plot_img(stacked)
plt.show()
good_lights = [good_lights[i] for i in range(len(good_lights)) if ref_idx[i] == 1]
translations = [translations[i] for i in range(len(translations)) if ref_idx[i] == 1]
rotations = [rotations[i] for i in range(len(rotations)) if ref_idx[i] == 1]
stacked = stack_add(good_lights, translations, rotations)
plot_img(stacked)
plt.show()

"""plt.imshow(lights[s])
for i in range(len(stars)):
    #print(f"{i} : {stars[i]} -> {[np.around(relative_pos_stars[i][k], decimals=2) for k in range(len(relative_pos_stars[i]))]}")
    plt.plot(stars[i][0], stars[i][1], "o", color="red")
plt.show()"""