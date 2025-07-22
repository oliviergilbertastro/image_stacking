"""
Main module to stack images in the lights and darks folders.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
from transform import *

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
ref_img.choose_stars()
ref_img.show()

all_star_pos_list = []
all_relative_pos_list = []
for s in range(len(lights)):
    stars = find_stars(lights[s], bad_pixel_mask=bp_mask)
    print(stars)
    relative_pos_stars = get_stars_relative_positions(stars)
    all_star_pos_list.append(stars)
    all_relative_pos_list.append(relative_pos_stars)
    plt.imshow(lights[s])
    for i in range(len(stars)):
        print(f"{i} : {stars[i]} -> {[np.around(relative_pos_stars[i][k], decimals=2) for k in range(len(relative_pos_stars[i]))]}")
        plt.plot(stars[i][0], stars[i][1], "o", color="red")
    plt.show()