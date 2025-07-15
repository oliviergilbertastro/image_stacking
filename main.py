"""
Main module to stack images in the lights and darks folders.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from utils import find_loc_max
import os

lights_folder = r"lights/"
darks_folder = r"darks/"

lights_paths = [f for f in os.listdir(lights_folder) if os.path.isfile(os.path.join(lights_folder, f))]
darks_paths = [f for f in os.listdir(darks_folder) if os.path.isfile(os.path.join(darks_folder, f))]
print(lights_paths)
img = np.array(imread(f"{lights_folder}{lights_paths[0]}"))
img = np.array(imread(f"{darks_folder}{darks_paths[0]}"))



gray_img = np.mean(img, axis=2)
plt.imshow(gray_img)
plt.show()

plt.imshow(np.diff(gray_img))
plt.show()

#search_local_max(gray_img, radius=120, view=True)
#plt.show()

starsx, starsy = find_loc_max(gray_img, threshold=20, neighborhood_size=15)


plt.imshow(img)
for i in range(len(starsx)):
    print(f"{i} : ({starsx[i]},{starsy[i]}) : {np.max(img[int(starsy[i]),int(starsx[i])])}")
    plt.plot(starsx[i], starsy[i], "o", color="red")
plt.show()