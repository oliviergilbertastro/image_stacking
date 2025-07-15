import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from galight_modif.tools.measure_tools import search_local_max, find_loc_max

folder = r"C:\Users\lauri\Desktop\SharpCap Captures\2025-07-11\M57\23_40_59"+"\\"
img = np.array(imread(f"{folder}M57_00004.tif"))

dark = np.array(imread(f"C:\Users\lauri\Desktop\SharpCap Captures\2025-07-11\M57\23_40_59\MasterDark_ISO0_0s.tif"))

gray_img = np.mean(img, axis=2)
plt.imshow(gray_img)
plt.show()

plt.imshow(np.diff(gray_img))
plt.show()

#search_local_max(gray_img, radius=120, view=True)
#plt.show()

starsx, starsy = find_loc_max(gray_img, threshold=50, neighborhood_size=15)


plt.imshow(img)
for i in range(len(starsx)):
    print(f"{i} : ({starsx[i]},{starsy[i]})")
    plt.plot(starsx[i], starsy[i], "o", color="red")
plt.show()