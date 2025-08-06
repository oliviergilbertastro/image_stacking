import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import os
from matplotlib.image import imread


lights_folder = r"lights/"
darks_folder = r"darks/"

lights_paths = [f for f in os.listdir(lights_folder) if os.path.isfile(os.path.join(lights_folder, f))]
darks_paths = [f for f in os.listdir(darks_folder) if os.path.isfile(os.path.join(darks_folder, f))]
lights = [np.array(imread(f"{lights_folder}{lights_paths[i]}")) for i in range(len(lights_paths))]
darks = [np.array(imread(f"{darks_folder}{darks_paths[i]}")) for i in range(len(darks_paths))]

img_unrotated = lights[0]
ANGLE = -25
img_rotated = rotate(input=img_unrotated, angle=ANGLE, reshape=True)

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

dots_x = []
dots_y = []
dots_c = []
dots_y_temp = np.linspace(img_unrotated.shape[0]*0.1, img_unrotated.shape[0]*0.9, 15)
for i in range(15):
    dots_x += list(np.linspace(img_unrotated.shape[1]*0.1, img_unrotated.shape[1]*0.9, 15))
    dots_y += [dots_y_temp[i] for k in range(15)]
    dots_c += [i*15+k for k in range(15)]

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)#, sharex=ax1, sharey=ax1)
ax1.imshow(img_unrotated, origin="lower")
ax2.imshow(img_rotated, origin="lower")
pos = (500,400)
#ax1.plot(pos[0],pos[1], marker="o", color="red", ls="None")
ax1.scatter(dots_x, dots_y, c=dots_c, ls="None", marker=".")
pos_ = transform_fn(np.array(dots_x), np.array(dots_y), theta=ANGLE/180*np.pi, shape=img_unrotated.shape[:2])
ax2.scatter(pos_[0], pos_[1], c=dots_c, ls="None", marker=".")
#ax2.plot(pos_[0],pos_[1], marker="o", color="red", ls="None")
plt.show()
