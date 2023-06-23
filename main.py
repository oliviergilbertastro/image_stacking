import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from astropy.io import fits
import skimage

fits_img_list = []
img_list = []

path = r'C:\Users\Logi\OneDrive - Université Laval\Desktop\OlivierCode\Astro\M27_color\pipp_20230623_124025\M27_00001'

for i in range(1, 10):
    fits_img_list.append(fits.open(f'{path}\M2700{i}.fit')[0])
for i in range(10, 100):
    fits_img_list.append(fits.open(f'{path}\M270{i}.fit')[0])
for i in range(100, 103):
    fits_img_list.append(fits.open(f'{path}\M27{i}.fit')[0])

for i in fits_img_list:
    img_list.append(i.data)

for i in range(len(img_list)):
    img_list[i] = np.stack((img_list[i][0],img_list[i][1],img_list[i][2]), axis=2)/255

#we'll create an offset list built like [(dx0, dy0), (dx1, dy1), ...] to make all the images perfectly on top of each other
offset_list = []
for i in range(102):
    offset_list.append([0, 0])
offset_list[-2] = [8, 4]

#we'll make a list of big images of zeros and then replace the values so all the images are aligned
alist = []
for i in range(102):
    alist.append(np.zeros((2000, 2000, 3)))
    alist[i][500+offset_list[i][0]:1476+offset_list[i][0], 500+offset_list[i][1]:1804+offset_list[i][1], :] = img_list[i]
#alist[-1][500:500+976, 500:500+1304, :] = img_list[-1]
#alist[-2][508:508+976, 504:504+1304, :] = img_list[-2]
#THE SHAPE IS (976, 1304)

#print(fits_img_list[-1].header)


#530, 613
#526, 605
#-4, -8

image = img_list[-1]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image)
ax[1].hist(image.flatten(), bins=255, color='black')
ax[1].set_xlabel('Intensity')
ax[1].set_ylabel('Counts')
plt.show()
#plt.imshow(image)
#channels = ['R', 'G', 'B']
#fig, ax = plt.subplots(1, 3, figsize=(15, 10))
#for i in range(len(ax)):
#    ax[i].imshow(image[:, :, i], cmap='gray')
#    ax[i].axis('off')
#    ax[i].set_title(channels[i])
#plt.show()


print('Making the masterframe...')
#masterFrame = np.median(alist, axis=0)
if False:
    plt.imshow(masterFrame)
    plt.show()