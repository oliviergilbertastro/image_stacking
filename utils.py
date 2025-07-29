

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


def find_loc_max(image, neighborhood_size = 8, threshold = 5):
    """
    Find all the local maximazation in a 2D array, used to search the targets such as QSOs and PSFs.
    This function is created and inspired based on:
        https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
   
    Parameter
    --------
        image: 
            2D array type image.
        
        neighborhood_size: digit.
            Define the region size to filter the local minima.
        
        threshold: digit.
            Define the significance (flux value) of the maximazation point. The lower, the more would be found.
    
    Return
    --------
        A list of x and y of the searched local maximazations.
    """    
    data_max = filters.maximum_filter(image, neighborhood_size) 
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return x, y


# Print something in the terminal in a specific color
def print_color(message, color="yellow", **kwargs):
    """print(), but with a color option"""
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    if color == None or color == "grey":
        color = "0"
    elif type(color) == str:
        color = color.lower()
        if color in possible_colors:
            color = str(possible_colors.index(color)+30)
        else:
            print(f"Color '{color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    print(f"\x1b[{color}m{message}\x1b[0m", **kwargs)

def plot_img(img, **kwargs):
    fig, ax = plt.subplots()
    plt.imshow(img, **kwargs)
    plt.axis('off')
    fig.subplots_adjust(0,0,1,1)