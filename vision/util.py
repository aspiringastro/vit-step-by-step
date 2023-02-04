import torchvision
import torchvision.utils
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np


def make_image(img, mean=(0., 0., 0.), std=(1., 1., 1.)):
    #denormalize
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def show_images(imgs, mean=(0., 0., 0.), std=(1., 1., 1.)):
    grid_imgs = make_grid(imgs)
    grid_imgs = make_image(grid_imgs, mean, std)
    plt.imshow(grid_imgs)
    plt.axis('off')
    
