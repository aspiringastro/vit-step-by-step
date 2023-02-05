import torchvision
import torchvision.utils
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.bbox"] = 'tight'

def make_image(img, mean=(0., 0., 0.), std=(1., 1., 1.)):
    #denormalize
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def show_image(imgs, mean=(0., 0., 0.), std=(1., 1., 1.)):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        for j in range(3):
            img[j] = img[j] * std[j] + mean[j]
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_images(imgs, mean=(0., 0., 0.), std=(1., 1., 1.)):
    grid_imgs = make_grid(imgs)
    grid_imgs = make_image(grid_imgs, mean, std)
    plt.imshow(grid_imgs)
    plt.axis('off')

    
