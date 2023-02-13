import numpy as np
import torch

from PIL.Image import fromarray
from IPython import get_ipython



def display_np_arrays_as_images():
    def np_to_png(a):
        if 2 <= len(a.shape) <= 3:
            return fromarray(np.array(np.clip(a, 0, 1) * 255, dtype='uint8'))._repr_png_()
        else:
            return fromarray(np.zeros([1, 1], dtype='uint8'))._repr_png_()
    get_ipython().display_formatter.formatters['image/png'].for_type(np.ndarray, np_to_png)

def display_tensors_as_images():
    def tensor_to_png(t):
        if 2 <= len(t.shape) <= 3:
            return fromarray(np.array(t.numpy(), dtype='uint8'))._repr_png_()
        else:
            return fromarray(np.zeros([1, 1], dtype='uint8'))._repr_png_()
    get_ipython().display_formatter.formatters['image/png'].for_type(torch.Tensor, tensor_to_png)


        