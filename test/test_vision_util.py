import unittest

from dataset.cifar10 import CIFAR10DataSet
from vision.util import show_images, make_image

class Test_CIFAR10(unittest.TestCase):
    def test_cifar10_show_images(self):
        c = CIFAR10DataSet()
        dl = c.train_dataloader(batch_size=8)
        imgs,labels = c.get_batch(dl)
        show_images(imgs)