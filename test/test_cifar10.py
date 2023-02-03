import unittest

from dataset.cifar10 import CIFAR10DataSet

class Test_CIFAR10(unittest.TestCase):
    def test_cifar10_instantiate(self):
        c = CIFAR10DataSet()
        print("hello")
        assert(c != None)

    def test_cifar10_train_dataloader(self):
        c = CIFAR10DataSet()
        dl = c.train_dataloader()
        size = len(dl.dataset)
        print("Size of dataset: ", size)
        for batch, (imgs, labels) in enumerate(dl):
            print(batch, imgs.shape, labels)


