from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class CIFAR10DataSet():
    def __init__(self, data_dir="data/cifar10", train_val_split=0.8):
        self.data_dir = data_dir
        self.dataset = CIFAR10(root=self.data_dir, download=True)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.train_val_split = train_val_split

    def  train_dataloader(self, batch_size=32, resize=32, p=0.5, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), num_workers=4):
        tf = T.Compose([
                T.RandomResizedCrop(size=resize),
                T.RandomHorizontalFlip(p=p),
                T.RandomVerticalFlip(p=p),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        ds = CIFAR10(root=self.data_dir, train=True, transform=tf)
        num_train = len(ds)
        indices = list(range(num_train))
        split = int(np.floor(self.train_val_split * num_train))
        train_sampler = SubsetRandomSampler(indices[split:])
        dl = DataLoader(
            ds,
            batch_size=batch_size, 
            num_workers=num_workers, 
            sampler=train_sampler, 
            drop_last=True
            )
        return dl
    
    def  val_dataloader(self, batch_size=32, resize=32, p=0.5, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), num_workers=4):
        tf = T.Compose([
                T.RandomResizedCrop(size=resize),
                T.RandomHorizontalFlip(p=p),
                T.RandomVerticalFlip(p=p),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        ds = CIFAR10(root=self.data_dir, train=True, transform=tf)
        num_train = len(ds)
        indices = list(range(num_train))
        split = int(np.floor(self.train_val_split * num_train))
        val_sampler = SubsetRandomSampler(indices[:split])
        dl = DataLoader(
            ds,
            batch_size=batch_size, 
            num_workers=num_workers, 
            sampler=val_sampler, 
            drop_last=True
            )
        return dl

    
    def test_dataloader(self, batch_size=32, mean=(0.485, 0.456, 0.406) ,std=(0.229, 0.224, 0.225), num_workers=4):
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            ]
        )
        ds = CIFAR10(root=self.data_dir, train=False, transform=tf)
        dl = DataLoader(ds,batch_size=batch_size, num_workers=num_workers, drop_last=True)
        return dl
    
    def get_batch(self, dataloader):
        return next(iter(dataloader))
    
    def get_classes(self):
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return classes

